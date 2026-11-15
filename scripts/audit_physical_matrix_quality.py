#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "prefill_seconds",
    "decode_seconds",
    "total_seconds",
    "tokens_per_second",
    "peak_cuda_memory_gb",
    "resident_kv_gb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit independent-process physical benchmark matrices.")
    parser.add_argument("matrix_roots", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--spread-threshold", type=float, default=1.15)
    parser.add_argument("--half-drift-threshold", type=float, default=1.05)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))]


def audit_success(
    cell: Path,
    payload: dict[str, Any],
    spread_threshold: float,
    half_drift_threshold: float,
) -> dict[str, Any]:
    measurements = payload.get("measurements", [])
    issues: list[str] = []
    if len(measurements) != payload.get("repeats"):
        issues.append(f"measurements={len(measurements)} repeats={payload.get('repeats')}")
    expected_decode_steps = max(0, int(payload["output_tokens"]) - 1)
    if payload.get("decode_tokens_timed_per_request") != expected_decode_steps:
        issues.append("decode numerator is not output_tokens - 1")
    retained_ratio = float(payload["retained_ratio_actual"])
    if not 0 < retained_ratio <= 1:
        issues.append("retained ratio outside (0, 1]")

    distributions: dict[str, Any] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in measurements]
        if not values or any(not math.isfinite(value) or value < 0 for value in values):
            issues.append(f"invalid {metric}")
            continue
        p50 = percentile(values, 0.50)
        p95 = percentile(values, 0.95)
        saved_p50 = float(payload["summaries"][metric]["p50"])
        if abs(p50 - saved_p50) > max(1e-9, 1e-7 * abs(p50)):
            issues.append(f"saved P50 mismatch for {metric}")
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        split = max(1, len(values) // 2)
        first_half_median = statistics.median(values[:split])
        second_half_median = statistics.median(values[split:]) if values[split:] else first_half_median
        half_drift_ratio = (
            1.0
            if first_half_median == second_half_median == 0
            else max(first_half_median, second_half_median)
            / max(1e-30, min(first_half_median, second_half_median))
        )
        if half_drift_ratio > half_drift_threshold:
            issues.append(
                f"{metric} half-drift={half_drift_ratio:.6f} exceeds {half_drift_threshold}"
            )
        spread = None if p50 == 0 else p95 / p50
        if spread is not None and spread > spread_threshold:
            issues.append(f"{metric} P95/P50={spread:.6f} exceeds {spread_threshold}")
        distributions[metric] = {
            "min": min(values),
            "p50": p50,
            "p95": p95,
            "max": max(values),
            "p95_over_p50": spread,
            "mad_over_median": None if median == 0 else mad / median,
            "first_half_median": first_half_median,
            "second_half_median": second_half_median,
            "half_drift_ratio": half_drift_ratio,
        }
    return {
        "cell": cell.name,
        "status": "success",
        "measurement_count": len(measurements),
        "issues": issues,
        "distributions": distributions,
    }


def audit_root(root: Path, spread_threshold: float, half_drift_threshold: float) -> dict[str, Any]:
    cells = []
    for cell in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "summary"):
        metrics_path = cell / "metrics.json"
        exit_path = cell / "exit_code"
        log_path = cell / "run.log"
        if metrics_path.exists():
            cells.append(
                audit_success(
                    cell,
                    json.loads(metrics_path.read_text()),
                    spread_threshold,
                    half_drift_threshold,
                )
            )
            continue
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        if exit_path.exists():
            cells.append(
                {
                    "cell": cell.name,
                    "status": "oom" if "OutOfMemoryError" in log_text else "failed",
                    "exit_code": int(exit_path.read_text().strip()),
                    "issues": [] if "OutOfMemoryError" in log_text else ["finished without metrics and no OOM marker"],
                }
            )
        else:
            cells.append({"cell": cell.name, "status": "active_or_partial", "issues": []})

    status_counts: dict[str, int] = {}
    for cell in cells:
        status_counts[cell["status"]] = status_counts.get(cell["status"], 0) + 1
    success = [cell for cell in cells if cell["status"] == "success"]
    worst_spreads = []
    for cell in success:
        for metric, distribution in cell["distributions"].items():
            spread = distribution["p95_over_p50"]
            if spread is not None:
                worst_spreads.append({"cell": cell["cell"], "metric": metric, "p95_over_p50": spread})
    worst_spreads.sort(key=lambda row: row["p95_over_p50"], reverse=True)
    return {
        "matrix_root": str(root),
        "cell_count": len(cells),
        "status_counts": status_counts,
        "cells_with_issues": [cell for cell in cells if cell["issues"]],
        "worst_spreads": worst_spreads[:20],
        "cells": cells,
    }


def main() -> None:
    args = parse_args()
    report = {
        Path(root).name: audit_root(
            Path(root),
            args.spread_threshold,
            args.half_drift_threshold,
        )
        for root in args.matrix_roots
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        name: {
            "cell_count": value["cell_count"],
            "status_counts": value["status_counts"],
            "cells_with_issues": len(value["cells_with_issues"]),
            "worst_p95_over_p50": (
                value["worst_spreads"][0]["p95_over_p50"] if value["worst_spreads"] else None
            ),
        }
        for name, value in report.items()
    }
    print(json.dumps(summary, indent=2))
    if any(value["cells_with_issues"] for value in report.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
