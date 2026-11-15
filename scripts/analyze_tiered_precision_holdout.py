#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen INT8 tiered-KV confirmation analysis.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=541)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def paired_bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    if values.ndim != 1 or len(values) == 0 or draws <= 0:
        raise ValueError("invalid paired bootstrap input")
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    chunk = 4096
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        sampled[start:stop] = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(sampled, [0.025, 0.975])],
        "bootstrap_draws": draws,
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    metrics = json.loads(Path(args.metrics).read_text())
    rows = [
        json.loads(line)
        for line in Path(args.results).read_text().splitlines()
        if line.strip()
    ]
    ids = [str(row.get("id")) for row in rows]
    expected_n = int(metrics.get("max_samples", -1))
    integrity = {
        "exactly_one_int8_row_per_sample": (
            len(rows) == expected_n
            and len(set(ids)) == expected_n
            and all(row.get("precision") == "int8" for row in rows)
        ),
        "frozen_protocol_matches": (
            expected_n == 64
            and int(metrics.get("sample_offset", -1)) == 8
            and metrics.get("precisions") == ["int8"]
            and float(metrics.get("budget", -1.0)) == 0.08
            and int(metrics.get("target_prompt_tokens", -1)) == 8000
            and int(metrics.get("seed", -1)) == 17
            and int(metrics.get("packing_seed", -1)) == 911
        ),
        "selected_indices_are_unique": (
            len(metrics.get("selected_indices", [])) == expected_n
            and len(set(map(int, metrics.get("selected_indices", [])))) == expected_n
        ),
        "row_ranges_and_bytes_are_valid": all(
            int(row["baseline_cache_bytes"]) > 0
            and int(row["tiered_storage_bytes_initial"]) > 0
            and np.isclose(
                float(row["storage_reduction_factor"]),
                int(row["baseline_cache_bytes"])
                / int(row["tiered_storage_bytes_initial"]),
                rtol=1e-12,
                atol=0.0,
            )
            # An example with no auxiliary tokens has nothing to quantize.  It
            # may be a few metadata bytes larger than the all-BF16 cache, so
            # enforce per-row savings only when the quantized tier is nonempty.
            and (
                int(row.get("auxiliary_tokens", 0)) == 0
                or int(row["tiered_storage_bytes_initial"])
                < int(row["baseline_cache_bytes"])
            )
            and float(row["exact_generation_match"]) in (0.0, 1.0)
            and 0.0 <= float(row["baseline_answer_f1"]) <= 1.0
            and 0.0 <= float(row["tiered_answer_f1"]) <= 1.0
            for row in rows
        ),
        "finite_metrics": all(
            np.isfinite(float(row[key]))
            for row in rows
            for key in (
                "storage_reduction_factor",
                "exact_generation_match",
                "baseline_answer_f1",
                "tiered_answer_f1",
            )
        ),
    }
    storage_reduction = float(
        np.mean([float(row["storage_reduction_factor"]) for row in rows])
    )
    exact_rate = float(np.mean([float(row["exact_generation_match"]) for row in rows]))
    differences = np.asarray(
        [
            float(row["tiered_answer_f1"]) - float(row["baseline_answer_f1"])
            for row in rows
        ],
        dtype=np.float64,
    )
    statistics = paired_bootstrap(differences, args.bootstrap_draws, args.seed)
    gates = {
        "integrity": all(integrity.values()),
        "mean_storage_reduction_at_least_1_2": storage_reduction >= 1.2,
        "exact_generation_rate_at_least_0_95": exact_rate >= 0.95,
        "mean_f1_delta_at_least_minus_0_02": statistics["mean"] >= -0.02,
        "f1_ci95_lower_at_least_minus_0_02": statistics["ci95"][0] >= -0.02,
    }
    payload = {
        "protocol": {
            "precision": "int8",
            "sample_offset": 8,
            "noninferiority_margin": -0.02,
            "minimum_exact_generation_rate": 0.95,
            "minimum_storage_reduction": 1.2,
        },
        "integrity": integrity,
        "mean_storage_reduction_factor": storage_reduction,
        "exact_generation_match_rate": exact_rate,
        "paired_f1_delta": statistics,
        "gates": gates,
        "passed": all(gates.values()),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
