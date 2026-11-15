#!/usr/bin/env python3
"""Audit the frozen n=128 multi-length V22 formal evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(probability * (len(ordered) - 1))))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(str(row["id"]), str(row["method"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate (id, method) rows in {path}")
    return rows


def paired_inference(
    deltas: dict[str, list[float]],
    rng: random.Random,
    bootstrap_n: int,
    permutation_n: int,
) -> dict[str, Any]:
    cells = sorted(deltas)
    observed = sum(sum(deltas[cell]) / len(deltas[cell]) for cell in cells) / len(cells)
    boot = []
    for _ in range(bootstrap_n):
        cell_samples = []
        for cell in cells:
            values = deltas[cell]
            cell_samples.append(
                sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            )
        boot.append(100.0 * sum(cell_samples) / len(cell_samples))
    extreme = 0
    for _ in range(permutation_n):
        cell_samples = []
        for cell in cells:
            values = deltas[cell]
            cell_samples.append(
                sum(value if rng.random() < 0.5 else -value for value in values)
                / len(values)
            )
        if abs(sum(cell_samples) / len(cell_samples)) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "macro_delta_points": 100.0 * observed,
        "ci95_points": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "permutation_p_two_sided": (extreme + 1) / (permutation_n + 1),
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * raw[name]))
        adjusted[name] = running
    return adjusted


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "FROZEN":
        raise ValueError("protocol is not frozen")
    if sha256_file(Path(protocol["runner"])) != protocol["runner_sha256"]:
        raise ValueError("runner changed after freeze")
    if (
        sha256_file(Path(protocol["audit_implementation"]))
        != protocol["audit_implementation_sha256"]
    ):
        raise ValueError("audit implementation changed after freeze")

    feature = protocol["feature"]
    threshold = float(protocol["threshold"])
    proof_method = protocol["proof_method"]
    fallback_method = protocol["fallback_method"]
    common_baselines = list(protocol["common_compact_baselines"])
    comparators = common_baselines + ["full"]
    n = int(protocol["samples_per_task_target"])

    deltas: dict[str, dict[str, list[float]]] = {
        comparator: {} for comparator in comparators
    }
    cell_metrics: dict[str, Any] = {}
    selected_retained = []
    integrity: dict[str, Any] = {}

    for task in protocol["tasks"]:
        for target in protocol["target_prompt_tokens"]:
            cell = f"{task}_{target}"
            methods = protocol["methods_by_target"][str(target)]
            directory = Path(protocol["output_directories"][cell])
            results_path = directory / "results.jsonl"
            rows = load_rows(results_path)
            if len(rows) != n * len(methods):
                raise ValueError(f"{cell}: incomplete row count {len(rows)}")
            by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            for row in rows:
                by_id[str(row["id"])][str(row["method"])] = row
            if len(by_id) != n or any(
                set(values) != set(methods) for values in by_id.values()
            ):
                raise ValueError(f"{cell}: incomplete method pairing")

            method_means = {
                method: sum(
                    float(values[method]["answer_f1"]) for values in by_id.values()
                )
                / n
                for method in methods
            }
            controller_scores = []
            route_counts = {proof_method: 0, fallback_method: 0}
            for values in by_id.values():
                proof_row = values[proof_method]
                margin = float(proof_row["selection_partition"][feature])
                selected_method = proof_method if margin > threshold else fallback_method
                selected_row = values[selected_method]
                controller_score = float(selected_row["answer_f1"])
                controller_scores.append(controller_score)
                selected_retained.append(float(selected_row["retained_ratio"]))
                route_counts[selected_method] += 1
                for comparator in comparators:
                    deltas[comparator].setdefault(cell, []).append(
                        controller_score - float(values[comparator]["answer_f1"])
                    )
            controller_mean = sum(controller_scores) / n
            best_common = max(common_baselines, key=lambda name: method_means[name])
            cell_metrics[cell] = {
                "task": task,
                "target_prompt_tokens": target,
                "n": n,
                "controller_f1": controller_mean,
                "method_f1": method_means,
                "best_common_compact_baseline": best_common,
                "delta_vs_best_common_compact_points": 100.0
                * (controller_mean - method_means[best_common]),
                "delta_vs_each_method_points": {
                    method: 100.0 * (controller_mean - method_means[method])
                    for method in methods
                },
                "route_counts": route_counts,
            }
            integrity[cell] = {
                "rows": len(rows),
                "paired_ids": len(by_id),
                "results_sha256": sha256_file(results_path),
                "manifest_sha256": sha256_file(directory / "run_manifest.json"),
            }

    rng = random.Random(int(protocol["inference_seed"]))
    inference = {
        comparator: paired_inference(
            deltas[comparator],
            rng,
            int(protocol["bootstrap_replicates"]),
            int(protocol["permutation_replicates"]),
        )
        for comparator in comparators
    }
    adjusted_p = holm_adjust(
        {
            comparator: inference[comparator]["permutation_p_two_sided"]
            for comparator in comparators
        }
    )
    for comparator in comparators:
        inference[comparator]["holm_adjusted_p"] = adjusted_p[comparator]

    target_macro_vs_primary = {}
    primary = protocol["primary_development_selected_comparator"]
    for target in protocol["target_prompt_tokens"]:
        cells = [
            values
            for values in cell_metrics.values()
            if values["target_prompt_tokens"] == target
        ]
        target_macro_vs_primary[str(target)] = sum(
            values["delta_vs_each_method_points"][primary] for values in cells
        ) / len(cells)

    macro_vs_compact = {
        comparator: inference[comparator]["macro_delta_points"]
        for comparator in common_baselines
    }
    minimum_cell_vs_full = min(
        values["delta_vs_each_method_points"]["full"]
        for values in cell_metrics.values()
    )
    mean_selected_retained = sum(selected_retained) / len(selected_retained)
    minimum_branch_count = min(
        count
        for values in cell_metrics.values()
        for count in values["route_counts"].values()
    )

    gate = protocol["formal_gate"]
    checks = {
        "primary_ci95": inference[primary]["ci95_points"][0]
        > gate["primary_comparator_ci95_low_points_gt"],
        "primary_holm_p": inference[primary]["holm_adjusted_p"]
        < gate["primary_comparator_holm_p_lt"],
        "full_ci95": inference["full"]["ci95_points"][0]
        > gate["full_context_ci95_low_points_gt"],
        "full_holm_p": inference["full"]["holm_adjusted_p"]
        < gate["full_context_holm_p_lt"],
        "macro_vs_each_common_compact": min(macro_vs_compact.values())
        > gate["macro_delta_vs_each_common_compact_baseline_points_gt"],
        "each_target_vs_primary": min(target_macro_vs_primary.values())
        > gate["minimum_target_macro_delta_vs_primary_points_gt"],
        "minimum_cell_vs_full": minimum_cell_vs_full
        >= gate["minimum_cell_delta_vs_full_points_gte"],
        "selected_retained_ratio": mean_selected_retained
        <= gate["mean_selected_retained_ratio_lte"],
        "both_routes_each_cell": minimum_branch_count
        >= gate["minimum_routes_per_branch_per_cell_gte"],
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "REJECT",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "cell_metrics": cell_metrics,
        "paired_inference": inference,
        "target_macro_delta_vs_primary_points": target_macro_vs_primary,
        "macro_delta_vs_common_compact_baselines_points": macro_vs_compact,
        "minimum_cell_delta_vs_full_points": minimum_cell_vs_full,
        "mean_selected_retained_ratio": mean_selected_retained,
        "minimum_branch_count_per_cell": minimum_branch_count,
        "checks": checks,
        "integrity": integrity,
        "audit_implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
