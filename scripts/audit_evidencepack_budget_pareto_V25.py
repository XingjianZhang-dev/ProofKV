#!/usr/bin/env python3
"""Audit the frozen EvidencePack quality--budget Pareto evaluation."""

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
    parser.add_argument("--mode", choices=("partial", "final"), required=True)
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
    tasks = sorted(deltas)
    observed = sum(sum(deltas[task]) / len(deltas[task]) for task in tasks) / len(tasks)
    bootstrap = []
    for _ in range(bootstrap_n):
        task_means = []
        for task in tasks:
            values = deltas[task]
            task_means.append(
                sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            )
        bootstrap.append(100.0 * sum(task_means) / len(task_means))
    extreme = 0
    for _ in range(permutation_n):
        task_means = []
        for task in tasks:
            values = deltas[task]
            task_means.append(
                sum(value if rng.random() < 0.5 else -value for value in values)
                / len(values)
            )
        if abs(sum(task_means) / len(task_means)) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "macro_delta_points": 100.0 * observed,
        "ci95_points": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
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
        raise ValueError("runner changed after protocol freeze")
    if (
        sha256_file(Path(protocol["audit_implementation"]))
        != protocol["audit_implementation_sha256"]
    ):
        raise ValueError("audit implementation changed after protocol freeze")
    if (
        sha256_file(Path(protocol["source_controller_protocol"]))
        != protocol["source_controller_protocol_sha256"]
        or sha256_file(Path(protocol["source_formal_audit"]))
        != protocol["source_formal_audit_sha256"]
    ):
        raise ValueError("source controller evidence changed after protocol freeze")

    n = int(protocol["samples_per_task"])
    methods = list(protocol["compact_methods"])
    controller = protocol["controller"]
    feature = controller["feature"]
    proof_method = controller["proof_method"]
    fallback_method = controller["fallback_method"]
    threshold = float(controller["threshold"])
    budgets = [0.04, 0.08] if args.mode == "partial" else list(protocol["budgets"])

    by_task_budget: dict[str, dict[float, dict[str, dict[str, dict[str, Any]]]]] = {}
    integrity: dict[str, Any] = {}
    cell_metrics: dict[str, Any] = {}
    common_ids_by_task: dict[str, set[str]] = {}

    for task in protocol["tasks"]:
        by_task_budget[task] = {}
        for budget in budgets:
            code = f"{round(100 * budget):02d}"
            key = f"{task}_b{code}"
            directory = Path(protocol["output_directories"][key])
            rows = load_rows(directory / "results.jsonl")
            if len(rows) != n * len(methods):
                raise ValueError(f"{key}: incomplete compact row count {len(rows)}")
            by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            for row in rows:
                by_id[str(row["id"])][str(row["method"])] = row
            if len(by_id) != n or any(set(values) != set(methods) for values in by_id.values()):
                raise ValueError(f"{key}: incomplete compact method pairing")
            ids = set(by_id)
            if task in common_ids_by_task and ids != common_ids_by_task[task]:
                raise ValueError(f"{key}: sample IDs differ across budgets")
            common_ids_by_task[task] = ids
            by_task_budget[task][budget] = by_id

            controller_scores = []
            controller_retained = []
            route_counts = {proof_method: 0, fallback_method: 0}
            method_means = {
                method: sum(float(values[method]["answer_f1"]) for values in by_id.values())
                / n
                for method in methods
            }
            for values in by_id.values():
                feature_value = values[proof_method]["selection_partition"].get(feature)
                if not isinstance(feature_value, (int, float)):
                    raise ValueError(f"{key}: missing numeric predecode feature")
                selected_method = (
                    proof_method if float(feature_value) > threshold else fallback_method
                )
                selected = values[selected_method]
                controller_scores.append(float(selected["answer_f1"]))
                controller_retained.append(float(selected["retained_ratio"]))
                route_counts[selected_method] += 1
            controller_mean = sum(controller_scores) / n
            retained_mean = sum(controller_retained) / n
            cell_metrics[key] = {
                "task": task,
                "budget": budget,
                "n": n,
                "controller_f1": controller_mean,
                "controller_retained_ratio": retained_mean,
                "method_f1": method_means,
                "delta_vs_each_compact_points": {
                    method: 100.0 * (controller_mean - method_means[method])
                    for method in methods
                },
                "route_counts": route_counts,
            }
            integrity[key] = {
                "rows": len(rows),
                "paired_ids": len(by_id),
                "results_sha256": sha256_file(directory / "results.jsonl"),
                "manifest_sha256": sha256_file(directory / "run_manifest.json"),
                "metrics_sha256": sha256_file(directory / "metrics.json"),
            }

    if args.mode == "partial":
        primary_budget = float(protocol["early_sanity_gate"]["budget"])
        primary_cells = [
            cell_metrics[f"{task}_b{round(100 * primary_budget):02d}"]
            for task in protocol["tasks"]
        ]
        macro_delta = {
            method: sum(
                cell["delta_vs_each_compact_points"][method] for cell in primary_cells
            )
            / len(primary_cells)
            for method in methods
        }
        floor = float(
            protocol["early_sanity_gate"][
                "minimum_macro_delta_vs_each_compact_points_gte"
            ]
        )
        checks = {
            "all_rows_complete": True,
            "matching_ids_across_budgets": True,
            "predecode_feature_present": True,
            "no_severe_8pct_direction_reversal": min(macro_delta.values()) >= floor,
        }
        payload = {
            "status": "PASS" if all(checks.values()) else "REJECT",
            "mode": "partial",
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "macro_delta_at_8pct_vs_compact_points": macro_delta,
            "cell_metrics": cell_metrics,
            "checks": checks,
            "integrity": integrity,
            "audit_implementation_sha256": sha256_file(Path(__file__).resolve()),
        }
    else:
        full_by_task: dict[str, dict[str, dict[str, Any]]] = {}
        for task in protocol["tasks"]:
            key = f"{task}_full"
            directory = Path(protocol["output_directories"][key])
            rows = load_rows(directory / "results.jsonl")
            if len(rows) != n or any(str(row["method"]) != "full" for row in rows):
                raise ValueError(f"{key}: incomplete full-reference rows")
            by_id = {str(row["id"]): row for row in rows}
            if set(by_id) != common_ids_by_task[task]:
                raise ValueError(f"{key}: full-reference IDs differ from compact IDs")
            full_by_task[task] = by_id
            integrity[key] = {
                "rows": len(rows),
                "paired_ids": len(by_id),
                "results_sha256": sha256_file(directory / "results.jsonl"),
                "manifest_sha256": sha256_file(directory / "run_manifest.json"),
                "metrics_sha256": sha256_file(directory / "metrics.json"),
            }

        comparators = methods + ["full"]
        deltas: dict[str, dict[str, list[float]]] = {
            comparator: {task: [] for task in protocol["tasks"]}
            for comparator in comparators
        }
        for task in protocol["tasks"]:
            for sample_id in sorted(common_ids_by_task[task]):
                sample_controller_scores = []
                sample_comparator_scores = {
                    comparator: [] for comparator in comparators
                }
                for budget in budgets:
                    values = by_task_budget[task][budget][sample_id]
                    selected_method = (
                        proof_method
                        if float(values[proof_method]["selection_partition"][feature])
                        > threshold
                        else fallback_method
                    )
                    sample_controller_scores.append(float(values[selected_method]["answer_f1"]))
                    for method in methods:
                        sample_comparator_scores[method].append(
                            float(values[method]["answer_f1"])
                        )
                    sample_comparator_scores["full"].append(
                        float(full_by_task[task][sample_id]["answer_f1"])
                    )
                controller_curve_mean = sum(sample_controller_scores) / len(budgets)
                for comparator in comparators:
                    comparator_curve_mean = (
                        sum(sample_comparator_scores[comparator]) / len(budgets)
                    )
                    deltas[comparator][task].append(
                        controller_curve_mean - comparator_curve_mean
                    )

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
        adjusted = holm_adjust(
            {
                comparator: inference[comparator]["permutation_p_two_sided"]
                for comparator in comparators
            }
        )
        for comparator in comparators:
            inference[comparator]["holm_adjusted_p"] = adjusted[comparator]

        budget_macro = {}
        for budget in budgets:
            code = f"{round(100 * budget):02d}"
            cells = [cell_metrics[f"{task}_b{code}"] for task in protocol["tasks"]]
            budget_macro[str(budget)] = {
                "controller_f1": sum(cell["controller_f1"] for cell in cells)
                / len(cells),
                "controller_retained_ratio": sum(
                    cell["controller_retained_ratio"] for cell in cells
                )
                / len(cells),
                "delta_vs_each_compact_points": {
                    method: sum(
                        cell["delta_vs_each_compact_points"][method] for cell in cells
                    )
                    / len(cells)
                    for method in methods
                },
            }

        retained = [
            budget_macro[str(budget)]["controller_retained_ratio"] for budget in budgets
        ]
        compact_positive = all(
            inference[method]["macro_delta_points"] > 0.0 for method in methods
        )
        primary_ci_positive = inference[proof_method]["ci95_points"][0] > 0.0
        scientific_outcome = (
            "PRIMARY_SUPPORTED"
            if compact_positive and primary_ci_positive
            else "DESCRIPTIVE_CURVE_ONLY"
        )
        checks = {
            "all_rows_complete": True,
            "matching_ids_across_budgets_and_full": True,
            "predecode_feature_present": True,
            "controller_retention_monotone": all(
                retained[index] < retained[index + 1]
                for index in range(len(retained) - 1)
            ),
            "controller_retention_within_budget_ceiling": all(
                retained[index] <= budgets[index] + 0.03
                for index in range(len(budgets))
            ),
            "both_routes_observed": all(
                min(cell["route_counts"].values()) >= 4
                for cell in cell_metrics.values()
            ),
        }
        payload = {
            "status": "PASS" if all(checks.values()) else "INVALID",
            "mode": "final",
            "scientific_outcome": scientific_outcome,
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "paired_curve_inference": inference,
            "budget_macro": budget_macro,
            "cell_metrics": cell_metrics,
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
