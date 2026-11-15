#!/usr/bin/env python
"""Paired inference for the untouched upgraded ProofKV v10 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_longbench_risk_formal_statistics import (
    holm_adjust,
    paired_inference,
    task_macro_score,
)
from audit_longbench_structured_common import sha256_file


CAPACITIES = (64, 128, 256, 512)
PHYSICAL_METHODS = (
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
)
COVERAGE_METHODS = (
    "learned_and_or_recent_aux",
    "learned_and_or_stratified_aux",
    "learned_and_or_graph_aux",
    "learned_and_or_mixed_aux",
    "learned_and_or_lexical_anchor25_aux",
    "learned_and_or_lexical_anchor50_aux",
    "learned_and_or_lexical_anchor75_aux",
    "learned_and_or_control_anchor25_aux",
    "learned_and_or_control_anchor50_aux",
    "learned_and_or_control_anchor75_aux",
)
UTILITY_METHOD = "confirmation_stable_proofkv_v10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utility-controller", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--permutation-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--analysis-role",
        choices=("formal", "confirmation"),
        default="formal",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_run(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        candidates = sorted(run_dir.glob("pilot_metrics*.json"))
        if len(candidates) != 1:
            raise ValueError(f"{run_dir}: expected one completed metrics file")
        metrics_path = candidates[0]
    metrics = json.loads(metrics_path.read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return metrics, rows


def domain_paired_delta_matrix(
    rows: list[dict[str, object]],
    tasks: list[str],
    left_method: str,
    physical_by_task: dict[str, str],
    capacity: int,
) -> np.ndarray:
    score_by_key = {
        (
            str(row["task"]),
            str(row["id"]),
            str(row["method"]),
            str(row["capacity"]),
        ): 100 * float(row["score"])
        for row in rows
    }
    values = []
    for task in tasks:
        ids = sorted(
            {
                str(row["id"])
                for row in rows
                if row["task"] == task
                and row["method"] == left_method
                and int(row["capacity"]) == capacity
            }
        )
        values.append(
            [
                score_by_key[
                    (task, source_id, left_method, str(capacity))
                ]
                - score_by_key[
                    (
                        task,
                        source_id,
                        physical_by_task[task],
                        str(capacity),
                    )
                ]
                for source_id in ids
            ]
        )
    widths = {len(task_values) for task_values in values}
    if len(widths) != 1 or not widths or next(iter(widths)) < 1:
        raise ValueError("domain paired comparison is incomplete")
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1000 or args.permutation_replicates < 1000:
        raise ValueError("inference replicate counts must each be at least 1000")
    controller = json.loads(Path(args.utility_controller).read_text())
    if controller.get("status") != "PASS":
        raise ValueError("utility controller did not pass calibration")
    rng = np.random.default_rng(args.seed)
    capacity_robustness = []
    global_paired_blocks = []
    fixed_candidate_paired_blocks: dict[str, list[np.ndarray]] = {
        method: [] for method in COVERAGE_METHODS
    }
    runs = []
    for run_dir_value in args.run_dir:
        run_dir = Path(run_dir_value)
        metrics, rows = load_run(run_dir)
        model = str(metrics["model"])
        tasks = [str(task) for task in metrics["tasks"]]
        full_score = task_macro_score(rows, tasks, "full", "full")
        capacity_records = []
        for capacity in CAPACITIES:
            domain_conditioned = (
                controller.get("domain_conditioned") is True
            )
            policy_map = controller[
                (
                    "final_policy_by_model_task_capacity"
                    if domain_conditioned
                    else "final_policy_by_model_capacity"
                )
            ]
            task_fallbacks = {
                task: str(
                    policy_map[
                        (
                            f"{model}:{task}:{capacity}"
                            if domain_conditioned
                            else f"{model}:{capacity}"
                        )
                    ]["fallback_method"]
                )
                for task in tasks
            }
            fallback_methods = sorted(set(task_fallbacks.values()))
            fallback = (
                fallback_methods[0]
                if len(fallback_methods) == 1
                else "domain_conditioned"
            )
            utility_score = task_macro_score(
                rows,
                tasks,
                UTILITY_METHOD,
                capacity,
            )
            fallback_score = sum(
                task_macro_score(
                    rows,
                    [task],
                    task_fallbacks[task],
                    capacity,
                )
                for task in tasks
            ) / len(tasks)
            physical_scores = {
                method: task_macro_score(rows, tasks, method, capacity)
                for method in PHYSICAL_METHODS
            }
            coverage_scores = {
                method: task_macro_score(rows, tasks, method, capacity)
                for method in COVERAGE_METHODS
                if method in set(metrics["methods"])
            }
            best_coverage = (
                max(
                    coverage_scores,
                    key=lambda method: (coverage_scores[method], method),
                )
                if coverage_scores
                else None
            )
            best_physical = max(
                PHYSICAL_METHODS,
                key=lambda method: (physical_scores[method], method),
            )
            utility_rows = [
                row
                for row in rows
                if row["method"] == UTILITY_METHOD
                and int(row["capacity"]) == capacity
            ]
            proof_routes = sum(
                row["selection_partition"]["utility_route"]
                == "proof_coverage"
                for row in utility_rows
            )
            selected_candidates = {
                candidate: sum(
                    row["selection_partition"]["utility_selected_method"]
                    == candidate
                    for row in utility_rows
                )
                for candidate in COVERAGE_METHODS
            }
            task_breakdown = []
            for task in tasks:
                task_utility = task_macro_score(
                    rows,
                    [task],
                    UTILITY_METHOD,
                    capacity,
                )
                task_fallback = task_macro_score(
                    rows,
                    [task],
                    task_fallbacks[task],
                    capacity,
                )
                task_breakdown.append(
                    {
                        "task": task,
                        "utility_score": task_utility,
                        "fallback_score": task_fallback,
                        "utility_delta_vs_fallback": (
                            task_utility - task_fallback
                        ),
                    }
                )
            paired = domain_paired_delta_matrix(
                rows,
                tasks,
                UTILITY_METHOD,
                task_fallbacks,
                capacity,
            )
            inference = paired_inference(
                paired,
                args.bootstrap_replicates,
                args.permutation_replicates,
                rng,
            )
            inference.update(
                {
                    "model": model,
                    "capacity": capacity,
                    "comparison": (
                        "utility_router_vs_calibration_selected_physical"
                    ),
                    "utility_method": UTILITY_METHOD,
                    "physical_method": fallback,
                }
            )
            capacity_robustness.append(inference)
            global_paired_blocks.append(paired)
            for candidate in COVERAGE_METHODS:
                if candidate not in coverage_scores:
                    continue
                fixed_candidate_paired_blocks[candidate].append(
                    domain_paired_delta_matrix(
                        rows,
                        tasks,
                        UTILITY_METHOD,
                        {task: candidate for task in tasks},
                        capacity,
                    )
                )
            capacity_records.append(
                {
                    "capacity": capacity,
                    "utility_score": utility_score,
                    "calibration_selected_physical_method": fallback,
                    "calibration_selected_physical_by_task": (
                        task_fallbacks
                    ),
                    "calibration_selected_physical_score": fallback_score,
                    "utility_delta_vs_fallback": (
                        utility_score - fallback_score
                    ),
                    "formal_physical_scores": physical_scores,
                    "formal_coverage_scores": coverage_scores,
                    "best_formal_coverage_method_descriptive_only": (
                        best_coverage
                    ),
                    "best_formal_coverage_score_descriptive_only": (
                        coverage_scores[best_coverage]
                        if best_coverage is not None
                        else None
                    ),
                    "utility_delta_vs_best_formal_coverage_descriptive_only": (
                        utility_score - coverage_scores[best_coverage]
                        if best_coverage is not None
                        else None
                    ),
                    "best_formal_physical_method_descriptive_only": (
                        best_physical
                    ),
                    "best_formal_physical_score_descriptive_only": (
                        physical_scores[best_physical]
                    ),
                    "utility_delta_vs_best_formal_physical_descriptive_only": (
                        utility_score - physical_scores[best_physical]
                    ),
                    "full_score": full_score,
                    "utility_relative_full": (
                        utility_score / full_score if full_score else None
                    ),
                    "proof_routes": proof_routes,
                    "utility_rows": len(utility_rows),
                    "proof_route_coverage": (
                        proof_routes / len(utility_rows)
                        if utility_rows
                        else None
                    ),
                    "selected_candidate_counts": selected_candidates,
                    "task_breakdown": task_breakdown,
                }
            )
        runs.append(
            {
                "run_dir": str(run_dir),
                "model": model,
                "tasks": tasks,
                "samples_per_task": int(metrics["n_per_task"]),
                "selection_start_per_task": int(
                    metrics["selection_start_per_task"]
                ),
                "full_score": full_score,
                "capacities": capacity_records,
            }
        )
    holm_adjust(capacity_robustness)
    global_paired = np.stack(global_paired_blocks, axis=0).mean(axis=0)
    global_primary = paired_inference(
        global_paired,
        args.bootstrap_replicates,
        args.permutation_replicates,
        np.random.default_rng(args.seed + 1),
    )
    global_primary.update(
        {
            "comparison": (
                "utility_router_vs_frozen_domain_physical_fallback"
            ),
            "equal_weighting": (
                "average models and capacities within each frozen "
                "task-sample pair, then task macro"
            ),
            "task_groups": int(global_paired.shape[0]),
            "samples_per_task": int(global_paired.shape[1]),
        }
    )
    fixed_candidate_inference = []
    for index, candidate in enumerate(COVERAGE_METHODS):
        blocks = fixed_candidate_paired_blocks[candidate]
        if not blocks:
            continue
        paired = np.stack(blocks, axis=0).mean(axis=0)
        inference = paired_inference(
            paired,
            args.bootstrap_replicates,
            args.permutation_replicates,
            np.random.default_rng(args.seed + 10 + index),
        )
        inference.update(
            {
                "comparison": (
                    f"utility_router_vs_fixed_{candidate}"
                ),
                "utility_method": UTILITY_METHOD,
                "fixed_candidate_method": candidate,
                "equal_weighting": (
                    "average models and capacities within each frozen "
                    "task-sample pair, then task macro"
                ),
            }
        )
        fixed_candidate_inference.append(inference)
    holm_adjust(fixed_candidate_inference)
    payload = {
        "protocol": "paired_task_stratified_longbench_utility_inference_v1",
        "analysis_role": args.analysis_role,
        "excluded_from_final_effect_estimates": (
            args.analysis_role == "confirmation"
        ),
        "score_unit": "official LongBench QA-F1 points",
        "controller": str(Path(args.utility_controller)),
        "utility_controller_sha256": sha256_file(
            Path(args.utility_controller)
        ),
        "analysis_implementation_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "input_artifact_sha256": {
            str(Path(value)): {
                "results.jsonl": sha256_file(
                    Path(value) / "results.jsonl"
                ),
                "run_manifest.json": sha256_file(
                    Path(value) / "run_manifest.json"
                ),
                "metrics": sha256_file(
                    (
                        Path(value) / "metrics.json"
                        if (Path(value) / "metrics.json").is_file()
                        else sorted(
                            Path(value).glob(
                                "pilot_metrics*.json"
                            )
                        )[0]
                    )
                ),
            }
            for value in args.run_dir
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "seed": args.seed,
        "precommitted_primary_endpoint": (
            "equal-weighted macro utility of the router versus each "
            "model-task-capacity policy's frozen physical fallback"
        ),
        "primary_inference": global_primary,
        "capacity_robustness_family": (
            "one paired test per model and capacity"
        ),
        "capacity_robustness_multiple_comparison_correction": (
            "Holm family-wise error control"
        ),
        "fixed_candidate_robustness_family": (
            "one global paired comparison against each of ten "
            "predeclared static proof-plus-coverage candidates"
        ),
        "fixed_candidate_robustness_multiple_comparison_correction": (
            "Holm family-wise error control"
        ),
        "runs": runs,
        "capacity_robustness_inference": capacity_robustness,
        "fixed_candidate_robustness_inference": (
            fixed_candidate_inference
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "models": len(runs),
                "primary_tests": 1,
                "capacity_robustness_tests": len(
                    capacity_robustness
                ),
                "fixed_candidate_robustness_tests": len(
                    fixed_candidate_inference
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
