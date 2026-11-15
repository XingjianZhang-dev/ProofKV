#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CAPACITIES = (64, 128, 256, 512)
RISK_METHOD = "risk_calibrated_proof_coverage"
PHYSICAL_METHODS = (
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired task-stratified inference for the untouched LongBench "
            "risk-calibrated evaluation."
        )
    )
    parser.add_argument("--risk-controller", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--permutation-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty collection")
    return sum(values) / len(values)


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return metrics, rows


def score_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], float]:
    output = {}
    for row in rows:
        key = (
            str(row["task"]),
            str(row["id"]),
            str(row["method"]),
            str(row["capacity"]),
        )
        if key in output:
            raise ValueError(f"duplicate row key: {key}")
        output[key] = 100 * float(row["score"])
    return output


def paired_delta_matrix(
    rows: list[dict[str, Any]],
    tasks: list[str],
    left_method: str,
    right_method: str,
    capacity: int,
) -> np.ndarray:
    lookup = score_lookup(rows)
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
        task_values = [
            lookup[(task, source_id, left_method, str(capacity))]
            - lookup[(task, source_id, right_method, str(capacity))]
            for source_id in ids
        ]
        values.append(task_values)
    widths = {len(task_values) for task_values in values}
    if len(widths) != 1 or not widths or next(iter(widths)) < 1:
        raise ValueError("paired comparison has incomplete or unequal task rows")
    return np.asarray(values, dtype=np.float64)


def paired_inference(
    deltas: np.ndarray,
    bootstrap_replicates: int,
    permutation_replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if deltas.ndim != 2:
        raise ValueError("paired delta matrix must be task by sample")
    task_count, samples_per_task = deltas.shape
    observed = float(deltas.mean(axis=1).mean())
    indices = rng.integers(
        0,
        samples_per_task,
        size=(bootstrap_replicates, task_count, samples_per_task),
    )
    bootstrapped = np.take_along_axis(
        np.broadcast_to(
            deltas,
            (bootstrap_replicates, task_count, samples_per_task),
        ),
        indices,
        axis=2,
    ).mean(axis=2).mean(axis=1)
    ci_low, ci_high = np.quantile(
        bootstrapped,
        [0.025, 0.975],
        method="linear",
    )

    extreme = 0
    remaining = permutation_replicates
    while remaining:
        batch = min(5000, remaining)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(batch, task_count, samples_per_task),
        )
        randomized = (signs * deltas).mean(axis=2).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(randomized) >= abs(observed)))
        remaining -= batch
    p_value = (extreme + 1) / (permutation_replicates + 1)
    return {
        "paired_task_macro_delta": observed,
        "bootstrap_95_ci_low": float(ci_low),
        "bootstrap_95_ci_high": float(ci_high),
        "two_sided_sign_flip_p": p_value,
        "task_count": int(task_count),
        "pairs": int(task_count * samples_per_task),
    }


def holm_adjust(records: list[dict[str, Any]]) -> None:
    ordered = sorted(
        enumerate(records),
        key=lambda item: item[1]["two_sided_sign_flip_p"],
    )
    running = 0.0
    count = len(ordered)
    for rank, (index, record) in enumerate(ordered):
        adjusted = min(
            1.0,
            (count - rank) * float(record["two_sided_sign_flip_p"]),
        )
        running = max(running, adjusted)
        records[index]["holm_adjusted_p"] = running


def task_macro_score(
    rows: list[dict[str, Any]],
    tasks: list[str],
    method: str,
    capacity: int | str,
) -> float:
    return mean(
        [
            mean(
                [
                    100 * float(row["score"])
                    for row in rows
                    if row["task"] == task
                    and row["method"] == method
                    and str(row["capacity"]) == str(capacity)
                ]
            )
            for task in tasks
        ]
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1000 or args.permutation_replicates < 1000:
        raise ValueError("inference replicate counts must each be at least 1000")
    controller = json.loads(Path(args.risk_controller).read_text())
    if controller.get("status") != "PASS":
        raise ValueError("risk controller did not pass calibration")
    proof_method_by_capacity = {
        int(capacity): str(method)
        for capacity, method in controller[
            "proof_method_by_capacity"
        ].items()
    }
    rng = np.random.default_rng(args.seed)
    primary_records = []
    run_records = []
    for run_dir_value in args.run_dir:
        run_dir = Path(run_dir_value)
        metrics, rows = load_run(run_dir)
        model = str(metrics["model"])
        tasks = list(metrics["tasks"])
        full_score = task_macro_score(rows, tasks, "full", "full")
        capacity_records = []
        for capacity in CAPACITIES:
            proof_method = proof_method_by_capacity[capacity]
            fallback = str(
                controller["fallback_by_model_capacity"][
                    f"{model}:{capacity}"
                ]
            )
            risk_score = task_macro_score(
                rows,
                tasks,
                RISK_METHOD,
                capacity,
            )
            fallback_score = task_macro_score(
                rows,
                tasks,
                fallback,
                capacity,
            )
            proof_score = task_macro_score(
                rows,
                tasks,
                proof_method,
                capacity,
            )
            formal_physical_scores = {
                method: task_macro_score(
                    rows,
                    tasks,
                    method,
                    capacity,
                )
                for method in PHYSICAL_METHODS
            }
            best_formal_physical_method = max(
                PHYSICAL_METHODS,
                key=lambda method: (
                    formal_physical_scores[method],
                    method,
                ),
            )
            best_formal_physical_score = formal_physical_scores[
                best_formal_physical_method
            ]
            task_breakdown = []
            for task in tasks:
                task_risk_score = task_macro_score(
                    rows,
                    [task],
                    RISK_METHOD,
                    capacity,
                )
                task_fallback_score = task_macro_score(
                    rows,
                    [task],
                    fallback,
                    capacity,
                )
                task_proof_score = task_macro_score(
                    rows,
                    [task],
                    proof_method,
                    capacity,
                )
                task_breakdown.append(
                    {
                        "task": task,
                        "risk_score": task_risk_score,
                        "fallback_score": task_fallback_score,
                        "proof_score": task_proof_score,
                        "risk_delta_vs_fallback": (
                            task_risk_score - task_fallback_score
                        ),
                    }
                )
            inference = paired_inference(
                paired_delta_matrix(
                    rows,
                    tasks,
                    RISK_METHOD,
                    fallback,
                    capacity,
                ),
                args.bootstrap_replicates,
                args.permutation_replicates,
                rng,
            )
            inference.update(
                {
                    "model": model,
                    "capacity": capacity,
                    "comparison": "risk_vs_calibration_selected_physical",
                    "risk_method": RISK_METHOD,
                    "physical_method": fallback,
                }
            )
            primary_records.append(inference)
            risk_rows = [
                row
                for row in rows
                if row["method"] == RISK_METHOD
                and int(row["capacity"]) == capacity
            ]
            proof_routes = sum(
                row["selection_partition"]["risk_route"]
                == "proof_coverage"
                for row in risk_rows
            )
            capacity_records.append(
                {
                    "capacity": capacity,
                    "risk_score": risk_score,
                    "calibration_selected_physical_method": fallback,
                    "calibration_selected_physical_score": fallback_score,
                    "formal_physical_scores": formal_physical_scores,
                    "best_formal_physical_method_descriptive_only": (
                        best_formal_physical_method
                    ),
                    "best_formal_physical_score_descriptive_only": (
                        best_formal_physical_score
                    ),
                    "selected_proof_method": proof_method,
                    "selected_proof_score": proof_score,
                    "risk_delta_vs_physical": risk_score - fallback_score,
                    "risk_delta_vs_best_formal_physical_descriptive_only": (
                        risk_score - best_formal_physical_score
                    ),
                    "risk_delta_vs_proof": risk_score - proof_score,
                    "full_score": full_score,
                    "risk_relative_full": (
                        risk_score / full_score if full_score else None
                    ),
                    "proof_routes": proof_routes,
                    "risk_rows": len(risk_rows),
                    "proof_route_coverage": (
                        proof_routes / len(risk_rows) if risk_rows else None
                    ),
                    "task_breakdown": task_breakdown,
                }
            )
        run_records.append(
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

    holm_adjust(primary_records)
    payload = {
        "protocol": "paired_task_stratified_longbench_risk_inference_v1",
        "score_unit": "official LongBench QA-F1 points",
        "controller": str(Path(args.risk_controller)),
        "proof_method_by_capacity": {
            str(capacity): method
            for capacity, method in sorted(
                proof_method_by_capacity.items()
            )
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "seed": args.seed,
        "primary_family": (
            "risk-calibrated method versus the calibration-selected fixed "
            "physical fallback, one paired test per model and capacity"
        ),
        "multiple_comparison_correction": "Holm family-wise error control",
        "runs": run_records,
        "primary_inference": primary_records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"models": len(run_records), "tests": len(primary_records)}))


if __name__ == "__main__":
    main()
