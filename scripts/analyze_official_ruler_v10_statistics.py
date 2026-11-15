#!/usr/bin/env python
"""Paired task-stratified inference for formal official-RULER v10 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_longbench_risk_formal_statistics import (
    holm_adjust,
    paired_inference,
    task_macro_score,
)


PHYSICAL_METHODS = (
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
)
UTILITY_METHOD = "confirmation_stable_proofkv_v10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=50000)
    parser.add_argument("--permutation-replicates", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return metrics, rows


def paired_matrix(
    rows: list[dict[str, Any]],
    tasks: list[str],
    capacity: int,
    fallback_by_task: dict[str, str],
) -> np.ndarray:
    scores = {
        (
            str(row["task"]),
            str(row["id"]),
            str(row["method"]),
            str(row["capacity"]),
        ): 100.0 * float(row["score"])
        for row in rows
    }
    values = []
    for task in tasks:
        ids = sorted(
            {
                str(row["id"])
                for row in rows
                if row["task"] == task
                and row["method"] == UTILITY_METHOD
                and int(row["capacity"]) == capacity
            }
        )
        values.append(
            [
                scores[(task, source_id, UTILITY_METHOD, str(capacity))]
                - scores[
                    (
                        task,
                        source_id,
                        fallback_by_task[task],
                        str(capacity),
                    )
                ]
                for source_id in ids
            ]
        )
    widths = {len(row) for row in values}
    if len(widths) != 1 or not widths or next(iter(widths)) < 1:
        raise ValueError("paired official-RULER matrix is incomplete")
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1000 or args.permutation_replicates < 1000:
        raise ValueError("inference replicate counts must be at least 1000")
    controller_path = Path(args.controller)
    controller = json.loads(controller_path.read_text())
    capacities_to_test = tuple(
        int(value) for value in controller["capacities"]
    )
    policies = controller["final_policy_by_model_task_capacity"]
    per_model_capacity = []
    primary_blocks = []
    run_records = []
    for run_number, value in enumerate(args.run_dir):
        run_dir = Path(value)
        metrics, rows = load_run(run_dir)
        model = str(metrics["model"])
        tasks = [str(task) for task in metrics["tasks"]]
        full_score = task_macro_score(rows, tasks, "full", "full")
        capacities = []
        for capacity_number, capacity in enumerate(capacities_to_test):
            fallbacks = {
                task: str(
                    policies[f"{model}:{task}:{capacity}"][
                        "fallback_method"
                    ]
                )
                for task in tasks
            }
            paired = paired_matrix(rows, tasks, capacity, fallbacks)
            inference = paired_inference(
                paired,
                args.bootstrap_replicates,
                args.permutation_replicates,
                np.random.default_rng(
                    args.seed + 100 * run_number + capacity_number
                ),
            )
            inference.update(
                {
                    "model": model,
                    "capacity": capacity,
                    "comparison": "v10_vs_development_frozen_physical_fallback",
                }
            )
            per_model_capacity.append(inference)
            primary_blocks.append(paired)
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
                    fallbacks[task],
                    capacity,
                )
                for task in tasks
            ) / len(tasks)
            physical_scores = {
                method: task_macro_score(
                    rows,
                    tasks,
                    method,
                    capacity,
                )
                for method in PHYSICAL_METHODS
            }
            capacities.append(
                {
                    "capacity": capacity,
                    "v10_score": utility_score,
                    "frozen_fallback_by_task": fallbacks,
                    "frozen_fallback_score": fallback_score,
                    "v10_delta_vs_fallback": utility_score - fallback_score,
                    "physical_scores": physical_scores,
                    "full_score": full_score,
                    "task_breakdown": [
                        {
                            "task": task,
                            "v10_score": task_macro_score(
                                rows,
                                [task],
                                UTILITY_METHOD,
                                capacity,
                            ),
                            "fallback_score": task_macro_score(
                                rows,
                                [task],
                                fallbacks[task],
                                capacity,
                            ),
                        }
                        for task in tasks
                    ],
                }
            )
        run_records.append(
            {
                "run_dir": str(run_dir.resolve()),
                "model": model,
                "tasks": tasks,
                "samples_per_task": int(metrics["n_per_task"]),
                "full_score": full_score,
                "capacities": capacities,
            }
        )
    holm_adjust(per_model_capacity)
    global_paired = np.stack(primary_blocks, axis=0).mean(axis=0)
    primary = paired_inference(
        global_paired,
        args.bootstrap_replicates,
        args.permutation_replicates,
        np.random.default_rng(args.seed + 10000),
    )
    primary.update(
        {
            "comparison": "v10_vs_development_frozen_physical_fallback",
            "equal_weighting": (
                "average models and capacities within each frozen "
                "task-sample pair, then task macro"
            ),
            "task_groups": int(global_paired.shape[0]),
            "samples_per_task": int(global_paired.shape[1]),
        }
    )
    payload = {
        "protocol": "proofkv_v10_official_ruler_paired_inference_v1",
        "score_unit": "official RULER reference-containment percentage points",
        "controller": str(controller_path.resolve()),
        "controller_sha256": sha256_file(controller_path),
        "analysis_implementation_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "seed": args.seed,
        "precommitted_primary_endpoint": (
            "equal-weighted task macro of v10 versus each "
            "model-task-capacity development-frozen physical fallback"
        ),
        "primary_inference": primary,
        "capacity_family_correction": "Holm",
        "capacity_inference": per_model_capacity,
        "runs": run_records,
        "input_artifact_sha256": {
            str(Path(value).resolve()): {
                name: sha256_file(Path(value) / name)
                for name in [
                    "results.jsonl",
                    "metrics.json",
                    "run_manifest.json",
                ]
            }
            for value in args.run_dir
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "models": len(run_records),
                "primary_delta": primary["paired_task_macro_delta"],
                "primary_p": primary["two_sided_sign_flip_p"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
