#!/usr/bin/env python
"""Calibrate a conservative multi-candidate predecode utility router."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from calibrate_longbench_risk_controller import (
    CAPACITIES,
    COVERAGE_METHODS,
    PHYSICAL_METHODS,
    RISK_FEATURES,
    risk_features,
)


ALPHAS = (0.1, 1.0, 10.0, 100.0)
MARGINS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 1.0e9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-dir", action="append", required=True)
    parser.add_argument("--coverage-run-dir", action="append", required=True)
    parser.add_argument("--extension-run-dir", action="append", required=True)
    parser.add_argument(
        "--minimum-overall-proof-route-coverage",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--minimum-nested-loto-mean-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-nested-loto-nonnegative-tasks",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--minimum-per-model-nested-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-nonnegative-capacities",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--minimum-worst-task-delta",
        type=float,
        default=-0.50,
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_artifact_sha256(run_dir: Path) -> dict[str, str]:
    files = [run_dir / "run_manifest.json", run_dir / "results.jsonl"]
    files.extend(sorted(run_dir.glob("*metrics*.json")))
    return {
        path.name: sha256_file(path)
        for path in files
        if path.is_file()
    }


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = (
        ([run_dir / "metrics.json"] if (run_dir / "metrics.json").exists() else [])
        + sorted(run_dir.glob("pilot_metrics_n*.json"))
        + sorted(run_dir.glob("pilot_metrics_s*_n*.json"))
    )
    if len(candidates) != 1:
        raise ValueError(f"{run_dir}: expected exactly one metrics file")
    metrics = json.loads(candidates[0].read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    keys = [
        (
            str(row["task"]),
            str(row["id"]),
            str(row["method"]),
            str(row["capacity"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{run_dir}: duplicate result keys")
    return metrics, rows


def row_lookup(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    output = {}
    for row in rows:
        if row["method"] == "full":
            continue
        key = (
            str(row["task"]),
            str(row["id"]),
            str(row["method"]),
            int(row["capacity"]),
        )
        if key in output:
            raise ValueError(f"duplicate merged row: {key}")
        output[key] = row
    return output


def task_macro(
    lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    sample_ids: dict[str, list[str]],
    tasks: Iterable[str],
    method: str,
    capacity: int,
) -> float:
    values = []
    for task in tasks:
        scores = [
            100
            * float(
                lookup[(str(task), str(sample_id), method, capacity)]["score"]
            )
            for sample_id in sample_ids[str(task)]
        ]
        values.append(mean(scores))
    return mean(values)


def choose_fallback(
    physical_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    sample_ids: dict[str, list[str]],
    tasks: list[str],
    capacity: int,
) -> str:
    return max(
        PHYSICAL_METHODS,
        key=lambda method: (
            task_macro(
                physical_lookup,
                sample_ids,
                tasks,
                method,
                capacity,
            ),
            method,
        ),
    )


def feature_vector(row: dict[str, Any]) -> np.ndarray:
    values = risk_features(row)
    return np.asarray([values[name] for name in RISK_FEATURES], dtype=np.float64)


def fit_ridge(
    examples: list[dict[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot fit utility model without examples")
    x = np.vstack([example["x"] for example in examples])
    y = np.asarray([example["utility"] for example in examples])
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    model = Ridge(alpha=alpha, fit_intercept=True).fit(transformed, y)
    return {
        "alpha": float(alpha),
        "feature_names": list(RISK_FEATURES),
        "feature_mean": [float(value) for value in scaler.mean_],
        "feature_scale": [float(value) for value in scaler.scale_],
        "coefficients": [float(value) for value in model.coef_],
        "intercept": float(model.intercept_),
        "training_examples": len(examples),
    }


def predict(model: dict[str, Any], x: np.ndarray) -> float:
    center = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    return float(model["intercept"] + np.dot((x - center) / scale, coefficients))


def candidate_examples(
    coverage_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    physical_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    sample_ids: dict[str, list[str]],
    tasks: list[str],
    capacity: int,
    candidate: str,
    fallback: str,
) -> list[dict[str, Any]]:
    output = []
    for task in tasks:
        for sample_id in sample_ids[task]:
            proof = coverage_lookup[(task, sample_id, candidate, capacity)]
            physical = physical_lookup[(task, sample_id, fallback, capacity)]
            proof_score = 100 * float(proof["score"])
            fallback_score = 100 * float(physical["score"])
            output.append(
                {
                    "task": task,
                    "id": sample_id,
                    "candidate": candidate,
                    "x": feature_vector(proof),
                    "proof_score": proof_score,
                    "fallback_score": fallback_score,
                    "utility": proof_score - fallback_score,
                }
            )
    return output


def fit_candidate_models(
    coverage_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    physical_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    sample_ids: dict[str, list[str]],
    tasks: list[str],
    capacity: int,
    fallback: str,
    alpha: float,
) -> dict[str, dict[str, Any]]:
    return {
        candidate: fit_ridge(
            candidate_examples(
                coverage_lookup,
                physical_lookup,
                sample_ids,
                tasks,
                capacity,
                candidate,
                fallback,
            ),
            alpha,
        )
        for candidate in COVERAGE_METHODS
    }


def predict_task_records(
    coverage_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    physical_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    sample_ids: dict[str, list[str]],
    task: str,
    capacity: int,
    fallback: str,
    models: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for sample_id in sample_ids[task]:
        fallback_row = physical_lookup[
            (task, sample_id, fallback, capacity)
        ]
        candidates = []
        for method in COVERAGE_METHODS:
            proof = coverage_lookup[(task, sample_id, method, capacity)]
            candidates.append(
                {
                    "method": method,
                    "predicted_utility": predict(
                        models[method],
                        feature_vector(proof),
                    ),
                    "score": 100 * float(proof["score"]),
                }
            )
        chosen = max(
            candidates,
            key=lambda value: (
                value["predicted_utility"],
                value["method"],
            ),
        )
        output.append(
            {
                "task": task,
                "id": sample_id,
                "fallback_score": 100 * float(fallback_row["score"]),
                "chosen_method": chosen["method"],
                "predicted_utility": chosen["predicted_utility"],
                "proof_score": chosen["score"],
            }
        )
    return output


def evaluate_predictions(
    records: list[dict[str, Any]],
    margin: float,
) -> dict[str, Any]:
    grouped_route: dict[str, list[float]] = defaultdict(list)
    grouped_fallback: dict[str, list[float]] = defaultdict(list)
    proof_routes = 0
    for record in records:
        use_proof = float(record["predicted_utility"]) >= margin
        proof_routes += int(use_proof)
        grouped_route[str(record["task"])].append(
            float(record["proof_score"])
            if use_proof
            else float(record["fallback_score"])
        )
        grouped_fallback[str(record["task"])].append(
            float(record["fallback_score"])
        )
    task_deltas = {
        task: mean(grouped_route[task]) - mean(grouped_fallback[task])
        for task in sorted(grouped_route)
    }
    return {
        "routed_score": mean(mean(values) for values in grouped_route.values()),
        "fallback_score": mean(
            mean(values) for values in grouped_fallback.values()
        ),
        "delta": mean(task_deltas.values()),
        "worst_task_delta": min(task_deltas.values()),
        "task_deltas": task_deltas,
        "proof_route_coverage": proof_routes / len(records),
        "proof_routes": proof_routes,
        "rows": len(records),
    }


def inner_oof_predictions(
    coverage_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    physical_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    sample_ids: dict[str, list[str]],
    tasks: list[str],
    capacity: int,
    alpha: float,
) -> list[dict[str, Any]]:
    records = []
    for held_task in tasks:
        train_tasks = [task for task in tasks if task != held_task]
        fallback = choose_fallback(
            physical_lookup,
            sample_ids,
            train_tasks,
            capacity,
        )
        models = fit_candidate_models(
            coverage_lookup,
            physical_lookup,
            sample_ids,
            train_tasks,
            capacity,
            fallback,
            alpha,
        )
        records.extend(
            predict_task_records(
                coverage_lookup,
                physical_lookup,
                sample_ids,
                held_task,
                capacity,
                fallback,
                models,
            )
        )
    return records


def select_hyperparameters(
    coverage_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    physical_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    sample_ids: dict[str, list[str]],
    tasks: list[str],
    capacity: int,
) -> dict[str, Any]:
    candidates = []
    for alpha in ALPHAS:
        records = inner_oof_predictions(
            coverage_lookup,
            physical_lookup,
            sample_ids,
            tasks,
            capacity,
            alpha,
        )
        for margin in MARGINS:
            evaluation = evaluate_predictions(records, margin)
            robust = evaluation["worst_task_delta"] >= -0.25
            candidates.append(
                {
                    "alpha": alpha,
                    "margin": margin,
                    "robust_inner_tasks": robust,
                    **evaluation,
                }
            )
    return max(
        candidates,
        key=lambda value: (
            bool(value["robust_inner_tasks"]),
            float(value["delta"]),
            float(value["worst_task_delta"]),
            float(value["proof_route_coverage"]),
            -float(value["alpha"]),
            float(value["margin"]),
        ),
    )


def routed_records_with_fitted_policy(
    coverage_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    physical_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    sample_ids: dict[str, list[str]],
    train_tasks: list[str],
    evaluation_tasks: list[str],
    capacity: int,
    alpha: float,
    margin: float,
) -> tuple[list[dict[str, Any]], str, dict[str, dict[str, Any]]]:
    fallback = choose_fallback(
        physical_lookup,
        sample_ids,
        train_tasks,
        capacity,
    )
    models = fit_candidate_models(
        coverage_lookup,
        physical_lookup,
        sample_ids,
        train_tasks,
        capacity,
        fallback,
        alpha,
    )
    records = []
    for task in evaluation_tasks:
        records.extend(
            predict_task_records(
                coverage_lookup,
                physical_lookup,
                sample_ids,
                task,
                capacity,
                fallback,
                models,
            )
        )
    for record in records:
        record["use_proof"] = (
            float(record["predicted_utility"]) >= float(margin)
        )
        record["routed_score"] = (
            float(record["proof_score"])
            if record["use_proof"]
            else float(record["fallback_score"])
        )
        record["capacity"] = capacity
    return records, fallback, models


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    routed: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    fallback: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    proof_routes = 0
    for record in records:
        key = (
            str(record["model"]),
            str(record["task"]),
            int(record["capacity"]),
        )
        routed[key].append(float(record["routed_score"]))
        fallback[key].append(float(record["fallback_score"]))
        proof_routes += int(record["use_proof"])
    group_deltas = {
        key: mean(routed[key]) - mean(fallback[key]) for key in routed
    }
    model_deltas = {}
    for model in sorted({key[0] for key in group_deltas}):
        model_deltas[model] = mean(
            value
            for key, value in group_deltas.items()
            if key[0] == model
        )
    capacity_deltas = {}
    for capacity in CAPACITIES:
        capacity_deltas[str(capacity)] = mean(
            value
            for key, value in group_deltas.items()
            if key[2] == capacity
        )
    task_deltas = {}
    for task in sorted({key[1] for key in group_deltas}):
        task_deltas[task] = mean(
            value
            for key, value in group_deltas.items()
            if key[1] == task
        )
    return {
        "delta": mean(group_deltas.values()),
        "model_deltas": model_deltas,
        "capacity_deltas": capacity_deltas,
        "task_deltas": task_deltas,
        "worst_task_delta": min(task_deltas.values()),
        "proof_route_coverage": proof_routes / len(records),
        "proof_routes": proof_routes,
        "rows": len(records),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.minimum_overall_proof_route_coverage <= 1:
        raise ValueError("invalid proof-route coverage threshold")

    base_runs = [load_run(Path(value)) for value in args.base_run_dir]
    coverage_runs = [
        load_run(Path(value)) for value in args.coverage_run_dir
    ]
    extension_runs = [
        load_run(Path(value)) for value in args.extension_run_dir
    ]
    base_by_model = {
        str(metrics["model"]): rows for metrics, rows in base_runs
    }
    coverage_by_model = {
        str(metrics["model"]): rows for metrics, rows in coverage_runs
    }
    extension_by_model = {
        str(metrics["model"]): rows for metrics, rows in extension_runs
    }
    if not (
        set(base_by_model)
        == set(coverage_by_model)
        == set(extension_by_model)
    ):
        raise ValueError("model sets differ across calibration runs")
    tasks = [str(value) for value in base_runs[0][0]["tasks"]]
    models = sorted(base_by_model)

    physical_lookups = {}
    coverage_lookups = {}
    sample_ids_by_model = {}
    for model in models:
        extension = extension_by_model[model]
        physical_rows = [
            row
            for row in base_by_model[model] + extension
            if row["method"] in PHYSICAL_METHODS
        ]
        coverage_rows = [
            row
            for row in coverage_by_model[model] + extension
            if row["method"] in COVERAGE_METHODS
        ]
        physical_lookups[model] = row_lookup(physical_rows)
        coverage_lookups[model] = row_lookup(coverage_rows)
        sample_ids: dict[str, list[str]] = {}
        for task in tasks:
            ids = sorted(
                {
                    key[1]
                    for key in physical_lookups[model]
                    if key[0] == task
                }
            )
            sample_ids[task] = ids
        counts = {task: len(ids) for task, ids in sample_ids.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"{model}: unbalanced sample counts {counts}")
        sample_ids_by_model[model] = sample_ids

    nested_records = []
    nested_fold_records = []
    for held_task in tasks:
        train_tasks = [task for task in tasks if task != held_task]
        fold_records = []
        policies = {}
        for model in models:
            for capacity in CAPACITIES:
                hp = select_hyperparameters(
                    coverage_lookups[model],
                    physical_lookups[model],
                    sample_ids_by_model[model],
                    train_tasks,
                    capacity,
                )
                records, fallback, _ = routed_records_with_fitted_policy(
                    coverage_lookups[model],
                    physical_lookups[model],
                    sample_ids_by_model[model],
                    train_tasks,
                    [held_task],
                    capacity,
                    float(hp["alpha"]),
                    float(hp["margin"]),
                )
                for record in records:
                    record["model"] = model
                fold_records.extend(records)
                policies[f"{model}:{capacity}"] = {
                    "alpha": hp["alpha"],
                    "margin": hp["margin"],
                    "fallback": fallback,
                    "inner_oof_delta": hp["delta"],
                    "inner_oof_worst_task_delta": hp[
                        "worst_task_delta"
                    ],
                    "inner_oof_proof_route_coverage": hp[
                        "proof_route_coverage"
                    ],
                }
        fold_aggregate = aggregate_records(fold_records)
        nested_records.extend(fold_records)
        nested_fold_records.append(
            {
                "held_task": held_task,
                "policies_selected_without_held_task": policies,
                **fold_aggregate,
            }
        )
    nested = aggregate_records(nested_records)

    final_policies = {}
    final_calibration_records = []
    for model in models:
        for capacity in CAPACITIES:
            hp = select_hyperparameters(
                coverage_lookups[model],
                physical_lookups[model],
                sample_ids_by_model[model],
                tasks,
                capacity,
            )
            records, fallback, candidate_models = (
                routed_records_with_fitted_policy(
                    coverage_lookups[model],
                    physical_lookups[model],
                    sample_ids_by_model[model],
                    tasks,
                    tasks,
                    capacity,
                    float(hp["alpha"]),
                    float(hp["margin"]),
                )
            )
            for record in records:
                record["model"] = model
            final_calibration_records.extend(records)
            final_policies[f"{model}:{capacity}"] = {
                "fallback_method": fallback,
                "alpha": hp["alpha"],
                "utility_margin": hp["margin"],
                "inner_oof_delta": hp["delta"],
                "inner_oof_worst_task_delta": hp[
                    "worst_task_delta"
                ],
                "inner_oof_proof_route_coverage": hp[
                    "proof_route_coverage"
                ],
                "candidate_models": candidate_models,
            }
    final_calibration = aggregate_records(final_calibration_records)

    nonnegative_nested_tasks = sum(
        value >= 0 for value in nested["task_deltas"].values()
    )
    nonnegative_capacities = sum(
        value >= 0 for value in nested["capacity_deltas"].values()
    )
    status = (
        "PASS"
        if nested["delta"] >= args.minimum_nested_loto_mean_delta
        and nonnegative_nested_tasks
        >= args.minimum_nested_loto_nonnegative_tasks
        and all(
            value >= args.minimum_per_model_nested_delta
            for value in nested["model_deltas"].values()
        )
        and nonnegative_capacities
        >= args.minimum_nonnegative_capacities
        and nested["worst_task_delta"]
        >= args.minimum_worst_task_delta
        and nested["proof_route_coverage"]
        >= args.minimum_overall_proof_route_coverage
        else "FAIL"
    )
    payload = {
        "status": status,
        "protocol": "nested_task_held_out_multi_candidate_utility_router_v1",
        "score_unit": "LongBench QA-F1 points",
        "models": models,
        "tasks": tasks,
        "capacities": list(CAPACITIES),
        "candidate_methods": list(COVERAGE_METHODS),
        "feature_names": list(RISK_FEATURES),
        "predecode_only": True,
        "calibration_implementation_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "input_artifact_sha256": {
            str(Path(value)): run_artifact_sha256(Path(value))
            for value in (
                args.base_run_dir
                + args.coverage_run_dir
                + args.extension_run_dir
            )
        },
        "sample_counts_per_task": {
            model: len(sample_ids_by_model[model][tasks[0]])
            for model in models
        },
        "hyperparameter_grid": {
            "alphas": list(ALPHAS),
            "utility_margins": list(MARGINS),
            "selection": (
                "inner leave-one-task-out; require worst inner task >= -0.25 "
                "when available, then maximize mean utility"
            ),
        },
        "gate_thresholds": {
            "minimum_overall_proof_route_coverage": (
                args.minimum_overall_proof_route_coverage
            ),
            "minimum_nested_loto_mean_delta": (
                args.minimum_nested_loto_mean_delta
            ),
            "minimum_nested_loto_nonnegative_tasks": (
                args.minimum_nested_loto_nonnegative_tasks
            ),
            "minimum_per_model_nested_delta": (
                args.minimum_per_model_nested_delta
            ),
            "minimum_nonnegative_capacities": (
                args.minimum_nonnegative_capacities
            ),
            "minimum_worst_task_delta": args.minimum_worst_task_delta,
        },
        "nested_leave_one_task_out": {
            **nested,
            "nonnegative_tasks": nonnegative_nested_tasks,
            "nonnegative_capacities": nonnegative_capacities,
            "folds": nested_fold_records,
        },
        "final_calibration": final_calibration,
        "final_policy_by_model_capacity": final_policies,
        "formal_evaluation_contract": {
            "selection_start_per_task": 28,
            "samples_per_task": 57,
            "v4_offsets_10_through_27_reclassified_as_development": True,
            "policy_frozen_before_opening_offsets_28_through_84": True,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "nested_delta": nested["delta"],
                "nested_worst_task_delta": nested["worst_task_delta"],
                "nested_proof_route_coverage": nested[
                    "proof_route_coverage"
                ],
            }
        )
    )
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
