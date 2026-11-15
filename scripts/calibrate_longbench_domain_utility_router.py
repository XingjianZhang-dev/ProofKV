#!/usr/bin/env python
"""Calibrate a domain-stable, query-specific proof utility router."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from calibrate_longbench_risk_controller import (
    CAPACITIES,
    COVERAGE_METHODS,
    PHYSICAL_METHODS,
    RISK_FEATURES,
)
from calibrate_longbench_utility_router import (
    ALPHAS,
    MARGINS,
    feature_vector,
    fit_ridge,
    load_run,
    predict,
    row_lookup,
    run_artifact_sha256,
    sha256_file,
)


FOLDS = 5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_QUANTILE = 0.10
MINIMUM_PHASE_DELTA = 0.10
MINIMUM_POOLED_BOOTSTRAP_LOWER = 0.0


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
        "--minimum-cross-validated-delta",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--minimum-per-model-cross-validated-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-nonnegative-capacities",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--minimum-nonnegative-tasks",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--minimum-worst-task-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def stable_seed(*values: object) -> int:
    digest = hashlib.sha256(
        "\x1f".join(str(value) for value in values).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def score(
    lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    task: str,
    sample_id: str,
    method: str,
    capacity: int,
) -> float:
    return 100 * float(
        lookup[(task, sample_id, method, capacity)]["score"]
    )


def choose_fallback(
    physical: dict[tuple[str, str, str, int], dict[str, Any]],
    task: str,
    sample_ids: list[str],
    capacity: int,
) -> str:
    return max(
        PHYSICAL_METHODS,
        key=lambda method: (
            mean(
                score(
                    physical,
                    task,
                    sample_id,
                    method,
                    capacity,
                )
                for sample_id in sample_ids
            ),
            method,
        ),
    )


def paired_deltas(
    coverage: dict[tuple[str, str, str, int], dict[str, Any]],
    physical: dict[tuple[str, str, str, int], dict[str, Any]],
    task: str,
    sample_ids: list[str],
    capacity: int,
    candidate: str,
    fallback: str,
) -> np.ndarray:
    return np.asarray(
        [
            score(coverage, task, sample_id, candidate, capacity)
            - score(physical, task, sample_id, fallback, capacity)
            for sample_id in sample_ids
        ],
        dtype=np.float64,
    )


def lower_bootstrap_mean(
    values: np.ndarray,
    seed: int,
) -> float:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap values must be a nontrivial vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        values.size,
        size=(BOOTSTRAP_REPLICATES, values.size),
    )
    means = values[indices].mean(axis=1)
    return float(
        np.quantile(means, BOOTSTRAP_QUANTILE, method="linear")
    )


def stable_candidate_gate(
    coverage: dict[tuple[str, str, str, int], dict[str, Any]],
    physical: dict[tuple[str, str, str, int], dict[str, Any]],
    model: str,
    task: str,
    prefix_ids: list[str],
    extension_ids: list[str],
    capacity: int,
    fallback: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if not prefix_ids or not extension_ids:
        raise ValueError("both calibration phases must be represented")
    summaries = {}
    eligible = []
    for candidate in COVERAGE_METHODS:
        prefix = paired_deltas(
            coverage,
            physical,
            task,
            prefix_ids,
            capacity,
            candidate,
            fallback,
        )
        extension = paired_deltas(
            coverage,
            physical,
            task,
            extension_ids,
            capacity,
            candidate,
            fallback,
        )
        pooled = np.concatenate([prefix, extension])
        lower = lower_bootstrap_mean(
            pooled,
            stable_seed(
                model,
                task,
                capacity,
                candidate,
                fallback,
                "prefix",
                *prefix_ids,
                "extension",
                *extension_ids,
            ),
        )
        passes = (
            float(prefix.mean()) >= MINIMUM_PHASE_DELTA
            and float(extension.mean()) >= MINIMUM_PHASE_DELTA
            and lower >= MINIMUM_POOLED_BOOTSTRAP_LOWER
        )
        summaries[candidate] = {
            "prefix_delta": float(prefix.mean()),
            "extension_delta": float(extension.mean()),
            "pooled_delta": float(pooled.mean()),
            "pooled_bootstrap_10pct_lower": lower,
            "prefix_samples": int(prefix.size),
            "extension_samples": int(extension.size),
            "eligible": passes,
        }
        if passes:
            eligible.append(candidate)
    return eligible, summaries


def candidate_examples(
    coverage: dict[tuple[str, str, str, int], dict[str, Any]],
    physical: dict[tuple[str, str, str, int], dict[str, Any]],
    task: str,
    sample_ids: list[str],
    capacity: int,
    candidate: str,
    fallback: str,
) -> list[dict[str, Any]]:
    output = []
    for sample_id in sample_ids:
        proof = coverage[(task, sample_id, candidate, capacity)]
        output.append(
            {
                "x": feature_vector(proof),
                "utility": (
                    score(
                        coverage,
                        task,
                        sample_id,
                        candidate,
                        capacity,
                    )
                    - score(
                        physical,
                        task,
                        sample_id,
                        fallback,
                        capacity,
                    )
                ),
            }
        )
    return output


def fit_candidate_models(
    coverage: dict[tuple[str, str, str, int], dict[str, Any]],
    physical: dict[tuple[str, str, str, int], dict[str, Any]],
    task: str,
    sample_ids: list[str],
    capacity: int,
    fallback: str,
    alpha: float,
) -> dict[str, dict[str, Any]]:
    return {
        candidate: fit_ridge(
            candidate_examples(
                coverage,
                physical,
                task,
                sample_ids,
                capacity,
                candidate,
                fallback,
            ),
            alpha,
        )
        for candidate in COVERAGE_METHODS
    }


def evaluate_records(
    records: list[dict[str, Any]],
    margin: float,
) -> dict[str, Any]:
    routed: dict[str, list[float]] = defaultdict(list)
    fallback: dict[str, list[float]] = defaultdict(list)
    proof_routes = 0
    for record in records:
        use_proof = (
            bool(record["eligible"])
            and float(record["predicted_utility"]) >= margin
        )
        proof_routes += int(use_proof)
        task = str(record["task"])
        routed[task].append(
            float(record["proof_score"])
            if use_proof
            else float(record["fallback_score"])
        )
        fallback[task].append(float(record["fallback_score"]))
    task_deltas = {
        task: mean(routed[task]) - mean(fallback[task])
        for task in sorted(routed)
    }
    return {
        "delta": mean(task_deltas.values()),
        "worst_task_delta": min(task_deltas.values()),
        "task_deltas": task_deltas,
        "proof_routes": proof_routes,
        "proof_route_coverage": proof_routes / len(records),
        "rows": len(records),
    }


def phase_fold_ids(
    phase_ids: list[str],
    fold: int,
) -> tuple[list[str], list[str]]:
    train = [
        sample_id
        for index, sample_id in enumerate(phase_ids)
        if index % FOLDS != fold
    ]
    held = [
        sample_id
        for index, sample_id in enumerate(phase_ids)
        if index % FOLDS == fold
    ]
    return train, held


def cross_validated_records(
    coverage: dict[tuple[str, str, str, int], dict[str, Any]],
    physical: dict[tuple[str, str, str, int], dict[str, Any]],
    model: str,
    phase_ids: dict[str, dict[str, list[str]]],
    tasks: list[str],
    capacity: int,
    alpha: float,
) -> list[dict[str, Any]]:
    output = []
    for task in tasks:
        for fold in range(FOLDS):
            prefix_train, prefix_held = phase_fold_ids(
                phase_ids[task]["prefix"],
                fold,
            )
            extension_train, extension_held = phase_fold_ids(
                phase_ids[task]["extension"],
                fold,
            )
            train_ids = [*prefix_train, *extension_train]
            held_ids = [*prefix_held, *extension_held]
            fallback = choose_fallback(
                physical,
                task,
                train_ids,
                capacity,
            )
            eligible, _ = stable_candidate_gate(
                coverage,
                physical,
                model,
                task,
                prefix_train,
                extension_train,
                capacity,
                fallback,
            )
            models = fit_candidate_models(
                coverage,
                physical,
                task,
                train_ids,
                capacity,
                fallback,
                alpha,
            )
            for sample_id in held_ids:
                candidates = [
                    {
                        "method": candidate,
                        "predicted_utility": predict(
                            models[candidate],
                            feature_vector(
                                coverage[
                                    (
                                        task,
                                        sample_id,
                                        candidate,
                                        capacity,
                                    )
                                ]
                            ),
                        ),
                        "score": score(
                            coverage,
                            task,
                            sample_id,
                            candidate,
                            capacity,
                        ),
                    }
                    for candidate in eligible
                ]
                chosen = (
                    max(
                        candidates,
                        key=lambda value: (
                            value["predicted_utility"],
                            value["method"],
                        ),
                    )
                    if candidates
                    else None
                )
                output.append(
                    {
                        "model": model,
                        "task": task,
                        "id": sample_id,
                        "capacity": capacity,
                        "eligible": bool(chosen),
                        "chosen_method": (
                            chosen["method"] if chosen else None
                        ),
                        "predicted_utility": (
                            chosen["predicted_utility"]
                            if chosen
                            else -1.0e30
                        ),
                        "proof_score": (
                            chosen["score"] if chosen else 0.0
                        ),
                        "fallback_score": score(
                            physical,
                            task,
                            sample_id,
                            fallback,
                            capacity,
                        ),
                    }
                )
    return output


def select_hyperparameters(
    coverage: dict[tuple[str, str, str, int], dict[str, Any]],
    physical: dict[tuple[str, str, str, int], dict[str, Any]],
    model: str,
    phase_ids: dict[str, dict[str, list[str]]],
    tasks: list[str],
    capacity: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    records_by_alpha = {}
    for alpha in ALPHAS:
        records = cross_validated_records(
            coverage,
            physical,
            model,
            phase_ids,
            tasks,
            capacity,
            alpha,
        )
        records_by_alpha[alpha] = records
        for margin in MARGINS:
            evaluation = evaluate_records(records, margin)
            candidates.append(
                {
                    "alpha": alpha,
                    "margin": margin,
                    "robust_all_tasks": (
                        evaluation["worst_task_delta"] >= 0.0
                    ),
                    **evaluation,
                }
            )
    selected = max(
        candidates,
        key=lambda value: (
            bool(value["robust_all_tasks"]),
            float(value["delta"]),
            float(value["worst_task_delta"]),
            float(value["proof_route_coverage"]),
            -float(value["alpha"]),
            float(value["margin"]),
        ),
    )
    return selected, records_by_alpha[float(selected["alpha"])]


def aggregate_selected(
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    all_records = []
    model_deltas = {}
    capacity_deltas = {}
    task_groups: dict[str, list[float]] = defaultdict(list)
    for item in selected:
        evaluation = evaluate_records(
            item["records"],
            float(item["margin"]),
        )
        item["evaluation"] = evaluation
        all_records.extend(
            {
                **record,
                "use_proof": (
                    bool(record["eligible"])
                    and float(record["predicted_utility"])
                    >= float(item["margin"])
                ),
            }
            for record in item["records"]
        )
        for task, delta in evaluation["task_deltas"].items():
            task_groups[task].append(delta)
    for model in sorted({item["model"] for item in selected}):
        model_deltas[model] = mean(
            item["evaluation"]["delta"]
            for item in selected
            if item["model"] == model
        )
    for capacity in CAPACITIES:
        capacity_deltas[str(capacity)] = mean(
            item["evaluation"]["delta"]
            for item in selected
            if item["capacity"] == capacity
        )
    task_deltas = {
        task: mean(values) for task, values in sorted(task_groups.items())
    }
    proof_routes = sum(record["use_proof"] for record in all_records)
    return {
        "delta": mean(
            item["evaluation"]["delta"] for item in selected
        ),
        "model_deltas": model_deltas,
        "capacity_deltas": capacity_deltas,
        "task_deltas": task_deltas,
        "worst_task_delta": min(task_deltas.values()),
        "proof_routes": proof_routes,
        "proof_route_coverage": proof_routes / len(all_records),
        "rows": len(all_records),
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
        raise ValueError("model sets differ across calibration artifacts")
    models = sorted(base_by_model)
    tasks = [str(task) for task in base_runs[0][0]["tasks"]]

    physical_by_model = {}
    coverage_lookup_by_model = {}
    phase_ids_by_model = {}
    for model in models:
        extension = extension_by_model[model]
        physical = row_lookup(
            [
                row
                for row in base_by_model[model] + extension
                if row["method"] in PHYSICAL_METHODS
            ]
        )
        coverage = row_lookup(
            [
                row
                for row in coverage_by_model[model] + extension
                if row["method"] in COVERAGE_METHODS
            ]
        )
        physical_by_model[model] = physical
        coverage_lookup_by_model[model] = coverage
        phase_ids = {}
        extension_ids = {
            task: sorted(
                {
                    str(row["id"])
                    for row in extension
                    if row["task"] == task
                    and row["method"] in PHYSICAL_METHODS
                }
            )
            for task in tasks
        }
        for task in tasks:
            prefix_physical = {
                key[1]
                for key in physical
                if key[0] == task
                and key[1] not in set(extension_ids[task])
            }
            prefix_coverage = {
                key[1]
                for key in coverage
                if key[0] == task
                and key[1] not in set(extension_ids[task])
            }
            prefix = sorted(prefix_physical & prefix_coverage)
            phase_ids[task] = {
                "prefix": prefix,
                "extension": extension_ids[task],
                "all": [*prefix, *extension_ids[task]],
            }
            if len(prefix) != 10 or len(extension_ids[task]) != 18:
                raise ValueError(
                    f"{model}/{task}: expected 10+18 calibration samples"
                )
        phase_ids_by_model[model] = phase_ids

    selected_cv = []
    hyperparameters = {}
    for model in models:
        for capacity in CAPACITIES:
            selected, records = select_hyperparameters(
                coverage_lookup_by_model[model],
                physical_by_model[model],
                model,
                phase_ids_by_model[model],
                tasks,
                capacity,
            )
            selected_cv.append(
                {
                    "model": model,
                    "capacity": capacity,
                    "alpha": selected["alpha"],
                    "margin": selected["margin"],
                    "records": records,
                }
            )
            hyperparameters[f"{model}:{capacity}"] = {
                key: value
                for key, value in selected.items()
                if key
                not in {
                    "task_deltas",
                    "rows",
                }
            }
    cross_validated = aggregate_selected(selected_cv)

    final_policies = {}
    eligibility_counts: dict[str, int] = defaultdict(int)
    for model in models:
        physical = physical_by_model[model]
        coverage = coverage_lookup_by_model[model]
        for task in tasks:
            phase_ids = phase_ids_by_model[model][task]
            for capacity in CAPACITIES:
                hp = hyperparameters[f"{model}:{capacity}"]
                fallback = choose_fallback(
                    physical,
                    task,
                    phase_ids["all"],
                    capacity,
                )
                eligible, summaries = stable_candidate_gate(
                    coverage,
                    physical,
                    model,
                    task,
                    phase_ids["prefix"],
                    phase_ids["extension"],
                    capacity,
                    fallback,
                )
                active_eligible = (
                    eligible
                    if float(hp["margin"]) < 1.0e8
                    else []
                )
                for candidate in active_eligible:
                    eligibility_counts[candidate] += 1
                models_for_candidates = fit_candidate_models(
                    coverage,
                    physical,
                    task,
                    phase_ids["all"],
                    capacity,
                    fallback,
                    float(hp["alpha"]),
                )
                final_policies[f"{model}:{task}:{capacity}"] = {
                    "fallback_method": fallback,
                    "eligible_candidate_methods": active_eligible,
                    "stability_eligible_candidate_methods": eligible,
                    "alpha": hp["alpha"],
                    "utility_margin": (
                        hp["margin"] if active_eligible else 1.0e9
                    ),
                    "candidate_stability": summaries,
                    "candidate_models": models_for_candidates,
                }

    nonnegative_tasks = sum(
        value >= 0 for value in cross_validated["task_deltas"].values()
    )
    nonnegative_capacities = sum(
        value >= 0
        for value in cross_validated["capacity_deltas"].values()
    )
    status = (
        "PASS"
        if cross_validated["delta"]
        >= args.minimum_cross_validated_delta
        and all(
            value >= args.minimum_per_model_cross_validated_delta
            for value in cross_validated["model_deltas"].values()
        )
        and nonnegative_capacities
        >= args.minimum_nonnegative_capacities
        and nonnegative_tasks >= args.minimum_nonnegative_tasks
        and cross_validated["worst_task_delta"]
        >= args.minimum_worst_task_delta
        and cross_validated["proof_route_coverage"]
        >= args.minimum_overall_proof_route_coverage
        else "FAIL"
    )
    payload = {
        "status": status,
        "protocol": (
            "two_phase_domain_stability_gate_plus_query_utility_router_V2"
        ),
        "score_unit": "LongBench QA-F1 points",
        "models": models,
        "tasks": tasks,
        "capacities": list(CAPACITIES),
        "candidate_methods": list(COVERAGE_METHODS),
        "feature_names": list(RISK_FEATURES),
        "predecode_only": True,
        "domain_conditioned": True,
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
            "prefix": 10,
            "extension": 18,
            "pooled": 28,
        },
        "domain_stability_gate": {
            "require_prefix_mean_delta_at_least": (
                MINIMUM_PHASE_DELTA
            ),
            "require_extension_mean_delta_at_least": (
                MINIMUM_PHASE_DELTA
            ),
            "require_pooled_bootstrap_lower_at_least": (
                MINIMUM_POOLED_BOOTSTRAP_LOWER
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "one_sided_lower_quantile": BOOTSTRAP_QUANTILE,
            "eligible_policy_count_by_candidate": dict(
                eligibility_counts
            ),
        },
        "cross_validation": {
            "folds": FOLDS,
            "phase_stratified": True,
            "hyperparameter_grid": {
                "alphas": list(ALPHAS),
                "utility_margins": list(MARGINS),
            },
            "selected_by_model_capacity": hyperparameters,
            **cross_validated,
            "nonnegative_tasks": nonnegative_tasks,
            "nonnegative_capacities": nonnegative_capacities,
        },
        "gate_thresholds": {
            "minimum_overall_proof_route_coverage": (
                args.minimum_overall_proof_route_coverage
            ),
            "minimum_cross_validated_delta": (
                args.minimum_cross_validated_delta
            ),
            "minimum_per_model_cross_validated_delta": (
                args.minimum_per_model_cross_validated_delta
            ),
            "minimum_nonnegative_capacities": (
                args.minimum_nonnegative_capacities
            ),
            "minimum_nonnegative_tasks": (
                args.minimum_nonnegative_tasks
            ),
            "minimum_worst_task_delta": (
                args.minimum_worst_task_delta
            ),
        },
        "final_policy_by_model_task_capacity": final_policies,
        "formal_evaluation_contract": {
            "selection_start_per_task": 38,
            "samples_per_task": 47,
            "policy_frozen_before_opening_offsets_38_through_84": True,
            "confirmation_slice": {
                "selection_start_per_task": 28,
                "samples_per_task": 10,
                "excluded_from_final_effect_estimates": True,
            }
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "cross_validated_delta": cross_validated["delta"],
                "worst_task_delta": cross_validated[
                    "worst_task_delta"
                ],
                "proof_route_coverage": cross_validated[
                    "proof_route_coverage"
                ],
            }
        )
    )
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
