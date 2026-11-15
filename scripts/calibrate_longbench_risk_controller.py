#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


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
)
CAPACITIES = (64, 128, 256, 512)
RISK_FEATURES = (
    "overlay_confidence",
    "overlay_top_proof_probability",
    "overlay_proof_probability_margin",
    "overlay_root_top_probability",
    "overlay_root_probability_margin",
    "overlay_terminal_top_probability",
    "overlay_terminal_probability_margin",
    "overlay_num_proofs",
    "overlay_top_proof_spans",
    "proof_core_fraction",
    "coverage_auxiliary_fraction",
    "context_log_tokens",
    "question_budget_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a proof-plus-coverage residual and calibrate a transparent "
            "pre-decoding physical-fallback controller."
        )
    )
    parser.add_argument("--base-run-dir", action="append", required=True)
    parser.add_argument("--coverage-run-dir", action="append", required=True)
    parser.add_argument("--minimum-proof-route-coverage", type=float, default=0.20)
    parser.add_argument(
        "--minimum-calibration-routed-delta",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--minimum-loto-mean-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-loto-nonnegative-tasks",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--minimum-per-model-routed-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-nonnegative-capacities",
        type=int,
        default=3,
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_pilot(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_files = sorted(run_dir.glob("pilot_metrics_n*.json"))
    if len(metrics_files) != 1:
        raise ValueError(f"{run_dir}: expected one pilot_metrics_n*.json")
    metrics = json.loads(metrics_files[0].read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return metrics, rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty collection")
    return sum(values) / len(values)


def task_macro(
    rows: list[dict[str, Any]],
    tasks: Iterable[str],
    method: str,
    capacity: int,
) -> float:
    task_scores = []
    for task in tasks:
        values = [
            100 * float(row["score"])
            for row in rows
            if row["task"] == task
            and row["method"] == method
            and int(row["capacity"]) == capacity
        ]
        if not values:
            raise ValueError(f"missing {task}/{method}/{capacity}")
        task_scores.append(mean(values))
    return mean(task_scores)


def choose_fallbacks(
    base_by_model: dict[str, list[dict[str, Any]]],
    tasks: list[str],
) -> dict[tuple[str, int], str]:
    output = {}
    for model, rows in base_by_model.items():
        for capacity in CAPACITIES:
            output[(model, capacity)] = max(
                PHYSICAL_METHODS,
                key=lambda method: (
                    task_macro(rows, tasks, method, capacity),
                    method,
                ),
            )
    return output


def choose_coverage_methods(
    coverage_by_model: dict[str, list[dict[str, Any]]],
    base_by_model: dict[str, list[dict[str, Any]]],
    tasks: list[str],
    fallbacks: dict[tuple[str, int], str],
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    records = []
    selected: dict[int, str] = {}
    for capacity in CAPACITIES:
        capacity_records = []
        for method in COVERAGE_METHODS:
            deltas = []
            wins = 0
            for model, coverage_rows in coverage_by_model.items():
                base_rows = base_by_model[model]
                proof_score = task_macro(
                    coverage_rows, tasks, method, capacity
                )
                fallback = fallbacks[(model, capacity)]
                fallback_score = task_macro(
                    base_rows, tasks, fallback, capacity
                )
                delta = proof_score - fallback_score
                deltas.append(delta)
                wins += int(delta > 0)
            record = {
                "capacity": capacity,
                "method": method,
                "mean_delta_vs_calibrated_physical": mean(deltas),
                "wins": wins,
                "conditions": len(deltas),
                "deltas": deltas,
            }
            records.append(record)
            capacity_records.append(record)
        selected[capacity] = str(
            max(
                capacity_records,
                key=lambda record: (
                    record["mean_delta_vs_calibrated_physical"],
                    record["wins"],
                    record["method"],
                ),
            )["method"]
        )
    return selected, records


def risk_features(row: dict[str, Any]) -> dict[str, float]:
    partition = row.get("selection_partition") or {}
    capacity = int(row["capacity"])
    features = {
        key: float(partition[key])
        for key in RISK_FEATURES
        if key in partition
    }
    features.update(
        {
            "proof_core_fraction": float(partition["proof_core_tokens"]) / capacity,
            "coverage_auxiliary_fraction": float(
                partition["coverage_auxiliary_tokens"]
            )
            / capacity,
            "context_log_tokens": math.log1p(int(row["context_tokens"])),
            "question_budget_fraction": float(row["question_tokens"]) / capacity,
        }
    )
    missing = set(RISK_FEATURES) - set(features)
    if missing:
        raise ValueError(f"missing predecode risk features: {sorted(missing)}")
    if any(not math.isfinite(value) for value in features.values()):
        raise ValueError("non-finite risk feature")
    return features


def build_observations(
    base_by_model: dict[str, list[dict[str, Any]]],
    coverage_by_model: dict[str, list[dict[str, Any]]],
    proof_methods: dict[int, str],
    fallbacks: dict[tuple[str, int], str],
    tasks: list[str],
) -> list[dict[str, Any]]:
    observations = []
    for model, coverage_rows in coverage_by_model.items():
        base_lookup = {
            (str(row["task"]), str(row["id"]), str(row["method"]), int(row["capacity"])): row
            for row in base_by_model[model]
            if row["method"] != "full"
        }
        for row in coverage_rows:
            capacity = int(row["capacity"])
            if (
                row["method"] != proof_methods[capacity]
                or row["task"] not in tasks
            ):
                continue
            fallback_method = fallbacks[(model, capacity)]
            fallback = base_lookup[
                (str(row["task"]), str(row["id"]), fallback_method, capacity)
            ]
            observations.append(
                {
                    "model": model,
                    "task": str(row["task"]),
                    "id": str(row["id"]),
                    "capacity": capacity,
                    "proof_score": 100 * float(row["score"]),
                    "fallback_score": 100 * float(fallback["score"]),
                    "features": risk_features(row),
                }
            )
    return observations


def routed_macro(
    observations: list[dict[str, Any]],
    feature: str | None,
    direction: str | None,
    threshold: float | None,
) -> tuple[float, float]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    proof_routes = 0
    for row in observations:
        if feature is None:
            use_proof = False
        elif direction == "high":
            use_proof = float(row["features"][feature]) >= float(threshold)
        elif direction == "low":
            use_proof = float(row["features"][feature]) <= float(threshold)
        else:
            raise ValueError("direction must be high or low")
        proof_routes += int(use_proof)
        grouped[(row["model"], row["capacity"], row["task"])].append(
            row["proof_score"] if use_proof else row["fallback_score"]
        )
    return mean(mean(values) for values in grouped.values()), proof_routes / len(observations)


def quantile_thresholds(values: list[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no feature values")
    thresholds = {ordered[0], ordered[-1]}
    for index in range(11):
        position = round(index * (len(ordered) - 1) / 10)
        thresholds.add(ordered[position])
    return sorted(thresholds)


def fit_rule(
    observations: list[dict[str, Any]],
    minimum_proof_route_coverage: float,
) -> dict[str, Any]:
    candidates = []
    for feature in RISK_FEATURES:
        values = [float(row["features"][feature]) for row in observations]
        for direction in ("high", "low"):
            for threshold in quantile_thresholds(values):
                score, coverage = routed_macro(
                    observations, feature, direction, threshold
                )
                if coverage + 1e-12 < minimum_proof_route_coverage:
                    continue
                candidates.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "score": score,
                        "proof_route_coverage": coverage,
                    }
                )
    if not candidates:
        raise ValueError("no risk rule satisfies the proof-route coverage floor")
    return max(
        candidates,
        key=lambda row: (
            row["score"],
            row["proof_route_coverage"],
            row["feature"],
            row["direction"],
            row["threshold"],
        ),
    )


def fit_rules_by_capacity(
    observations: list[dict[str, Any]],
    minimum_proof_route_coverage: float,
) -> dict[int, dict[str, Any]]:
    output = {}
    for capacity in CAPACITIES:
        capacity_observations = [
            row
            for row in observations
            if row["capacity"] == capacity
        ]
        if not capacity_observations:
            raise ValueError(f"no risk observations for capacity {capacity}")
        output[capacity] = fit_rule(
            capacity_observations,
            minimum_proof_route_coverage,
        )
    return output


def routed_macro_by_capacity_rules(
    observations: list[dict[str, Any]],
    rules: dict[int, dict[str, Any]],
) -> tuple[float, float]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    proof_routes = 0
    for row in observations:
        rule = rules[int(row["capacity"])]
        feature_value = float(row["features"][str(rule["feature"])])
        direction = str(rule["direction"])
        threshold = float(rule["threshold"])
        if direction == "high":
            use_proof = feature_value >= threshold
        elif direction == "low":
            use_proof = feature_value <= threshold
        else:
            raise ValueError("direction must be high or low")
        proof_routes += int(use_proof)
        grouped[(row["model"], row["capacity"], row["task"])].append(
            row["proof_score"] if use_proof else row["fallback_score"]
        )
    return (
        mean(mean(values) for values in grouped.values()),
        proof_routes / len(observations),
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.minimum_proof_route_coverage <= 1.0:
        raise ValueError("minimum proof-route coverage must be in (0, 1]")
    if not math.isfinite(args.minimum_calibration_routed_delta):
        raise ValueError("minimum calibration delta must be finite")
    if not math.isfinite(args.minimum_loto_mean_delta):
        raise ValueError("minimum leave-one-task-out delta must be finite")
    if not math.isfinite(args.minimum_per_model_routed_delta):
        raise ValueError("minimum per-model routed delta must be finite")
    if not 0 <= args.minimum_nonnegative_capacities <= len(CAPACITIES):
        raise ValueError(
            "minimum nonnegative capacities is outside the capacity axis"
        )
    base_metrics_rows = [load_pilot(Path(value)) for value in args.base_run_dir]
    coverage_metrics_rows = [
        load_pilot(Path(value)) for value in args.coverage_run_dir
    ]
    base_by_model = {
        str(metrics["model"]): rows for metrics, rows in base_metrics_rows
    }
    coverage_by_model = {
        str(metrics["model"]): rows
        for metrics, rows in coverage_metrics_rows
    }
    if set(base_by_model) != set(coverage_by_model):
        raise ValueError("base and coverage model sets differ")
    tasks = list(base_metrics_rows[0][0]["tasks"])
    if not 0 <= args.minimum_loto_nonnegative_tasks <= len(tasks):
        raise ValueError(
            "minimum nonnegative leave-one-task-out tasks is outside "
            f"[0, {len(tasks)}]"
        )
    if any(list(metrics["tasks"]) != tasks for metrics, _ in base_metrics_rows + coverage_metrics_rows):
        raise ValueError("task axes differ")

    fallbacks = choose_fallbacks(base_by_model, tasks)
    proof_methods, coverage_records = choose_coverage_methods(
        coverage_by_model,
        base_by_model,
        tasks,
        fallbacks,
    )
    observations = build_observations(
        base_by_model,
        coverage_by_model,
        proof_methods,
        fallbacks,
        tasks,
    )
    final_rules = fit_rules_by_capacity(
        observations,
        args.minimum_proof_route_coverage,
    )
    calibration_routed_score, calibration_proof_coverage = (
        routed_macro_by_capacity_rules(observations, final_rules)
    )
    fallback_score, _ = routed_macro(observations, None, None, None)
    proof_score = mean(
        mean(values)
        for values in (
            [
                row["proof_score"]
                for row in observations
                if (
                    row["model"],
                    row["capacity"],
                    row["task"],
                )
                == group
            ]
            for group in sorted(
                {
                    (row["model"], row["capacity"], row["task"])
                    for row in observations
                }
            )
        )
    )

    loto_records = []
    for held_task in tasks:
        train_tasks = [task for task in tasks if task != held_task]
        fold_fallbacks = choose_fallbacks(base_by_model, train_tasks)
        fold_methods, _ = choose_coverage_methods(
            coverage_by_model,
            base_by_model,
            train_tasks,
            fold_fallbacks,
        )
        train_observations = build_observations(
            base_by_model,
            coverage_by_model,
            fold_methods,
            fold_fallbacks,
            train_tasks,
        )
        fold_rules = fit_rules_by_capacity(
            train_observations,
            args.minimum_proof_route_coverage,
        )
        held_observations = build_observations(
            base_by_model,
            coverage_by_model,
            fold_methods,
            fold_fallbacks,
            [held_task],
        )
        routed_score, proof_coverage = routed_macro_by_capacity_rules(
            held_observations,
            fold_rules,
        )
        held_fallback_score, _ = routed_macro(
            held_observations, None, None, None
        )
        loto_records.append(
            {
                "held_task": held_task,
                "proof_methods_selected_without_held_task": {
                    str(capacity): method
                    for capacity, method in sorted(fold_methods.items())
                },
                "risk_rules_selected_without_held_task": {
                    str(capacity): rule
                    for capacity, rule in sorted(fold_rules.items())
                },
                "held_routed_score": routed_score,
                "held_fallback_score": held_fallback_score,
                "held_delta": routed_score - held_fallback_score,
                "held_proof_route_coverage": proof_coverage,
            }
        )

    loto_delta = mean(record["held_delta"] for record in loto_records)
    loto_nonnegative_tasks = sum(
        record["held_delta"] >= 0.0 for record in loto_records
    )
    calibration_routed_delta = (
        calibration_routed_score - fallback_score
    )
    per_model_records = []
    for model in sorted(base_by_model):
        model_observations = [
            row for row in observations if row["model"] == model
        ]
        routed_score, proof_coverage = routed_macro_by_capacity_rules(
            model_observations,
            final_rules,
        )
        model_fallback_score, _ = routed_macro(
            model_observations,
            None,
            None,
            None,
        )
        per_model_records.append(
            {
                "model": model,
                "routed_score": routed_score,
                "fallback_score": model_fallback_score,
                "routed_delta": routed_score - model_fallback_score,
                "proof_route_coverage": proof_coverage,
            }
        )
    per_capacity_records = []
    for capacity in CAPACITIES:
        capacity_observations = [
            row
            for row in observations
            if row["capacity"] == capacity
        ]
        routed_score, proof_coverage = routed_macro_by_capacity_rules(
            capacity_observations,
            final_rules,
        )
        capacity_fallback_score, _ = routed_macro(
            capacity_observations,
            None,
            None,
            None,
        )
        per_capacity_records.append(
            {
                "capacity": capacity,
                "routed_score": routed_score,
                "fallback_score": capacity_fallback_score,
                "routed_delta": routed_score - capacity_fallback_score,
                "proof_route_coverage": proof_coverage,
            }
        )
    nonnegative_capacities = sum(
        record["routed_delta"] >= 0.0
        for record in per_capacity_records
    )
    status = (
        "PASS"
        if calibration_routed_delta
        >= args.minimum_calibration_routed_delta
        and loto_delta >= args.minimum_loto_mean_delta
        and loto_nonnegative_tasks
        >= args.minimum_loto_nonnegative_tasks
        and all(
            record["routed_delta"]
            >= args.minimum_per_model_routed_delta
            for record in per_model_records
        )
        and nonnegative_capacities
        >= args.minimum_nonnegative_capacities
        and all(
            rule["proof_route_coverage"]
            >= args.minimum_proof_route_coverage
            for rule in final_rules.values()
        )
        else "FAIL"
    )
    payload = {
        "status": status,
        "protocol": (
            "task_held_out_capacity_adaptive_residual_and_risk_calibration_v3"
        ),
        "score_unit": "LongBench QA-F1 points",
        "models": sorted(base_by_model),
        "tasks": tasks,
        "capacities": list(CAPACITIES),
        "predecode_features_only": list(RISK_FEATURES),
        "minimum_proof_route_coverage": args.minimum_proof_route_coverage,
        "gate_thresholds": {
            "minimum_calibration_routed_delta": (
                args.minimum_calibration_routed_delta
            ),
            "minimum_loto_mean_delta": args.minimum_loto_mean_delta,
            "minimum_loto_nonnegative_tasks": (
                args.minimum_loto_nonnegative_tasks
            ),
            "minimum_per_model_routed_delta": (
                args.minimum_per_model_routed_delta
            ),
            "minimum_nonnegative_capacities": (
                args.minimum_nonnegative_capacities
            ),
        },
        "coverage_architecture_records": coverage_records,
        "coverage_architecture_selection_scope": (
            "one residual mode per capacity, shared across models and tasks"
        ),
        "proof_method_by_capacity": {
            str(capacity): method
            for capacity, method in sorted(proof_methods.items())
        },
        "fallback_by_model_capacity": {
            f"{model}:{capacity}": method
            for (model, capacity), method in sorted(fallbacks.items())
        },
        "always_physical_fallback_score": fallback_score,
        "always_selected_proof_score": proof_score,
        "final_risk_rule_by_capacity": {
            str(capacity): rule
            for capacity, rule in sorted(final_rules.items())
        },
        "calibration_routed_score": calibration_routed_score,
        "calibration_proof_route_coverage": calibration_proof_coverage,
        "calibration_routed_delta": calibration_routed_delta,
        "per_model_calibration": per_model_records,
        "per_capacity_calibration": per_capacity_records,
        "nonnegative_capacities": nonnegative_capacities,
        "leave_one_task_out": loto_records,
        "leave_one_task_out_mean_delta": loto_delta,
        "leave_one_task_out_nonnegative_tasks": loto_nonnegative_tasks,
        "formal_evaluation_contract": {
            "selection_start_per_task": 10,
            "samples_per_task": 75,
            "controller_and_fallback_map_frozen_before_opening_evaluation": True,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "proof_method_by_capacity": {
                    str(capacity): method
                    for capacity, method in sorted(proof_methods.items())
                },
                "calibration_routed_delta": calibration_routed_delta,
                "leave_one_task_out_mean_delta": loto_delta,
            }
        )
    )
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
