#!/usr/bin/env python3
"""Audit and aggregate physical-efficiency evidence from ProofKV v10 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


UTILITY_METHOD = "confirmation_stable_proofkv_v10"
CAPACITIES = (64, 128, 256, 512)
NUMERIC_FIELDS = (
    "score",
    "actual_retained_tokens",
    "actual_retained_ratio",
    "actual_kv_bytes",
    "selection_seconds",
    "generation_seconds",
    "seconds",
    "peak_cuda_memory_gb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--utility-method", default=UTILITY_METHOD)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values]
    if not finite or any(not math.isfinite(value) for value in finite):
        raise ValueError("distribution received empty or non-finite values")
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p95": quantile(finite, 0.95),
        "min": min(finite),
        "max": max(finite),
    }


def assert_close(left: float, right: float, label: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"{label}: {left} != {right}")


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from exc
    return rows


def load_and_validate_run(
    run_dir: Path,
    utility_method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    results_path = run_dir / "results.jsonl"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "run_manifest.json"
    for path in (results_path, metrics_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required completed-run artifact missing: {path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_json_rows(results_path)
    model = str(metrics["model"])
    tasks = [str(task) for task in metrics["tasks"]]
    methods = [str(method) for method in metrics["methods"]]
    capacities = [int(capacity) for capacity in metrics["capacities"]]
    n_per_task = int(metrics["n_per_task"])
    kv_bytes_per_token = float(metrics["kv_bytes_per_token_bf16"])
    if utility_method not in methods:
        raise ValueError(f"{run_dir}: utility method is absent")
    if capacities != list(CAPACITIES):
        raise ValueError(f"{run_dir}: unexpected capacities {capacities}")
    if int(metrics.get("truncated_inputs", 0)) != 0:
        raise ValueError(f"{run_dir}: formal efficiency input was truncated")

    configurations_per_sample = int("full" in methods) + len(
        [method for method in methods if method != "full"]
    ) * len(capacities)
    expected_rows = len(tasks) * n_per_task * configurations_per_sample
    if len(rows) != expected_rows:
        raise ValueError(
            f"{run_dir}: expected {expected_rows} rows, found {len(rows)}"
        )

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
        raise ValueError(f"{run_dir}: duplicate result crossings")
    id_counts = Counter((str(row["task"]), str(row["id"])) for row in rows)
    if set(id_counts.values()) != {configurations_per_sample}:
        raise ValueError(f"{run_dir}: incomplete per-example method crossings")
    if len(id_counts) != len(tasks) * n_per_task:
        raise ValueError(f"{run_dir}: unexpected number of task-example groups")

    empty_predictions = 0
    empty_string_predictions = 0
    whitespace_only_predictions = 0
    empty_prediction_counts: Counter[tuple[str, str, str]] = Counter()
    max_capacity_excess = 0.0
    for row in rows:
        for field in NUMERIC_FIELDS:
            if not math.isfinite(float(row[field])):
                raise ValueError(f"{run_dir}: non-finite {field}")
        prediction = row.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError(f"{run_dir}: non-string prediction")
        if not prediction.strip():
            if not math.isclose(
                float(row["score"]), 0.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{run_dir}: blank prediction has a nonzero score"
                )
            empty_predictions += 1
            empty_string_predictions += int(prediction == "")
            whitespace_only_predictions += int(prediction != "")
            empty_prediction_counts[
                (
                    str(row["task"]),
                    str(row["method"]),
                    str(row["capacity"]),
                )
            ] += 1
        assert_close(
            float(row["seconds"]),
            float(row["selection_seconds"]) + float(row["generation_seconds"]),
            f"{run_dir}: timing decomposition",
        )
        retained = float(row["actual_retained_tokens"])
        prompt_tokens = int(row["prompt_tokens"])
        assert_close(
            float(row["actual_retained_ratio"]),
            retained / prompt_tokens,
            f"{run_dir}: retained ratio",
        )
        assert_close(
            float(row["actual_kv_bytes"]),
            retained * kv_bytes_per_token,
            f"{run_dir}: physical KV bytes",
        )
        if str(row["method"]) == "full":
            assert_close(retained, prompt_tokens, f"{run_dir}: full KV tokens")
        else:
            capacity = int(row["capacity"])
            excess = retained - min(capacity, prompt_tokens)
            max_capacity_excess = max(max_capacity_excess, excess)
            if excess > 1e-6:
                raise ValueError(f"{run_dir}: capacity exceeded")

    report = {
        "run_dir": str(run_dir),
        "model": model,
        "tasks": tasks,
        "n_per_task": n_per_task,
        "methods": methods,
        "capacities": capacities,
        "configurations_per_sample": configurations_per_sample,
        "rows": len(rows),
        "task_example_groups": len(id_counts),
        "max_capacity_excess_tokens": max_capacity_excess,
        "empty_predictions": empty_predictions,
        "empty_string_predictions": empty_string_predictions,
        "whitespace_only_predictions": whitespace_only_predictions,
        "empty_prediction_policy": (
            "Blank decoded outputs are retained as real zero-score model "
            "outputs and reported by task, method, and capacity. A blank "
            "prediction with nonzero score is invalid."
        ),
        "empty_predictions_by_task_method_capacity": [
            {
                "task": task,
                "method": method,
                "capacity": capacity,
                "count": count,
            }
            for (task, method, capacity), count in sorted(
                empty_prediction_counts.items()
            )
        ],
        "kv_bytes_per_token_bf16": kv_bytes_per_token,
        "results_sha256": sha256_file(results_path),
        "metrics_sha256": sha256_file(metrics_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "selection_manifest_sha256": metrics["selection_manifest_sha256"],
        "implementation_sha256": manifest["implementation_sha256"],
        "scorer_checkpoint_sha256": manifest["scorer_checkpoint_sha256"],
    }
    return metrics, rows, report


def aggregate_method_rows(
    model: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["task"]), str(row["method"]), str(row["capacity"]))
        ].append(row)
    records = []
    for (task, method, capacity), values in sorted(grouped.items()):
        records.append(
            {
                "model": model,
                "task": task,
                "method": method,
                "capacity": capacity,
                "qa_f1": distribution(100.0 * float(row["score"]) for row in values),
                "actual_retained_tokens": distribution(
                    float(row["actual_retained_tokens"]) for row in values
                ),
                "actual_retained_ratio": distribution(
                    float(row["actual_retained_ratio"]) for row in values
                ),
                "actual_kv_bytes": distribution(
                    float(row["actual_kv_bytes"]) for row in values
                ),
                "selection_seconds": distribution(
                    float(row["selection_seconds"]) for row in values
                ),
                "generation_seconds": distribution(
                    float(row["generation_seconds"]) for row in values
                ),
                "total_seconds": distribution(
                    float(row["seconds"]) for row in values
                ),
                "peak_cuda_memory_gb": distribution(
                    float(row["peak_cuda_memory_gb"]) for row in values
                ),
            }
        )
    return records


def pair_values(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, float]:
    right_total = float(right["seconds"])
    right_peak = float(right["peak_cuda_memory_gb"])
    right_bytes = float(right["actual_kv_bytes"])
    return {
        "qa_f1_delta": 100.0 * (float(left["score"]) - float(right["score"])),
        "actual_retained_tokens_delta": float(left["actual_retained_tokens"])
        - float(right["actual_retained_tokens"]),
        "actual_kv_bytes_delta": float(left["actual_kv_bytes"])
        - right_bytes,
        "selection_seconds_delta": float(left["selection_seconds"])
        - float(right["selection_seconds"]),
        "generation_seconds_delta": float(left["generation_seconds"])
        - float(right["generation_seconds"]),
        "total_seconds_delta": float(left["seconds"]) - right_total,
        "peak_cuda_memory_gb_delta": float(left["peak_cuda_memory_gb"])
        - right_peak,
        "left_total_seconds": float(left["seconds"]),
        "right_total_seconds": right_total,
        "left_peak_cuda_memory_gb": float(left["peak_cuda_memory_gb"]),
        "right_peak_cuda_memory_gb": right_peak,
        "left_actual_kv_bytes": float(left["actual_kv_bytes"]),
        "right_actual_kv_bytes": right_bytes,
        "total_seconds_ratio": float(left["seconds"]) / right_total
        if right_total > 0
        else math.nan,
        "peak_cuda_memory_ratio": float(left["peak_cuda_memory_gb"]) / right_peak
        if right_peak > 0
        else math.nan,
        "actual_kv_bytes_ratio": float(left["actual_kv_bytes"]) / right_bytes
        if right_bytes > 0
        else math.nan,
    }


def summarize_pair_groups(
    pairs: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[tuple(str(pair[field]) for field in group_fields)].append(pair)
    metric_fields = (
        "qa_f1_delta",
        "actual_retained_tokens_delta",
        "actual_kv_bytes_delta",
        "selection_seconds_delta",
        "generation_seconds_delta",
        "total_seconds_delta",
        "peak_cuda_memory_gb_delta",
        "total_seconds_ratio",
        "peak_cuda_memory_ratio",
        "actual_kv_bytes_ratio",
    )
    records = []
    for key, values in sorted(grouped.items()):
        record: dict[str, Any] = dict(zip(group_fields, key))
        record["n"] = len(values)
        record["route_counts"] = dict(
            sorted(Counter(str(value["route"]) for value in values).items())
        )
        record["baseline_method_counts"] = dict(
            sorted(
                Counter(str(value["baseline_method"]) for value in values).items()
            )
        )
        for metric in metric_fields:
            record[metric] = distribution(value[metric] for value in values)
        for label, left_field, right_field in (
            ("total_seconds", "left_total_seconds", "right_total_seconds"),
            (
                "peak_cuda_memory",
                "left_peak_cuda_memory_gb",
                "right_peak_cuda_memory_gb",
            ),
            (
                "actual_kv_bytes",
                "left_actual_kv_bytes",
                "right_actual_kv_bytes",
            ),
        ):
            left_mean = statistics.fmean(
                float(value[left_field]) for value in values
            )
            right_mean = statistics.fmean(
                float(value[right_field]) for value in values
            )
            ratio = left_mean / right_mean if right_mean > 0 else math.nan
            record[f"{label}_ratio_of_means"] = ratio
            record[f"{label}_percent_change_of_means"] = 100.0 * (ratio - 1.0)
        records.append(record)
    return records


def build_pairs(
    model: str,
    rows: list[dict[str, Any]],
    utility_method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = {
        (
            str(row["task"]),
            str(row["id"]),
            str(row["method"]),
            str(row["capacity"]),
        ): row
        for row in rows
    }
    fallback_pairs = []
    full_pairs = []
    for utility in rows:
        if str(utility["method"]) != utility_method:
            continue
        task = str(utility["task"])
        source_id = str(utility["id"])
        capacity = str(utility["capacity"])
        partition = utility.get("selection_partition")
        if not isinstance(partition, dict):
            raise ValueError(f"{model}/{task}/{source_id}: missing partition")
        fallback_method = str(partition["utility_fallback_method"])
        fallback = index.get((task, source_id, fallback_method, capacity))
        full = index.get((task, source_id, "full", "full"))
        if fallback is None or full is None:
            raise ValueError(f"{model}/{task}/{source_id}: paired source missing")
        route = str(partition["utility_route"])
        if route == "physical_fallback":
            if (
                utility["prediction"] != fallback["prediction"]
                or float(utility["score"]) != float(fallback["score"])
                or float(utility["actual_kv_bytes"])
                != float(fallback["actual_kv_bytes"])
            ):
                raise ValueError(
                    f"{model}/{task}/{source_id}: fallback replay mismatch"
                )

        common = {
            "model": model,
            "task": task,
            "id": source_id,
            "capacity": capacity,
            "route": route,
        }
        fallback_pairs.append(
            {
                **common,
                "baseline_method": fallback_method,
                **pair_values(utility, fallback),
            }
        )
        full_pairs.append(
            {
                **common,
                "baseline_method": "full",
                **pair_values(utility, full),
            }
        )
    return fallback_pairs, full_pairs


def equal_weighted_task_macro(
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    by_model_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_model_task[(str(pair["model"]), str(pair["task"]))].append(pair)
    metric_fields = (
        "qa_f1_delta",
        "actual_retained_tokens_delta",
        "actual_kv_bytes_delta",
        "selection_seconds_delta",
        "generation_seconds_delta",
        "total_seconds_delta",
        "peak_cuda_memory_gb_delta",
    )
    task_records = []
    for (model, task), values in sorted(by_model_task.items()):
        record: dict[str, Any] = {
            "model": model,
            "task": task,
            "pairs": len(values),
        }
        for metric in metric_fields:
            record[metric] = statistics.fmean(float(value[metric]) for value in values)
        for label, left_field, right_field in (
            ("total_seconds", "left_total_seconds", "right_total_seconds"),
            (
                "peak_cuda_memory",
                "left_peak_cuda_memory_gb",
                "right_peak_cuda_memory_gb",
            ),
            (
                "actual_kv_bytes",
                "left_actual_kv_bytes",
                "right_actual_kv_bytes",
            ),
        ):
            left_mean = statistics.fmean(
                float(value[left_field]) for value in values
            )
            right_mean = statistics.fmean(
                float(value[right_field]) for value in values
            )
            ratio = left_mean / right_mean if right_mean > 0 else math.nan
            record[f"{label}_ratio_of_means"] = ratio
            record[f"{label}_percent_change_of_means"] = 100.0 * (ratio - 1.0)
        task_records.append(record)
    macro_fields = (
        *metric_fields,
        "total_seconds_ratio_of_means",
        "total_seconds_percent_change_of_means",
        "peak_cuda_memory_ratio_of_means",
        "peak_cuda_memory_percent_change_of_means",
        "actual_kv_bytes_ratio_of_means",
        "actual_kv_bytes_percent_change_of_means",
    )
    macro = {
        metric: statistics.fmean(float(record[metric]) for record in task_records)
        for metric in macro_fields
    }
    return {
        "weighting": (
            "mean over samples and capacities within each model-task, "
            "then equal mean over model-task groups"
        ),
        "model_task_groups": len(task_records),
        "model_task_records": task_records,
        "macro": macro,
    }


def main() -> None:
    args = parse_args()
    run_reports = []
    aggregate_records = []
    fallback_pairs = []
    full_pairs = []
    selection_hashes = set()
    implementation_hashes = set()
    scorer_hashes = set()
    total_rows = 0

    for value in args.run_dir:
        run_dir = Path(value)
        metrics, rows, report = load_and_validate_run(
            run_dir, args.utility_method
        )
        run_reports.append(report)
        total_rows += len(rows)
        selection_hashes.add(str(report["selection_manifest_sha256"]))
        implementation_hashes.add(str(report["implementation_sha256"]))
        scorer_hashes.add(str(report["scorer_checkpoint_sha256"]))
        model = str(metrics["model"])
        aggregate_records.extend(aggregate_method_rows(model, rows))
        run_fallback, run_full = build_pairs(model, rows, args.utility_method)
        fallback_pairs.extend(run_fallback)
        full_pairs.extend(run_full)

    if len(selection_hashes) != 1:
        raise ValueError("run inputs use different selection manifests")
    if len(implementation_hashes) != 1:
        raise ValueError("run inputs use different evaluator implementations")
    if len(scorer_hashes) != 1:
        raise ValueError("run inputs use different scorer checkpoints")

    payload = {
        "status": "PASS",
        "protocol": "proofkv_v10_physical_efficiency_audit_v1",
        "analysis_role": "descriptive_postformal_physical_efficiency",
        "analysis_implementation_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "utility_method": args.utility_method,
        "input_runs": run_reports,
        "validation": {
            "completed_runs": len(run_reports),
            "rows": total_rows,
            "selection_manifest_sha256": next(iter(selection_hashes)),
            "implementation_sha256": next(iter(implementation_hashes)),
            "scorer_checkpoint_sha256": next(iter(scorer_hashes)),
            "checks": [
                "completed metrics and run manifest present",
                "exact method/capacity/example crossing with no duplicates",
                (
                    "finite row metrics; blank decoded outputs retained as "
                    "zero-score outputs and explicitly reported"
                ),
                "selection plus generation timing decomposition",
                "retained-token ratio and exact physical KV-byte accounting",
                "capacity compliance and full-cache token equality",
                "physical-fallback prediction, score, and byte replay equality",
            ],
        },
        "measurement_scope": {
            "row_timings": (
                "matched offline evaluation timings; use the separate H100 "
                "serving experiment for deployment-throughput claims"
            ),
            "peak_cuda_memory": (
                "PyTorch peak allocated memory includes model and prefill; "
                "it is not an isolated post-prefill KV-cache measurement"
            ),
            "physical_kv_bytes": (
                "exact retained tokens multiplied by model-specific BF16 "
                "KV bytes per token"
            ),
        },
        "method_task_capacity_records": aggregate_records,
        "utility_vs_frozen_physical_fallback": {
            "pair_groups": summarize_pair_groups(
                fallback_pairs, ("model", "task", "capacity")
            ),
            "equal_weighted_task_macro": equal_weighted_task_macro(
                fallback_pairs
            ),
        },
        "utility_vs_full_cache": {
            "pair_groups": summarize_pair_groups(
                full_pairs, ("model", "task", "capacity")
            ),
            "equal_weighted_task_macro": equal_weighted_task_macro(full_pairs),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "runs": len(run_reports),
                "rows": total_rows,
                "fallback_pairs": len(fallback_pairs),
                "full_pairs": len(full_pairs),
                "out": str(out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
