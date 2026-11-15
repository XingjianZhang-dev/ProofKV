#!/usr/bin/env python
"""Independent audit of the untouched utility-routed LongBench matrix."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_longbench_risk_formal import expected_rows
from audit_longbench_structured_common import (
    aggregate,
    assert_close,
    best_qa_f1,
    load_rows,
    sha256_file,
)


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
UTILITY_METHOD = "utility_calibrated_proof_coverage"
CONFIRMATION_METHODS = (
    "full",
    *PHYSICAL_METHODS,
    *COVERAGE_METHODS,
    UTILITY_METHOD,
)
FORMAL_METHODS = (
    "full",
    *PHYSICAL_METHODS,
    *COVERAGE_METHODS,
    UTILITY_METHOD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--utility-controller", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument(
        "--slice",
        choices=("formal", "confirmation"),
        default="formal",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def feature_value(name: str, row: dict[str, Any]) -> float:
    partition = row["selection_partition"]
    capacity = int(row["capacity"])
    if name == "proof_core_fraction":
        return float(partition["proof_core_tokens"]) / capacity
    if name == "coverage_auxiliary_fraction":
        return float(partition["coverage_auxiliary_tokens"]) / capacity
    if name == "context_log_tokens":
        return math.log1p(int(row["context_tokens"]))
    if name == "question_budget_fraction":
        return float(row["question_tokens"]) / capacity
    return float(partition[name])


def predicted_utility(
    utility_model: dict[str, Any],
    candidate_row: dict[str, Any],
) -> float:
    values = {
        name: feature_value(name, candidate_row)
        for name in utility_model["feature_names"]
    }
    return predicted_utility_from_features(utility_model, values)


def predicted_utility_from_features(
    utility_model: dict[str, Any],
    feature_values: dict[str, Any],
) -> float:
    names = list(utility_model["feature_names"])
    means = [float(value) for value in utility_model["feature_mean"]]
    scales = [float(value) for value in utility_model["feature_scale"]]
    coefficients = [
        float(value) for value in utility_model["coefficients"]
    ]
    if not (
        len(names)
        == len(means)
        == len(scales)
        == len(coefficients)
    ):
        raise ValueError("invalid serialized utility model")
    if any(scale <= 0 or not math.isfinite(scale) for scale in scales):
        raise ValueError("invalid serialized utility feature scale")
    value = float(utility_model["intercept"])
    for name, center, scale, coefficient in zip(
        names,
        means,
        scales,
        coefficients,
    ):
        value += (
            (float(feature_values[name]) - center)
            / scale
            * coefficient
        )
    if not math.isfinite(value):
        raise ValueError("non-finite independently predicted utility")
    return value


def metrics_path(run_dir: Path) -> Path:
    path = run_dir / "metrics.json"
    if path.is_file():
        return path
    candidates = sorted(run_dir.glob("pilot_metrics_s*_n*.json"))
    if len(candidates) != 1:
        raise ValueError(f"{run_dir}: expected one completed metrics file")
    return candidates[0]


def check_file_hashes(
    run_dir: Path,
    manifest: dict[str, Any],
) -> tuple[int, int]:
    implementation = manifest.get("implementation_sha256")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError(f"{run_dir}: implementation hashes missing")
    for path_value, expected_hash in implementation.items():
        path = Path(path_value)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"{run_dir}: implementation hash mismatch: {path}")
    checkpoint = manifest.get("scorer_checkpoint_sha256")
    checkpoint_root = Path(manifest["arguments"]["scorer_checkpoint"])
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(f"{run_dir}: scorer checkpoint hashes missing")
    for relative_path, expected_hash in checkpoint.items():
        path = checkpoint_root / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"{run_dir}: checkpoint hash mismatch: {path}")
    return len(implementation), len(checkpoint)


def audit_run(
    run_dir: Path,
    selection: dict[str, Any],
    selection_sha256: str,
    controller: dict[str, Any],
    controller_sha256: str,
    slice_name: str,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path(run_dir).read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    rows = load_rows(run_dir / "results.jsonl")
    tasks = [str(task) for task in selection["tasks"]]
    model = str(metrics["model"])
    if (
        controller.get("domain_conditioned") is True
        and controller.get("tasks") != tasks
    ):
        raise ValueError(f"{run_dir}: controller task domains differ")
    formal_contract = controller["formal_evaluation_contract"]
    contract = (
        formal_contract
        if slice_name == "formal"
        else formal_contract["confirmation_slice"]
    )
    methods = (
        list(FORMAL_METHODS)
        if slice_name == "formal"
        else list(CONFIRMATION_METHODS)
    )
    start = int(metrics["selection_start_per_task"])
    n_per_task = int(metrics["n_per_task"])
    if (
        start != int(contract["selection_start_per_task"])
        or n_per_task != int(contract["samples_per_task"])
    ):
        raise ValueError(f"{run_dir}: {slice_name} slice contract differs")
    if slice_name == "formal":
        if start + n_per_task != int(selection["max_samples_per_task"]):
            raise ValueError(f"{run_dir}: formal suffix is not complete")
    elif (
        contract.get("excluded_from_final_effect_estimates") is not True
        or start + n_per_task
        > int(formal_contract["selection_start_per_task"])
    ):
        raise ValueError(f"{run_dir}: confirmation exclusion differs")
    if metrics["methods"] != methods:
        raise ValueError(f"{run_dir}: method axis differs")
    if metrics["capacities"] != list(CAPACITIES):
        raise ValueError(f"{run_dir}: capacity axis differs")
    if metrics["tasks"] != tasks or int(metrics["truncated_inputs"]) != 0:
        raise ValueError(f"{run_dir}: task/truncation contract differs")
    if (
        metrics["selection_manifest_sha256"] != selection_sha256
        or manifest["selection_manifest_sha256"] != selection_sha256
    ):
        raise ValueError(f"{run_dir}: selection hash mismatch")
    if manifest.get("risk_controller_sha256") != controller_sha256:
        raise ValueError(f"{run_dir}: utility-controller hash mismatch")
    if (
        metrics.get("tokenization_contract")
        != "kvpress_split_context_question_token_ids_v1"
    ):
        raise ValueError(f"{run_dir}: tokenization contract differs")
    implementation_count, checkpoint_count = check_file_hashes(
        run_dir,
        manifest,
    )

    expected_by_id = expected_rows(
        selection,
        model,
        start,
        n_per_task,
    )
    configurations = {("full", "full")} | {
        (method, str(capacity))
        for method in methods
        if method != "full"
        for capacity in CAPACITIES
    }
    expected_count = len(tasks) * n_per_task * len(configurations)
    if len(rows) != expected_count:
        raise ValueError(f"{run_dir}: {len(rows)} != {expected_count}")
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
    lookup = dict(zip(keys, rows))

    grouped: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    kv_bytes_per_token = int(metrics["kv_bytes_per_token_bf16"])
    max_excess = 0.0
    proof_routes = 0
    fallback_routes = 0
    raw_empty = 0
    whitespace_only = 0
    selected_candidates: dict[str, int] = defaultdict(int)
    proof_routes_without_source_replay = 0
    for row in rows:
        task_id = (str(row["task"]), str(row["id"]))
        if task_id not in expected_by_id:
            raise ValueError(f"{run_dir}: unexpected sample {task_id}")
        grouped[task_id].add((str(row["method"]), str(row["capacity"])))
        expected = expected_by_id[task_id]
        tokens = expected["token_counts"]
        if (
            int(row["prompt_tokens"]) != int(tokens["prompt"])
            or int(row["question_tokens"]) != int(tokens["question"])
        ):
            raise ValueError(f"{run_dir}: tokenization drift for {task_id}")
        answers = [str(value) for value in row["answers"]]
        if answers != expected["answers"]:
            raise ValueError(f"{run_dir}: answer drift for {task_id}")
        prediction = row.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError(f"{run_dir}: invalid prediction for {task_id}")
        raw_empty += int(prediction == "")
        whitespace_only += int(prediction != "" and not prediction.strip())
        assert_close(
            row["score"],
            best_qa_f1(prediction, answers),
            f"{run_dir}: independent F1 for {task_id}",
        )
        for field in (
            "score",
            "actual_retained_tokens",
            "actual_retained_ratio",
            "actual_kv_bytes",
            "seconds",
            "selection_seconds",
            "generation_seconds",
            "peak_cuda_memory_gb",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"{run_dir}: non-finite {field}")
        assert_close(
            row["seconds"],
            float(row["selection_seconds"])
            + float(row["generation_seconds"]),
            f"{run_dir}: timing decomposition for {task_id}",
        )
        retained = float(row["actual_retained_tokens"])
        prompt_tokens = int(row["prompt_tokens"])
        assert_close(
            row["actual_retained_ratio"],
            retained / prompt_tokens,
            f"{run_dir}: retained ratio for {task_id}",
        )
        assert_close(
            row["actual_kv_bytes"],
            retained * kv_bytes_per_token,
            f"{run_dir}: KV bytes for {task_id}",
        )
        if row["method"] == "full":
            assert_close(
                retained,
                prompt_tokens,
                f"{run_dir}: full retained tokens for {task_id}",
            )
        else:
            excess = retained - min(int(row["capacity"]), prompt_tokens)
            max_excess = max(max_excess, excess)
            if excess > 1e-6:
                raise ValueError(f"{run_dir}: capacity exceeded for {task_id}")

        if row["method"] != UTILITY_METHOD:
            continue
        capacity = int(row["capacity"])
        domain_conditioned = controller.get("domain_conditioned") is True
        policy = controller[
            (
                "final_policy_by_model_task_capacity"
                if domain_conditioned
                else "final_policy_by_model_capacity"
            )
        ][
            (
                f"{model}:{row['task']}:{capacity}"
                if domain_conditioned
                else f"{model}:{capacity}"
            )
        ]
        partition = row.get("selection_partition")
        if not isinstance(partition, dict):
            raise ValueError(f"{run_dir}: utility metadata missing")
        eligible = policy.get(
            "eligible_candidate_methods",
            list(COVERAGE_METHODS),
        )
        if domain_conditioned:
            stability = policy.get("candidate_stability")
            stability_eligible = policy.get(
                "stability_eligible_candidate_methods"
            )
            if (
                not isinstance(stability, dict)
                or not isinstance(stability_eligible, list)
                or {
                    method
                    for method, record in stability.items()
                    if isinstance(record, dict)
                    and record.get("eligible") is True
                }
                != set(stability_eligible)
                or not set(eligible).issubset(stability_eligible)
            ):
                raise ValueError(
                    f"{run_dir}: candidate stability metadata differs"
                )
        independently_predicted = {}
        observed_features = partition.get(
            "utility_candidate_feature_values"
        )
        if not isinstance(observed_features, dict) or set(
            observed_features
        ) != set(eligible):
            raise ValueError(f"{run_dir}: candidate feature set differs")
        for candidate in eligible:
            feature_record = observed_features[candidate]
            if not isinstance(feature_record, dict) or set(
                feature_record
            ) != set(controller["feature_names"]):
                raise ValueError(f"{run_dir}: utility feature schema differs")
            candidate_key = (
                str(row["task"]),
                str(row["id"]),
                candidate,
                str(capacity),
            )
            if candidate_key in lookup:
                candidate_row = lookup[candidate_key]
                for feature_name in controller["feature_names"]:
                    assert_close(
                        feature_record[feature_name],
                        feature_value(feature_name, candidate_row),
                        (
                            f"{run_dir}: candidate feature "
                            f"{task_id}/{candidate}/{feature_name}"
                        ),
                    )
            independently_predicted[candidate] = (
                predicted_utility_from_features(
                    policy["candidate_models"][candidate],
                    feature_record,
                )
            )
        observed_predictions = partition.get(
            "utility_candidate_predictions"
        )
        if not isinstance(observed_predictions, dict) or set(
            observed_predictions
        ) != set(eligible):
            raise ValueError(f"{run_dir}: candidate prediction set differs")
        for candidate, value in independently_predicted.items():
            assert_close(
                observed_predictions[candidate],
                value,
                f"{run_dir}: utility prediction {task_id}/{candidate}",
            )
        chosen = (
            max(
                eligible,
                key=lambda candidate: (
                    independently_predicted[candidate],
                    candidate,
                ),
            )
            if eligible
            else None
        )
        margin = float(policy["utility_margin"])
        use_proof = bool(
            chosen is not None
            and independently_predicted[chosen] >= margin
        )
        expected_route = (
            "proof_coverage" if use_proof else "physical_fallback"
        )
        fallback = str(policy["fallback_method"])
        if (
            partition.get("utility_selected_method") != chosen
            or partition.get("utility_route") != expected_route
            or partition.get("utility_fallback_method") != fallback
        ):
            raise ValueError(f"{run_dir}: utility decision mismatch for {task_id}")
        if chosen is None:
            if partition.get("utility_selected_prediction") is not None:
                raise ValueError(
                    f"{run_dir}: blocked group has a selected utility"
                )
        else:
            for feature_name in controller["feature_names"]:
                assert_close(
                    observed_features[chosen][feature_name],
                    feature_value(feature_name, row),
                    (
                        f"{run_dir}: selected-plan feature "
                        f"{task_id}/{feature_name}"
                    ),
                )
            assert_close(
                partition["utility_selected_prediction"],
                independently_predicted[chosen],
                f"{run_dir}: selected utility for {task_id}",
            )
        assert_close(
            partition["utility_margin"],
            margin,
            f"{run_dir}: utility margin for {task_id}",
        )
        routed_method = chosen if use_proof else fallback
        routed_key = (
            str(row["task"]),
            str(row["id"]),
            routed_method,
            str(capacity),
        )
        if routed_key in lookup:
            routed_row = lookup[routed_key]
            if prediction != str(routed_row["prediction"]):
                raise ValueError(
                    f"{run_dir}: routed prediction mismatch for {task_id}"
                )
            for field in (
                "score",
                "actual_retained_tokens",
                "actual_retained_ratio",
                "actual_kv_bytes",
            ):
                assert_close(
                    row[field],
                    routed_row[field],
                    f"{run_dir}: routed {field} for {task_id}",
                )
        elif use_proof:
            proof_routes_without_source_replay += 1
        else:
            raise ValueError(
                f"{run_dir}: physical fallback source row missing"
            )
        proof_routes += int(use_proof)
        fallback_routes += int(not use_proof)
        if chosen is not None:
            selected_candidates[chosen] += 1

    if set(grouped) != set(expected_by_id):
        raise ValueError(f"{run_dir}: sample-id set differs")
    if any(values != configurations for values in grouped.values()):
        raise ValueError(f"{run_dir}: incomplete configuration crossing")
    expected_routes = len(tasks) * n_per_task * len(CAPACITIES)
    if proof_routes + fallback_routes != expected_routes:
        raise ValueError(f"{run_dir}: route count differs")

    recomputed_tasks, recomputed_macro = aggregate(rows, tasks)
    observed_tasks = {
        (
            str(record["task"]),
            str(record["method"]),
            str(record["capacity"]),
        ): record
        for record in metrics["task_records"]
    }
    observed_macro = {
        (str(record["method"]), str(record["capacity"])): record
        for record in metrics["macro_records"]
    }
    if (
        set(recomputed_tasks) != set(observed_tasks)
        or set(recomputed_macro) != set(observed_macro)
    ):
        raise ValueError(f"{run_dir}: aggregate key sets differ")
    for key, expected in recomputed_tasks.items():
        for field, value in expected.items():
            assert_close(
                observed_tasks[key][field],
                value,
                f"{run_dir}: task aggregate {key}/{field}",
            )
    for key, expected in recomputed_macro.items():
        for field, value in expected.items():
            assert_close(
                observed_macro[key][field],
                value,
                f"{run_dir}: macro aggregate {key}/{field}",
            )

    return {
        "run_dir": str(run_dir),
        "model": model,
        "rows": len(rows),
        "selection_start_per_task": start,
        "samples_per_task": n_per_task,
        "configurations_per_sample": len(configurations),
        "proof_routes": proof_routes,
        "physical_fallback_routes": fallback_routes,
        "proof_route_coverage": proof_routes / expected_routes,
        "selected_candidate_counts": dict(selected_candidates),
        "proof_routes_without_source_replay": (
            proof_routes_without_source_replay
        ),
        "max_capacity_excess_tokens": max_excess,
        "raw_empty_predictions": raw_empty,
        "whitespace_only_predictions": whitespace_only,
        "implementation_files_hashed": implementation_count,
        "scorer_checkpoint_files_hashed": checkpoint_count,
    }


def main() -> None:
    args = parse_args()
    selection_path = Path(args.selection_manifest)
    controller_path = Path(args.utility_controller)
    selection = json.loads(selection_path.read_text())
    controller = json.loads(controller_path.read_text())
    if controller.get("status") != "PASS":
        raise ValueError("utility controller did not pass calibration")
    selection_sha256 = sha256_file(selection_path)
    controller_sha256 = sha256_file(controller_path)
    payload = {
        "status": "PASS",
        "protocol": f"untouched_{args.slice}_utility_routing_audit_v1",
        "evaluation_slice": args.slice,
        "selection_manifest_sha256": selection_sha256,
        "utility_controller_sha256": controller_sha256,
        "runs": [
            audit_run(
                Path(run_dir),
                selection,
                selection_sha256,
                controller,
                controller_sha256,
                args.slice,
            )
            for run_dir in args.run_dir
        ],
        "checks": [
            f"untouched frozen {args.slice} slice and exact source answers",
            "complete model/method/capacity crossing",
            "independent QA-F1 and aggregate recomputation",
            "exact capacity, KV-byte, implementation, and checkpoint hashes",
            "all candidate utilities recomputed from serialized coefficients",
            "candidate choice, safety margin, route, and reused output equality",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "runs": len(payload["runs"])}))


if __name__ == "__main__":
    main()
