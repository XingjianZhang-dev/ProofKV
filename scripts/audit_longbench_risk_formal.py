#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

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
BASE_SELECTOR_METHODS = (
    "entity_graph_adaptive",
    "learned_topk_paths",
    "learned_and_or_aux",
)
CAPACITIES = (64, 128, 256, 512)
RISK_METHOD = "risk_calibrated_proof_coverage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independent audit of the untouched risk-calibrated LongBench "
            "evaluation matrix and every routing decision."
        )
    )
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--risk-controller", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def expected_rows(
    selection: dict[str, Any],
    model_label: str,
    start_per_task: int,
    n_per_task: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for task in selection["tasks"]:
        task_selection = selection["selections"][task]
        source_rows = [
            json.loads(line)
            for line in Path(task_selection["data_file"]).read_text().splitlines()
            if line.strip()
        ]
        for selected in task_selection["selected"][
            start_per_task : start_per_task + n_per_task
        ]:
            raw = source_rows[int(selected["index"])]
            key = (str(task), str(selected["id"]))
            if key in expected:
                raise ValueError(f"duplicate frozen sample: {key}")
            expected[key] = {
                "token_counts": selected["token_counts"][model_label],
                "answers": [str(value) for value in raw["answers"]],
            }
    return expected


def risk_feature_value(
    feature: str,
    row: dict[str, Any],
) -> float:
    partition = row["selection_partition"]
    capacity = int(row["capacity"])
    if feature == "proof_core_fraction":
        return float(partition["proof_core_tokens"]) / capacity
    if feature == "coverage_auxiliary_fraction":
        return float(partition["coverage_auxiliary_tokens"]) / capacity
    if feature == "context_log_tokens":
        return math.log1p(int(row["context_tokens"]))
    if feature == "question_budget_fraction":
        return float(row["question_tokens"]) / capacity
    return float(partition[feature])


def rule_uses_proof(value: float, direction: str, threshold: float) -> bool:
    if not math.isfinite(value) or not math.isfinite(threshold):
        raise ValueError("non-finite risk-rule value")
    if direction == "high":
        return value >= threshold
    if direction == "low":
        return value <= threshold
    raise ValueError(f"invalid risk direction: {direction}")


def audit_run(
    run_dir: Path,
    selection: dict[str, Any],
    selection_sha256: str,
    controller: dict[str, Any],
    controller_sha256: str,
) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    rows = load_rows(run_dir / "results.jsonl")
    model_label = str(metrics["model"])
    tasks = list(selection["tasks"])
    proof_method_by_capacity = {
        int(capacity): str(method)
        for capacity, method in controller[
            "proof_method_by_capacity"
        ].items()
    }
    proof_methods = list(
        dict.fromkeys(
            proof_method_by_capacity[capacity]
            for capacity in CAPACITIES
        )
    )
    methods = [
        "full",
        *PHYSICAL_METHODS,
        *BASE_SELECTOR_METHODS,
        *proof_methods,
        RISK_METHOD,
    ]
    start_per_task = int(metrics["selection_start_per_task"])
    n_per_task = int(metrics["n_per_task"])
    contract = controller["formal_evaluation_contract"]
    if start_per_task != int(contract["selection_start_per_task"]):
        raise ValueError(f"{run_dir}: formal selection start differs from controller")
    if n_per_task != int(contract["samples_per_task"]):
        raise ValueError(f"{run_dir}: formal sample count differs from controller")
    if start_per_task + n_per_task != int(selection["max_samples_per_task"]):
        raise ValueError(f"{run_dir}: formal slice is not the frozen untouched suffix")
    if metrics["methods"] != methods or metrics["capacities"] != list(CAPACITIES):
        raise ValueError(f"{run_dir}: method/capacity axes differ")
    if metrics["tasks"] != tasks or int(metrics["truncated_inputs"]) != 0:
        raise ValueError(f"{run_dir}: task or truncation contract differs")
    if (
        metrics["selection_manifest_sha256"] != selection_sha256
        or manifest["selection_manifest_sha256"] != selection_sha256
    ):
        raise ValueError(f"{run_dir}: selection-manifest hash mismatch")
    if manifest.get("risk_controller_sha256") != controller_sha256:
        raise ValueError(f"{run_dir}: risk-controller hash mismatch")
    implementation_hashes = manifest.get("implementation_sha256")
    if not isinstance(implementation_hashes, dict) or not implementation_hashes:
        raise ValueError(f"{run_dir}: implementation hashes missing")
    for path_value, expected_hash in implementation_hashes.items():
        path = Path(path_value)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(
                f"{run_dir}: implementation hash mismatch for {path}"
            )
    checkpoint_hashes = manifest.get("scorer_checkpoint_sha256")
    checkpoint_root = Path(
        manifest["arguments"]["scorer_checkpoint"]
    )
    if not isinstance(checkpoint_hashes, dict) or not checkpoint_hashes:
        raise ValueError(f"{run_dir}: scorer checkpoint hashes missing")
    for relative_path, expected_hash in checkpoint_hashes.items():
        path = checkpoint_root / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(
                f"{run_dir}: scorer checkpoint hash mismatch for {path}"
            )
    if (
        metrics.get("tokenization_contract")
        != "kvpress_split_context_question_token_ids_v1"
    ):
        raise ValueError(f"{run_dir}: split-tokenization contract missing")

    expected_by_id = expected_rows(
        selection,
        model_label,
        start_per_task,
        n_per_task,
    )
    expected_configurations = {("full", "full")} | {
        (method, str(capacity))
        for method in methods
        if method != "full"
        for capacity in CAPACITIES
    }
    expected_count = (
        len(tasks) * n_per_task * len(expected_configurations)
    )
    if len(rows) != expected_count:
        raise ValueError(f"{run_dir}: {len(rows)} rows != {expected_count}")

    lookup = {
        (
            str(row["task"]),
            str(row["id"]),
            str(row["method"]),
            str(row["capacity"]),
        ): row
        for row in rows
    }
    grouped: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    kv_bytes_per_token = int(metrics["kv_bytes_per_token_bf16"])
    max_capacity_excess = 0.0
    proof_routes = 0
    fallback_routes = 0
    raw_empty_predictions = 0
    whitespace_only_predictions = 0
    for row in rows:
        task_id = (str(row["task"]), str(row["id"]))
        if task_id not in expected_by_id:
            raise ValueError(f"{run_dir}: unexpected sample {task_id}")
        grouped[task_id].add((str(row["method"]), str(row["capacity"])))
        expected = expected_by_id[task_id]
        tokens = expected["token_counts"]
        if int(row["prompt_tokens"]) != int(tokens["prompt"]):
            raise ValueError(f"{run_dir}: prompt-token drift for {task_id}")
        if int(row["question_tokens"]) != int(tokens["question"]):
            raise ValueError(f"{run_dir}: question-token drift for {task_id}")
        if [str(value) for value in row["answers"]] != expected["answers"]:
            raise ValueError(f"{run_dir}: answer drift for {task_id}")
        if "prediction" not in row or not isinstance(row["prediction"], str):
            raise ValueError(f"{run_dir}: missing/non-string prediction for {task_id}")
        if row["prediction"] == "":
            raw_empty_predictions += 1
        elif not row["prediction"].strip():
            whitespace_only_predictions += 1
        assert_close(
            row["score"],
            best_qa_f1(str(row["prediction"]), expected["answers"]),
            f"{run_dir}: independently rescored QA-F1 for {task_id}",
        )
        actual_tokens = float(row["actual_retained_tokens"])
        prompt_tokens = int(row["prompt_tokens"])
        assert_close(
            row["actual_retained_ratio"],
            actual_tokens / prompt_tokens,
            f"{run_dir}: retained ratio for {task_id}",
        )
        assert_close(
            row["actual_kv_bytes"],
            actual_tokens * kv_bytes_per_token,
            f"{run_dir}: KV bytes for {task_id}",
        )
        if row["method"] == "full":
            assert_close(
                actual_tokens,
                prompt_tokens,
                f"{run_dir}: full tokens for {task_id}",
            )
        else:
            capacity = int(row["capacity"])
            excess = actual_tokens - min(capacity, prompt_tokens)
            max_capacity_excess = max(max_capacity_excess, excess)
            if excess > 1e-6:
                raise ValueError(f"{run_dir}: capacity exceeded for {task_id}")

        if row["method"] != RISK_METHOD:
            continue
        partition = row.get("selection_partition")
        if not isinstance(partition, dict):
            raise ValueError(f"{run_dir}: risk partition missing for {task_id}")
        rule = controller["final_risk_rule_by_capacity"][
            str(row["capacity"])
        ]
        feature = str(rule["feature"])
        direction = str(rule["direction"])
        threshold = float(rule["threshold"])
        value = risk_feature_value(feature, row)
        expected_proof = rule_uses_proof(value, direction, threshold)
        expected_route = (
            "proof_coverage" if expected_proof else "physical_fallback"
        )
        if partition.get("risk_route") != expected_route:
            raise ValueError(f"{run_dir}: risk decision mismatch for {task_id}")
        assert_close(
            partition["risk_feature_value"],
            value,
            f"{run_dir}: risk feature for {task_id}",
        )
        if (
            partition.get("risk_feature") != feature
            or partition.get("risk_direction") != direction
        ):
            raise ValueError(f"{run_dir}: risk-rule metadata drift for {task_id}")
        assert_close(
            partition["risk_threshold"],
            threshold,
            f"{run_dir}: risk threshold for {task_id}",
        )
        fallback_key = f"{model_label}:{int(row['capacity'])}"
        fallback_method = str(
            controller["fallback_by_model_capacity"][fallback_key]
        )
        proof_method = proof_method_by_capacity[int(row["capacity"])]
        if (
            partition.get("risk_fallback_method") != fallback_method
            or partition.get("risk_proof_method") != proof_method
        ):
            raise ValueError(f"{run_dir}: routed-method metadata drift for {task_id}")
        routed_method = proof_method if expected_proof else fallback_method
        routed_row = lookup[
            (
                str(row["task"]),
                str(row["id"]),
                routed_method,
                str(row["capacity"]),
            )
        ]
        if str(row["prediction"]) != str(routed_row["prediction"]):
            raise ValueError(f"{run_dir}: routed prediction mismatch for {task_id}")
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
        proof_routes += int(expected_proof)
        fallback_routes += int(not expected_proof)

    if set(grouped) != set(expected_by_id):
        raise ValueError(f"{run_dir}: frozen sample-id set differs")
    if any(
        configurations != expected_configurations
        for configurations in grouped.values()
    ):
        raise ValueError(f"{run_dir}: incomplete method/capacity crossing")
    if proof_routes + fallback_routes != len(tasks) * n_per_task * len(CAPACITIES):
        raise ValueError(f"{run_dir}: risk-route count differs")

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
        set(observed_tasks) != set(recomputed_tasks)
        or set(observed_macro) != set(recomputed_macro)
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

    total_routes = proof_routes + fallback_routes
    return {
        "run_dir": str(run_dir),
        "model": model_label,
        "rows": len(rows),
        "selection_start_per_task": start_per_task,
        "samples_per_task": n_per_task,
        "configurations_per_sample": len(expected_configurations),
        "proof_routes": proof_routes,
        "physical_fallback_routes": fallback_routes,
        "proof_route_coverage": proof_routes / total_routes,
        "max_capacity_excess_tokens": max_capacity_excess,
        "raw_empty_predictions": raw_empty_predictions,
        "whitespace_only_predictions": whitespace_only_predictions,
        "implementation_files_hashed": len(implementation_hashes),
        "scorer_checkpoint_files_hashed": len(checkpoint_hashes),
    }


def main() -> None:
    args = parse_args()
    selection_path = Path(args.selection_manifest)
    controller_path = Path(args.risk_controller)
    selection = json.loads(selection_path.read_text())
    controller = json.loads(controller_path.read_text())
    if controller.get("status") != "PASS":
        raise ValueError("risk controller did not pass its calibration gate")
    selection_sha256 = sha256_file(selection_path)
    controller_sha256 = sha256_file(controller_path)
    payload = {
        "status": "PASS",
        "protocol": "untouched_suffix_risk_routing_audit_v1",
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_sha256,
        "risk_controller": str(controller_path),
        "risk_controller_sha256": controller_sha256,
        "runs": [
            audit_run(
                Path(run_dir),
                selection,
                selection_sha256,
                controller,
                controller_sha256,
            )
            for run_dir in args.run_dir
        ],
        "checks": [
            "untouched frozen suffix and exact source answers",
            "complete model/method/capacity crossing",
            "independent official QA-F1 and aggregate recomputation",
            "present string predictions and separately reported degenerate outputs",
            "exact total-prefill capacity and KV-byte accounting",
            "implementation and relation-scorer checkpoint hashes",
            "independent predecode risk-rule recomputation",
            "exact equality with the routed proof or physical prediction",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "runs": len(payload["runs"])}))


if __name__ == "__main__":
    main()
