#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METHODS = [
    "full",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
    "entity_graph_adaptive",
    "learned_topk_paths",
    "learned_and_or_aux",
]
CAPACITIES = [64, 128, 256, 512]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent contract and score audit for common-interface LongBench runs."
    )
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction.lstrip()).split()
    gold_tokens = normalize_answer(answer).split()
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_qa_f1(prediction: str, answers: list[str]) -> float:
    return max((qa_f1(prediction, answer) for answer in answers), default=0.0)


def assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{label}: {observed} != {expected}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(str(row["task"]), str(row["id"]), str(row["method"]), str(row["capacity"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate task/id/method/capacity keys in {path}")
    return rows


def expected_rows(
    selection: dict[str, Any],
    model_label: str,
    n_per_task: int,
    start_per_task: int = 0,
) -> dict[tuple[str, str], dict[str, Any]]:
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for task in selection["tasks"]:
        for selected in selection["selections"][task]["selected"][
            start_per_task : start_per_task + n_per_task
        ]:
            key = (task, str(selected["id"]))
            if key in expected:
                raise ValueError(f"duplicate selected id in manifest: {key}")
            expected[key] = selected["token_counts"][model_label]
    return expected


def aggregate(
    rows: list[dict[str, Any]], tasks: list[str]
) -> tuple[dict[tuple[str, str, str], dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["method"]), str(row["capacity"]))].append(row)
    task_records: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, values in grouped.items():
        task_records[key] = {
            "n": float(len(values)),
            "score": 100 * sum(float(value["score"]) for value in values) / len(values),
            "actual_retained_ratio": sum(float(value["actual_retained_ratio"]) for value in values)
            / len(values),
            "actual_kv_bytes": sum(float(value["actual_kv_bytes"]) for value in values) / len(values),
            "seconds": sum(float(value["seconds"]) for value in values) / len(values),
            "peak_cuda_memory_gb": max(float(value["peak_cuda_memory_gb"]) for value in values),
        }
    macro_records: dict[tuple[str, str], dict[str, float]] = {}
    for method, capacity in sorted({(key[1], key[2]) for key in task_records}):
        records = [task_records[(task, method, capacity)] for task in tasks]
        macro_records[(method, capacity)] = {
            "tasks": float(len(records)),
            "n": sum(record["n"] for record in records),
            "macro_score": sum(record["score"] for record in records) / len(records),
            "actual_retained_ratio": sum(record["actual_retained_ratio"] for record in records)
            / len(records),
            "seconds": sum(record["seconds"] for record in records) / len(records),
        }
    return task_records, macro_records


def audit_run(
    run_dir: Path,
    selection: dict[str, Any],
    selection_sha256: str,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        pilot_metrics = sorted(run_dir.glob("pilot_metrics_n*.json"))
        if len(pilot_metrics) != 1:
            raise ValueError(
                f"{run_dir}: expected metrics.json or exactly one pilot_metrics_n*.json"
            )
        metrics_path = pilot_metrics[0]
    metrics = json.loads(metrics_path.read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    rows = load_rows(run_dir / "results.jsonl")
    model_label = str(metrics["model"])
    tasks = list(selection["tasks"])
    n_per_task = int(metrics["n_per_task"])
    start_per_task = int(metrics.get("selection_start_per_task", 0))
    if (
        start_per_task < 0
        or n_per_task < 1
        or start_per_task + n_per_task
        > int(selection["max_samples_per_task"])
    ):
        raise ValueError(f"{run_dir}: invalid n_per_task {n_per_task}")
    expected_by_id = expected_rows(
        selection,
        model_label,
        n_per_task,
        start_per_task=start_per_task,
    )
    expected_configurations = {("full", "full")} | {
        (method, str(capacity))
        for method in METHODS
        if method != "full"
        for capacity in CAPACITIES
    }
    expected_count = len(tasks) * n_per_task * len(expected_configurations)

    if len(rows) != expected_count:
        raise ValueError(f"{run_dir}: {len(rows)} rows != {expected_count}")
    if metrics["tasks"] != tasks or metrics["methods"] != METHODS or metrics["capacities"] != CAPACITIES:
        raise ValueError(f"{run_dir}: metrics protocol axes differ from the frozen selection")
    if metrics["truncated_inputs"] != 0:
        raise ValueError(f"{run_dir}: nonzero truncated_inputs")
    if (
        metrics_path.name == "metrics.json"
        and metrics.get("tokenization_contract")
        != "kvpress_split_context_question_token_ids_v1"
    ):
        raise ValueError(f"{run_dir}: missing split-tokenization contract")
    if metrics["selection_manifest_sha256"] != selection_sha256:
        raise ValueError(f"{run_dir}: metrics selection hash mismatch")
    if manifest["selection_manifest_sha256"] != selection_sha256:
        raise ValueError(f"{run_dir}: run-manifest selection hash mismatch")

    grouped_configs: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    raw_empty_predictions = 0
    whitespace_only_predictions = 0
    max_capacity_excess = 0.0
    kv_bytes_per_token = int(metrics["kv_bytes_per_token_bf16"])
    for row in rows:
        task_id = (str(row["task"]), str(row["id"]))
        if task_id not in expected_by_id:
            raise ValueError(f"{run_dir}: unexpected sample {task_id}")
        grouped_configs[task_id].add((str(row["method"]), str(row["capacity"])))
        expected_tokens = expected_by_id[task_id]
        if int(row["prompt_tokens"]) != int(expected_tokens["prompt"]):
            raise ValueError(f"{run_dir}: prompt-token drift for {task_id}")
        if int(row["question_tokens"]) != int(expected_tokens["question"]):
            raise ValueError(f"{run_dir}: question-token drift for {task_id}")
        if int(row["context_tokens"]) + int(row["question_tokens"]) != int(row["prompt_tokens"]):
            raise ValueError(f"{run_dir}: context/question token sum mismatch for {task_id}")
        if int(row["prompt_tokens"]) > int(selection["max_input_tokens"]):
            raise ValueError(f"{run_dir}: input exceeds frozen maximum for {task_id}")
        if "prediction" not in row or not isinstance(row["prediction"], str):
            raise ValueError(f"{run_dir}: missing/non-string prediction for {task_id}")
        if row["prediction"] == "":
            raw_empty_predictions += 1
        elif not row["prediction"].strip():
            whitespace_only_predictions += 1
        recomputed_score = best_qa_f1(str(row["prediction"]), [str(answer) for answer in row["answers"]])
        assert_close(row["score"], recomputed_score, f"{run_dir}: score for {task_id}")
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
                raise ValueError(f"{run_dir}: non-finite {field} for {task_id}")
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
            assert_close(actual_tokens, prompt_tokens, f"{run_dir}: Full tokens for {task_id}")
        else:
            capacity = int(row["capacity"])
            excess = actual_tokens - min(capacity, prompt_tokens)
            max_capacity_excess = max(max_capacity_excess, excess)
            if excess > 1e-6:
                raise ValueError(f"{run_dir}: capacity exceeded for {task_id}/{row['method']}/{capacity}")

    if set(grouped_configs) != set(expected_by_id):
        raise ValueError(f"{run_dir}: sample-id set differs from frozen selection")
    for task_id, configurations in grouped_configs.items():
        if configurations != expected_configurations:
            raise ValueError(f"{run_dir}: incomplete configuration crossing for {task_id}")

    recomputed_tasks, recomputed_macro = aggregate(rows, tasks)
    observed_tasks = {
        (str(record["task"]), str(record["method"]), str(record["capacity"])): record
        for record in metrics["task_records"]
    }
    observed_macro = {
        (str(record["method"]), str(record["capacity"])): record
        for record in metrics["macro_records"]
    }
    if set(observed_tasks) != set(recomputed_tasks) or set(observed_macro) != set(recomputed_macro):
        raise ValueError(f"{run_dir}: aggregate key sets differ")
    for key, expected in recomputed_tasks.items():
        for field, value in expected.items():
            assert_close(observed_tasks[key][field], value, f"{run_dir}: task aggregate {key}/{field}")
    for key, expected in recomputed_macro.items():
        for field, value in expected.items():
            assert_close(observed_macro[key][field], value, f"{run_dir}: macro aggregate {key}/{field}")

    return {
        "run_dir": str(run_dir),
        "model": model_label,
        "rows": len(rows),
        "tasks": len(tasks),
        "samples_per_task": n_per_task,
        "selection_start_per_task": start_per_task,
        "configurations_per_sample": len(expected_configurations),
        "truncated_inputs": 0,
        "raw_empty_predictions": raw_empty_predictions,
        "whitespace_only_predictions": whitespace_only_predictions,
        "max_capacity_excess_tokens": max_capacity_excess,
        "selection_manifest_sha256": selection_sha256,
    }


def main() -> None:
    args = parse_args()
    selection_path = Path(args.selection_manifest)
    selection = json.loads(selection_path.read_text())
    selection_sha256 = sha256_file(selection_path)
    payload = {
        "status": "PASS",
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_sha256,
        "runs": [
            audit_run(Path(run_dir), selection, selection_sha256)
            for run_dir in args.run_dir
        ],
        "checks": [
            "frozen selection and model-specific token counts",
            "complete task/sample/method/capacity crossing",
            "unique result keys",
            "independent row-level QA-F1 recomputation",
            "independent task and macro aggregate recomputation",
            "zero truncation, present string predictions, and separately reported degenerate outputs",
            "exact KV-byte accounting and capacity upper bounds",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "runs": len(payload["runs"])}))


if __name__ == "__main__":
    main()
