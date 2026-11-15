#!/usr/bin/env python3
"""Integrity audit and paired inference for official LongBench-v10 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


CAPACITIES = (64, 128, 256, 512)
TARGET_METHOD = "confirmation_stable_proofkv_v10"
PRIMARY_ENDPOINT = (
    "equal-weighted task macro of ProofKV-v10 versus the compatible "
    "official implementation for each model"
)
PRIMARY_WEIGHTING = (
    "average the two models and four capacities within each frozen "
    "task-sample pair, then task macro"
)
CONFIGURATIONS = {
    "qwen25": {
        "method": "official_compresskv",
        "commit": "12f920fbecba67e16806a6d8eac0ad389ddcad6f",
        "packages": {
            "transformers": "4.45.2",
            "tokenizers": "0.20.3",
            "flash-attn": "2.8.3",
        },
        "kv_bytes_per_native_token": 57_344,
    },
    "mistral02": {
        "method": "official_headkv_reasonkv",
        "commit": "0862a0955fe82e9ff611d59541918e02c5def625",
        "packages": {
            "transformers": "4.37.2",
            "tokenizers": "0.15.2",
            "flash-attn": "2.8.3",
            "nvtx": "0.2.15",
            "protobuf": "5.27.3",
            "sentencepiece": "0.2.0",
        },
        "kv_bytes_per_native_token": 131_072,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--formal-protocol", required=True)
    parser.add_argument("--qwen-run-dir", required=True)
    parser.add_argument("--mistral-run-dir", required=True)
    parser.add_argument("--qwen-v10-run-dir", required=True)
    parser.add_argument("--mistral-v10-run-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    parser.add_argument("--permutation-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capacity_vector_sha256(values: list[int]) -> str:
    canonical = [int(value) for value in values]
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode()
    ).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{path}: empty JSONL")
    return rows


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
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
    return max(qa_f1(prediction, answer) for answer in answers)


def validate_source_hashes(
    official_repo: Path,
    method: str,
    recorded: dict[str, str],
) -> None:
    root = (
        official_repo / "methods" / "compresskv"
        if method == "official_compresskv"
        else official_repo / "headkv"
    )
    observed = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
    }
    if observed != recorded:
        raise ValueError(f"{method}: official source hashes drifted")


def validate_auxiliary_hashes(
    official_repo: Path,
    recorded: dict[str, str],
) -> None:
    observed = {}
    for relative, expected_hash in recorded.items():
        path = official_repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = sha256_file(path)
        if observed[relative] != expected_hash:
            raise ValueError(f"official auxiliary artifact drifted: {relative}")


def audit_run(
    model: str,
    run_dir: Path,
    selection: dict[str, Any],
    selection_sha256: str,
    expected_runner_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    configuration = CONFIGURATIONS[model]
    method = configuration["method"]
    results_path = run_dir / "results.jsonl"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "run_manifest.json"
    for path in (results_path, metrics_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = load_jsonl(results_path)
    metrics = json.loads(metrics_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    tasks = [str(value) for value in selection["tasks"]]
    n_per_task = int(selection["max_samples_per_task"])
    expected_ids = {
        task: {
            str(value["id"])
            for value in selection["selections"][task]["selected"]
        }
        for task in tasks
    }
    expected_keys = {
        (task, source_id, capacity)
        for task in tasks
        for source_id in expected_ids[task]
        for capacity in CAPACITIES
    }
    observed_keys: set[tuple[str, str, int]] = set()
    max_excess = -math.inf
    runtime_capacity_hashes: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"{model} row {index}"
        if row.get("method") != method:
            raise ValueError(f"{prefix}: wrong method")
        task = str(row["task"])
        source_id = str(row["id"])
        capacity = int(row["capacity"])
        key = (task, source_id, capacity)
        if key in observed_keys:
            raise ValueError(f"{prefix}: duplicate key {key}")
        observed_keys.add(key)
        if key not in expected_keys:
            raise ValueError(f"{prefix}: unexpected key {key}")
        score = float(row["score"])
        prediction = row.get("prediction")
        answers = row.get("answers")
        if not isinstance(prediction, str):
            raise ValueError(f"{prefix}: prediction is not a string")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"{prefix}: answers are missing")
        replayed = best_qa_f1(
            prediction,
            [str(answer) for answer in answers],
        )
        if not math.isclose(score, replayed, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{prefix}: QA-F1 replay differs {score} != {replayed}"
            )
        seconds = float(row["seconds"])
        prompt_tokens = int(row["prompt_tokens"])
        retained = float(row["actual_retained_tokens"])
        ratio = float(row["actual_retained_ratio"])
        kv_bytes = float(row["actual_kv_bytes"])
        values = (score, seconds, retained, ratio, kv_bytes)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{prefix}: non-finite measurement")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{prefix}: score outside [0,1]")
        if seconds < 0 or retained <= 0 or kv_bytes <= 0:
            raise ValueError(f"{prefix}: invalid resource measurement")
        if prompt_tokens <= capacity:
            raise ValueError(f"{prefix}: prompt does not exercise compression")
        if not math.isclose(
            ratio,
            retained / prompt_tokens,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{prefix}: retained ratio is inconsistent")
        expected_kv_bytes = (
            retained * configuration["kv_bytes_per_native_token"]
        )
        if not math.isclose(
            kv_bytes,
            expected_kv_bytes,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{prefix}: physical KV bytes differ "
                f"{kv_bytes} != {expected_kv_bytes}"
            )
        if not math.isclose(
            float(row["generation_seconds"]),
            seconds,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ) or not math.isclose(
            float(row["selection_seconds"]),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{prefix}: timing decomposition differs")
        if method == "official_compresskv":
            expected_capacities = metrics["capacity_configurations"][
                str(capacity)
            ]["layer_capacities"]
            expected_runtime_hash = capacity_vector_sha256(
                expected_capacities
            )
            if (
                int(row.get("runtime_capacity_cluster_count", -1))
                != len(expected_capacities)
                or row.get("runtime_layer_capacities_sha256")
                != expected_runtime_hash
            ):
                raise ValueError(
                    f"{prefix}: live CompressKV capacity state differs"
                )
            runtime_capacity_hashes.add(expected_runtime_hash)
        max_excess = max(max_excess, retained - capacity)
    if observed_keys != expected_keys:
        missing = expected_keys - observed_keys
        extra = observed_keys - expected_keys
        raise ValueError(
            f"{model}: key coverage mismatch missing={len(missing)} "
            f"extra={len(extra)}"
        )
    expected_rows = len(tasks) * n_per_task * len(CAPACITIES)
    if len(rows) != expected_rows:
        raise ValueError(
            f"{model}: expected {expected_rows} rows, found {len(rows)}"
        )
    if max_excess > 1e-6:
        raise ValueError(f"{model}: physical capacity exceeded by {max_excess}")
    if method == "official_compresskv" and len(runtime_capacity_hashes) != len(
        CAPACITIES
    ):
        raise ValueError(
            f"{model}: runtime capacities are not distinct across the grid"
        )
    if metrics.get("status") != "PASS":
        raise ValueError(f"{model}: metrics did not pass")
    metric_contract = {
        "method": method,
        "model": model,
        "rows": expected_rows,
        "n_per_task": n_per_task,
        "selection_manifest_sha256": selection_sha256,
        "official_commit": configuration["commit"],
        "results_sha256": sha256_file(results_path),
    }
    for field, expected in metric_contract.items():
        if metrics.get(field) != expected:
            raise ValueError(
                f"{model}: metrics {field}={metrics.get(field)!r}, "
                f"expected {expected!r}"
            )
    if tuple(int(value) for value in metrics["capacities"]) != CAPACITIES:
        raise ValueError(f"{model}: metrics capacities drifted")
    arguments = manifest["arguments"]
    if arguments["method"] != method or arguments["model_label"] != model:
        raise ValueError(f"{model}: manifest model/method mismatch")
    if manifest["selection_manifest_sha256"] != selection_sha256:
        raise ValueError(f"{model}: manifest selection hash mismatch")
    if manifest["official_commit"] != configuration["commit"]:
        raise ValueError(f"{model}: manifest official commit mismatch")
    runner_hashes = [
        value
        for path, value in manifest["implementation_sha256"].items()
        if str(path).endswith("/run_official_longbench_v10_baseline.py")
    ]
    if runner_hashes != [expected_runner_sha256]:
        raise ValueError(
            f"{model}: runtime runner hash differs from frozen protocol"
        )
    packages = manifest["environment"]["packages"]
    for package, expected in configuration["packages"].items():
        if packages.get(package) != expected:
            raise ValueError(
                f"{model}: {package}={packages.get(package)!r}, "
                f"expected {expected!r}"
            )
    validate_source_hashes(
        Path(arguments["official_repo"]),
        method,
        manifest["official_source_sha256"],
    )
    validate_auxiliary_hashes(
        Path(arguments["official_repo"]),
        manifest["official_auxiliary_sha256"],
    )
    return {
        "status": "PASS",
        "model": model,
        "method": method,
        "rows": len(rows),
        "expected_rows": expected_rows,
        "unique_keys": len(observed_keys),
        "runtime_capacity_hashes": sorted(runtime_capacity_hashes),
        "max_capacity_excess_tokens": max_excess,
        "results_sha256": sha256_file(results_path),
        "metrics_sha256": sha256_file(metrics_path),
        "manifest_sha256": sha256_file(manifest_path),
        "environment": manifest["environment"],
        "official_commit": configuration["commit"],
    }, rows


def task_macro_score(
    rows: list[dict[str, Any]],
    tasks: list[str],
    method: str,
    capacity: int,
) -> float:
    values = []
    for task in tasks:
        task_values = [
            100 * float(row["score"])
            for row in rows
            if row["task"] == task
            and row["method"] == method
            and int(row["capacity"]) == capacity
        ]
        if not task_values:
            raise ValueError(f"missing {method}:{task}:{capacity}")
        values.append(float(np.mean(task_values)))
    return float(np.mean(values))


def paired_delta_matrix(
    official_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    tasks: list[str],
    official_method: str,
    capacity: int,
) -> np.ndarray:
    official = {
        (str(row["task"]), str(row["id"])): 100 * float(row["score"])
        for row in official_rows
        if row["method"] == official_method
        and int(row["capacity"]) == capacity
    }
    target = {
        (str(row["task"]), str(row["id"])): 100 * float(row["score"])
        for row in target_rows
        if row["method"] == TARGET_METHOD
        and int(row["capacity"]) == capacity
    }
    values = []
    for task in tasks:
        ids = sorted(
            source_id
            for candidate_task, source_id in official
            if candidate_task == task
        )
        if not ids:
            raise ValueError(f"no official rows for {task}:{capacity}")
        task_values = []
        for source_id in ids:
            key = (task, source_id)
            if key not in target:
                raise ValueError(f"missing target row {key}:{capacity}")
            task_values.append(target[key] - official[key])
        values.append(task_values)
    widths = {len(value) for value in values}
    if len(widths) != 1:
        raise ValueError("task sample widths differ")
    return np.asarray(values, dtype=np.float64)


def paired_inference(
    matrix: np.ndarray,
    bootstrap_replicates: int,
    permutation_replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    observed = float(matrix.mean())
    task_count, sample_count = matrix.shape
    bootstrap = np.zeros(bootstrap_replicates, dtype=np.float64)
    for task_index in range(task_count):
        indices = rng.integers(
            0,
            sample_count,
            size=(bootstrap_replicates, sample_count),
        )
        bootstrap += matrix[task_index][indices].mean(axis=1)
    bootstrap /= task_count
    flattened = matrix.reshape(-1)
    signs = rng.integers(
        0,
        2,
        size=(permutation_replicates, flattened.size),
        dtype=np.int8,
    )
    signs = signs * 2 - 1
    null = (signs * flattened).mean(axis=1)
    p_value = (
        1
        + int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12))
    ) / (permutation_replicates + 1)
    return {
        "delta_score_points": observed,
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "permutation_p_two_sided": p_value,
    }


def combine_paired_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    """Equal-weight model/capacity blocks within each task-sample pair."""

    if not matrices:
        raise ValueError("no paired matrices for the combined endpoint")
    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) != 1:
        raise ValueError(f"paired matrix shapes differ: {sorted(shapes)}")
    return np.stack(matrices, axis=0).mean(axis=0)


def holm_adjust(records: list[dict[str, Any]]) -> None:
    ordered = sorted(
        enumerate(records),
        key=lambda item: item[1]["permutation_p_two_sided"],
    )
    running = 0.0
    count = len(records)
    for rank, (index, record) in enumerate(ordered):
        adjusted = min(
            1.0,
            (count - rank) * record["permutation_p_two_sided"],
        )
        running = max(running, adjusted)
        records[index]["holm_p"] = running


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    if args.permutation_replicates < 1000:
        raise ValueError("at least 1000 permutation replicates are required")
    selection_path = Path(args.selection_manifest)
    selection = json.loads(selection_path.read_text())
    selection_sha256 = sha256_file(selection_path)
    protocol_path = Path(args.formal_protocol)
    protocol = json.loads(protocol_path.read_text())
    protocol_contract = {
        "status": "FROZEN",
        "protocol": "official_longbench_v10_model_compatible_formal_v1",
        "selection_manifest_sha256": selection_sha256,
        "audit_implementation_sha256": sha256_file(Path(__file__).resolve()),
        "precommitted_primary_endpoint": PRIMARY_ENDPOINT,
        "primary_equal_weighting": PRIMARY_WEIGHTING,
        "formal_outputs_unobserved_at_freeze": True,
    }
    for field, expected in protocol_contract.items():
        if protocol.get(field) != expected:
            raise ValueError(
                f"formal protocol {field}={protocol.get(field)!r}, "
                f"expected {expected!r}"
            )
    for path_field, hash_field in (
        ("controller", "controller_sha256"),
        ("ruler_audit", "ruler_audit_sha256"),
        ("runner", "runner_implementation_sha256"),
        ("chain", "chain_implementation_sha256"),
        ("preflight_audit", "preflight_audit_sha256"),
        (
            "preflight_audit_implementation",
            "preflight_audit_implementation_sha256",
        ),
    ):
        artifact_path = Path(protocol[path_field])
        require_hash = protocol[hash_field]
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        observed_hash = sha256_file(artifact_path)
        if observed_hash != require_hash:
            raise ValueError(
                f"formal protocol {path_field} drifted: "
                f"{observed_hash} != {require_hash}"
            )
    preflight = json.loads(Path(protocol["preflight_audit"]).read_text())
    if (
        preflight.get("status") != "PASS"
        or preflight.get("protocol")
        != "official_longbench_v10_current_code_smoke_preflight_v1"
        or preflight.get("runner_implementation_sha256")
        != protocol["runner_implementation_sha256"]
        or len(preflight.get("runs", [])) != 2
        or sum(
            int(run.get("rows", 0))
            for run in preflight.get("runs", [])
        )
        != 40
    ):
        raise ValueError("frozen current-code preflight contract differs")
    expected_models = {
        model: {
            "official_method": values["method"],
            "official_commit": values["commit"],
        }
        for model, values in CONFIGURATIONS.items()
    }
    protocol_values = {
        "models": expected_models,
        "capacities": list(CAPACITIES),
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "seed": args.seed,
    }
    for field, expected in protocol_values.items():
        if protocol.get(field) != expected:
            raise ValueError(
                f"formal protocol {field}={protocol.get(field)!r}, "
                f"expected {expected!r}"
            )
    tasks = [str(value) for value in selection["tasks"]]
    run_dirs = {
        "qwen25": Path(args.qwen_run_dir),
        "mistral02": Path(args.mistral_run_dir),
    }
    protocol_output_dirs = {
        str(Path(value).resolve())
        for value in protocol["formal_output_directories"]
    }
    requested_output_dirs = {
        str(value.resolve()) for value in run_dirs.values()
    }
    if protocol_output_dirs != requested_output_dirs:
        raise ValueError("formal output directories differ from frozen protocol")
    target_dirs = {
        "qwen25": Path(args.qwen_v10_run_dir),
        "mistral02": Path(args.mistral_v10_run_dir),
    }
    rng = np.random.default_rng(args.seed)
    audits = []
    comparisons = []
    primary_blocks = []
    source_hashes = {"selection_manifest": selection_sha256}
    for model in ("qwen25", "mistral02"):
        audit, official_rows = audit_run(
            model,
            run_dirs[model],
            selection,
            selection_sha256,
            str(protocol["runner_implementation_sha256"]),
        )
        audits.append(audit)
        target_path = target_dirs[model] / "results.jsonl"
        target_rows = load_jsonl(target_path)
        source_hashes[f"{model}_target_results"] = sha256_file(target_path)
        source_hashes[f"{model}_official_results"] = audit["results_sha256"]
        method = CONFIGURATIONS[model]["method"]
        for capacity in CAPACITIES:
            matrix = paired_delta_matrix(
                official_rows,
                target_rows,
                tasks,
                method,
                capacity,
            )
            primary_blocks.append(matrix)
            record = {
                "model": model,
                "capacity": capacity,
                "n": int(matrix.size),
                "target_method": TARGET_METHOD,
                "official_method": method,
                "target_score": task_macro_score(
                    target_rows,
                    tasks,
                    TARGET_METHOD,
                    capacity,
                ),
                "official_score": task_macro_score(
                    official_rows,
                    tasks,
                    method,
                    capacity,
                ),
                **paired_inference(
                    matrix,
                    args.bootstrap_replicates,
                    args.permutation_replicates,
                    rng,
                ),
            }
            comparisons.append(record)
    holm_adjust(comparisons)
    primary_matrix = combine_paired_matrices(primary_blocks)
    primary = paired_inference(
        primary_matrix,
        args.bootstrap_replicates,
        args.permutation_replicates,
        np.random.default_rng(args.seed + 10_000),
    )
    primary.update(
        {
            "comparison": (
                "proofkv_v10_vs_model_compatible_official_baseline"
            ),
            "equal_weighting": PRIMARY_WEIGHTING,
            "model_blocks": 2,
            "capacity_blocks": len(CAPACITIES),
            "task_groups": int(primary_matrix.shape[0]),
            "samples_per_task": int(primary_matrix.shape[1]),
        }
    )
    payload = {
        "status": "PASS",
        "protocol": (
            "frozen_model_compatible_official_longbench_v10_audit_and_"
            "paired_inference"
        ),
        "tasks": tasks,
        "capacities": list(CAPACITIES),
        "n_per_task": int(selection["max_samples_per_task"]),
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "seed": args.seed,
        "audit_implementation_sha256": sha256_file(Path(__file__).resolve()),
        "formal_protocol": str(protocol_path.resolve()),
        "formal_protocol_sha256": sha256_file(protocol_path),
        "precommitted_primary_endpoint": PRIMARY_ENDPOINT,
        "primary_inference": primary,
        "run_audits": audits,
        "comparisons": comparisons,
        "source_sha256": source_hashes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "runs": len(audits),
                "comparisons": len(comparisons),
                "primary_delta": primary["delta_score_points"],
                "primary_p": primary["permutation_p_two_sided"],
                "out": str(out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
