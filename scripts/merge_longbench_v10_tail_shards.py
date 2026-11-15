#!/usr/bin/env python3
"""Merge scheduling-only LongBench shards with exact duplicate replay checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any


CAPACITIES = (64, 128, 256, 512)
METHODS = (
    "full",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
    "learned_and_or_recent_aux",
    "learned_and_or_stratified_aux",
    "learned_and_or_graph_aux",
    "learned_and_or_mixed_aux",
    "learned_and_or_lexical_anchor25_aux",
    "learned_and_or_lexical_anchor50_aux",
    "learned_and_or_lexical_anchor75_aux",
    "learned_and_or_control_anchor25_aux",
    "learned_and_or_control_anchor50_aux",
    "learned_and_or_control_anchor75_aux",
    "confirmation_stable_proofkv_v10",
)
EXACT_REPLAY_FIELDS = (
    "answers",
    "prediction",
    "score",
    "prompt_tokens",
    "question_tokens",
    "context_tokens",
    "actual_retained_tokens",
    "actual_retained_ratio",
    "actual_kv_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--shard-index", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--original-run-dir", required=True)
    parser.add_argument(
        "--shard-run-dir",
        action="append",
        required=True,
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["task"]),
        str(row["id"]),
        str(row["method"]),
        str(row["capacity"]),
    )


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    return left == right


def validate_duplicate(
    key: tuple[str, str, str, str],
    retained: dict[str, Any],
    replay: dict[str, Any],
) -> None:
    for field in EXACT_REPLAY_FIELDS:
        if not values_equal(retained.get(field), replay.get(field)):
            raise ValueError(
                f"duplicate replay drift for {key} field={field}: "
                f"{retained.get(field)!r} != {replay.get(field)!r}"
            )


def expected_keys(
    selection: dict[str, Any],
) -> tuple[
    set[tuple[str, str, str, str]],
    dict[str, int],
    dict[str, int],
]:
    keys = set()
    task_order = {
        str(task): index for index, task in enumerate(selection["tasks"])
    }
    id_order = {}
    for task in selection["tasks"]:
        task = str(task)
        selected = selection["selections"][task]["selected"]
        for index, record in enumerate(selected):
            source_id = str(record["id"])
            id_order[f"{task}:{source_id}"] = index
            keys.add((task, source_id, "full", "full"))
            for method in METHODS:
                if method == "full":
                    continue
                for capacity in CAPACITIES:
                    keys.add((task, source_id, method, str(capacity)))
    return keys, task_order, id_order


def main() -> None:
    args = parse_args()
    selection_path = Path(args.selection_manifest)
    selection = json.loads(selection_path.read_text())
    selection_sha256 = sha256_file(selection_path)
    shard_index_path = Path(args.shard_index)
    shard_index = json.loads(shard_index_path.read_text())
    if shard_index.get("status") != "FROZEN":
        raise ValueError("tail shard index is not frozen")
    if (
        shard_index.get("parent_selection_manifest_sha256")
        != selection_sha256
    ):
        raise ValueError("tail shard index parent hash drifted")
    if shard_index.get("scientific_selection_unchanged") is not True:
        raise ValueError("tail shard index changes scientific selection")

    original_dir = Path(args.original_run_dir)
    original_results = original_dir / "results.jsonl"
    original_manifest = original_dir / "run_manifest.json"
    if not original_manifest.is_file():
        raise FileNotFoundError(original_manifest)
    original_manifest_payload = json.loads(original_manifest.read_text())
    metrics_path = original_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(
            "original metrics already exist; tail merge is no longer needed"
        )
    out = Path(args.out)
    if out.is_file():
        existing = json.loads(out.read_text())
        if existing.get("status") != "PASS":
            raise ValueError("existing tail merge manifest did not pass")
        if sha256_file(original_results) != existing["merged_results_sha256"]:
            raise ValueError("merged results drifted after tail merge")
        print(json.dumps(existing, sort_keys=True))
        return

    expected, task_order, id_order = expected_keys(selection)
    original_rows = load_jsonl(original_results)
    original_premerge_sha256 = sha256_file(original_results)
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in original_rows:
        key = row_key(row)
        if key not in expected:
            raise ValueError(f"unexpected original row: {key}")
        if key in merged:
            raise ValueError(f"duplicate inside original results: {key}")
        merged[key] = row

    shard_sources = []
    duplicate_rows = 0
    index_tasks = {
        str(record["name"]): [str(task) for task in record["tasks"]]
        for record in shard_index["shards"]
    }
    observed_shard_names = set()
    for value in args.shard_run_dir:
        shard_dir = Path(value)
        shard_name = shard_dir.name
        if shard_name not in index_tasks:
            raise ValueError(f"shard is absent from frozen index: {shard_name}")
        if shard_name in observed_shard_names:
            raise ValueError(f"duplicate shard run directory: {shard_name}")
        observed_shard_names.add(shard_name)
        results_path = shard_dir / "results.jsonl"
        metrics = shard_dir / "metrics.json"
        manifest = shard_dir / "run_manifest.json"
        for path in (results_path, metrics, manifest):
            if not path.is_file():
                raise FileNotFoundError(path)
        metric_payload = json.loads(metrics.read_text())
        manifest_payload = json.loads(manifest.read_text())
        if metric_payload.get("model") != args.model_label:
            raise ValueError(f"shard model drifted: {shard_dir}")
        metric_tasks = [str(task) for task in metric_payload["tasks"]]
        if metric_tasks != index_tasks[shard_name]:
            raise ValueError(
                f"shard tasks drifted for {shard_name}: "
                f"{metric_tasks} != {index_tasks[shard_name]}"
            )
        if (
            metric_payload.get("selection_manifest_sha256")
            != selection_sha256
        ):
            raise ValueError(f"shard selection hash drifted: {shard_dir}")
        if manifest_payload.get("selection_manifest_sha256") != selection_sha256:
            raise ValueError(f"shard run-manifest selection drifted: {shard_dir}")
        if str(manifest_payload["arguments"].get("tasks")) != ",".join(
            index_tasks[shard_name]
        ):
            raise ValueError(f"shard --tasks drifted: {shard_dir}")
        for field in (
            "implementation_sha256",
            "scorer_checkpoint_sha256",
            "risk_controller_sha256",
        ):
            if (
                manifest_payload.get(field)
                != original_manifest_payload.get(field)
            ):
                raise ValueError(
                    f"shard {field} differs from canonical run: {shard_dir}"
                )
        if int(metric_payload.get("n_per_task", -1)) != int(
            selection["max_samples_per_task"]
        ):
            raise ValueError(f"shard sample count drifted: {shard_dir}")
        rows = load_jsonl(results_path)
        expected_shard_rows = (
            len(metric_payload["tasks"])
            * int(selection["max_samples_per_task"])
            * (1 + (len(METHODS) - 1) * len(CAPACITIES))
        )
        if len(rows) != expected_shard_rows:
            raise ValueError(
                f"incomplete shard {shard_dir}: "
                f"{len(rows)} != {expected_shard_rows}"
            )
        for row in rows:
            key = row_key(row)
            if key not in expected:
                raise ValueError(f"unexpected shard row: {key}")
            if key in merged:
                validate_duplicate(key, merged[key], row)
                duplicate_rows += 1
            else:
                merged[key] = row
        shard_sources.append(
            {
                "name": shard_name,
                "run_dir": str(shard_dir.resolve()),
                "tasks": metric_tasks,
                "rows": len(rows),
                "results_sha256": sha256_file(results_path),
                "metrics_sha256": sha256_file(metrics),
                "run_manifest_sha256": sha256_file(manifest),
            }
        )
    if observed_shard_names != set(index_tasks):
        raise ValueError(
            "shard run directories do not cover the frozen index: "
            f"missing={sorted(set(index_tasks) - observed_shard_names)}"
        )

    observed = set(merged)
    if observed != expected:
        raise ValueError(
            "tail merge key coverage mismatch: "
            f"missing={len(expected - observed)} "
            f"extra={len(observed - expected)}"
        )

    method_order = {method: index for index, method in enumerate(METHODS)}
    capacity_order = {
        "full": -1,
        **{str(value): index for index, value in enumerate(CAPACITIES)},
    }
    ordered = sorted(
        merged.values(),
        key=lambda row: (
            task_order[str(row["task"])],
            id_order[f"{row['task']}:{row['id']}"],
            method_order[str(row["method"])],
            capacity_order[str(row["capacity"])],
        ),
    )
    backup_path = original_dir / "results_pre_tail_merge.jsonl"
    if backup_path.exists():
        if sha256_file(backup_path) != original_premerge_sha256:
            raise ValueError("pre-merge backup already exists with another hash")
    else:
        shutil.copy2(original_results, backup_path)
    temporary = original_dir / "results_tail_merge.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, original_results)

    payload = {
        "status": "PASS",
        "protocol": "longbench_v10_exact_tail_shard_merge_v1",
        "model": args.model_label,
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": selection_sha256,
        "shard_index": str(shard_index_path.resolve()),
        "shard_index_sha256": sha256_file(shard_index_path),
        "original_run_dir": str(original_dir.resolve()),
        "original_premerge_rows": len(original_rows),
        "original_premerge_results_sha256": original_premerge_sha256,
        "premerge_backup": str(backup_path.resolve()),
        "premerge_backup_sha256": sha256_file(backup_path),
        "shard_sources": shard_sources,
        "duplicate_rows_exactly_replayed": duplicate_rows,
        "exact_replay_fields": list(EXACT_REPLAY_FIELDS),
        "merged_rows": len(ordered),
        "expected_rows": len(expected),
        "merged_unique_keys": len(merged),
        "merged_results_sha256": sha256_file(original_results),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
