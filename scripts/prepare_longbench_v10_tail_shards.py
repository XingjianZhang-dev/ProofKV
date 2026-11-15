#!/usr/bin/env python3
"""Derive scheduling-only task shards from the frozen LongBench selection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument(
        "--shard",
        action="append",
        required=True,
        help="NAME=task[,task...]",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_shards(values: list[str]) -> dict[str, list[str]]:
    shards: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid shard declaration: {value}")
        name, task_csv = value.split("=", 1)
        tasks = [task.strip() for task in task_csv.split(",") if task.strip()]
        if not name or not tasks:
            raise ValueError(f"invalid shard declaration: {value}")
        if name in shards:
            raise ValueError(f"duplicate shard name: {name}")
        shards[name] = tasks
    return shards


def stable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "protocol",
            "parent_selection_manifest_sha256",
            "scheduling_only",
            "scientific_selection_unchanged",
            "tasks",
            "max_samples_per_task",
            "max_input_tokens",
            "selections",
        )
    }


def main() -> None:
    args = parse_args()
    parent_path = Path(args.selection_manifest)
    parent = json.loads(parent_path.read_text())
    parent_sha256 = sha256_file(parent_path)
    parent_tasks = [str(task) for task in parent["tasks"]]
    shards = parse_shards(args.shard)
    flattened = [task for tasks in shards.values() for task in tasks]
    if len(flattened) != len(set(flattened)):
        raise ValueError("tail shard tasks overlap")
    unknown = sorted(set(flattened) - set(parent_tasks))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, tasks in shards.items():
        payload = {
            key: value
            for key, value in parent.items()
            if key not in {"tasks", "selections"}
        }
        payload.update(
            {
                "protocol": parent["protocol"],
                "scheduling_protocol": (
                    "frozen_selection_task_partition_for_gpu_tail_v1"
                ),
                "scheduling_only": True,
                "scientific_selection_unchanged": True,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "parent_selection_manifest": str(parent_path.resolve()),
                "parent_selection_manifest_sha256": parent_sha256,
                "shard_name": name,
                "tasks": tasks,
                "selections": {
                    task: parent["selections"][task] for task in tasks
                },
            }
        )
        path = out_dir / f"{name}.json"
        if path.is_file():
            existing = json.loads(path.read_text())
            if stable_payload(existing) != stable_payload(payload):
                raise ValueError(f"existing shard drifted: {path}")
        else:
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        records.append(
            {
                "name": name,
                "tasks": tasks,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )

    index_path = out_dir / "index.json"
    index = {
        "status": "FROZEN",
        "protocol": "longbench_v10_tail_shard_index_v1",
        "parent_selection_manifest": str(parent_path.resolve()),
        "parent_selection_manifest_sha256": parent_sha256,
        "scientific_selection_unchanged": True,
        "shards": records,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    if index_path.is_file():
        existing = json.loads(index_path.read_text())
        comparable = {
            key: existing[key]
            for key in index
            if key != "implementation_sha256"
        }
        expected = {
            key: index[key]
            for key in index
            if key != "implementation_sha256"
        }
        if comparable != expected:
            raise ValueError("existing shard index drifted")
        if existing["implementation_sha256"] != index["implementation_sha256"]:
            raise ValueError("tail-shard preparation implementation drifted")
    else:
        index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(index, sort_keys=True))


if __name__ == "__main__":
    main()
