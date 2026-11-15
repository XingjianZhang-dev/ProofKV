#!/usr/bin/env python
"""Create deterministic task-subset manifests from the locked selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.selection_manifest)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    requested = [
        value.strip()
        for value in args.tasks.split(",")
        if value.strip()
    ]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("tasks must be a nonempty unique list")
    unknown = set(requested) - set(source["tasks"])
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    output = dict(source)
    output["tasks"] = requested
    output["selections"] = {
        task: source["selections"][task] for task in requested
    }
    output["shard_provenance"] = {
        "source_manifest": str(source_path),
        "source_protocol": source.get("protocol"),
        "task_subset_only": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": requested, "out": str(out)}))


if __name__ == "__main__":
    main()
