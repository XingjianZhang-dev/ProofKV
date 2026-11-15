#!/usr/bin/env python
"""Freeze the official-RULER v10 formal protocol before any formal output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_official_ruler_upgraded_v10 import implementation_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument(
        "--formal-output-dir",
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


def main() -> None:
    args = parse_args()
    development_path = Path(args.development_manifest)
    formal_path = Path(args.formal_manifest)
    controller_path = Path(args.controller)
    development = json.loads(development_path.read_text())
    formal = json.loads(formal_path.read_text())
    controller = json.loads(controller_path.read_text())
    if (
        development["tasks"] != formal["tasks"]
        or formal["tasks"] != controller["tasks"]
        or development["capacities"] != formal["capacities"]
        or formal["capacities"] != controller["capacities"]
    ):
        raise ValueError("official-RULER protocol axes differ")
    formal_output_directories = [
        str(Path(value).resolve())
        for value in args.formal_output_dir
    ]
    for value in formal_output_directories:
        results = Path(value) / "results.jsonl"
        if results.exists() and any(
            line.strip() for line in results.read_text().splitlines()
        ):
            raise ValueError(
                f"formal output exists before protocol freeze: {results}"
            )
    repo_root = Path(__file__).resolve().parents[1]
    analysis_paths = [
        Path(__file__).resolve(),
        repo_root / "scripts" / "prepare_official_ruler_v10_split.py",
        repo_root / "scripts" / "freeze_official_ruler_v10_controller.py",
        repo_root / "scripts" / "audit_official_ruler_v10.py",
        repo_root / "scripts" / "analyze_official_ruler_v10_statistics.py",
        repo_root / "scripts" / "analyze_longbench_v10_efficiency.py",
    ]
    implementation = implementation_sha256()
    implementation.update(
        {
            str(path): sha256_file(path)
            for path in analysis_paths
        }
    )
    samples_per_task = int(formal["max_samples_per_task"])
    capacities = [int(value) for value in formal["capacities"]]
    payload = {
        "status": "FROZEN",
        "protocol": "proofkv_v10_official_ruler_formal_protocol_v1",
        "frozen_before_any_formal_output": True,
        "source_repo": formal["source_repo"],
        "source_commit": formal["source_commit"],
        "tasks": formal["tasks"],
        "capacities": capacities,
        "models": controller["model_labels"],
        "samples_per_task": samples_per_task,
        "development_manifest": str(development_path.resolve()),
        "development_manifest_sha256": sha256_file(development_path),
        "formal_manifest": str(formal_path.resolve()),
        "formal_manifest_sha256": sha256_file(formal_path),
        "controller": str(controller_path.resolve()),
        "controller_sha256": sha256_file(controller_path),
        "formal_output_directories": formal_output_directories,
        "primary_methods": [
            "full",
            "streamingllm",
            "snapkv",
            "pyramidkv",
            "chunkkv_snapkv",
            "confirmation_stable_proofkv_v10",
        ],
        "rows_per_model": len(formal["tasks"])
        * samples_per_task
        * (1 + 5 * len(capacities)),
        "inference": {
            "bootstrap_replicates": 50000,
            "permutation_replicates": 50000,
            "capacity_family_correction": "Holm",
            "seed": 20260726,
        },
        "implementation_sha256": dict(sorted(implementation.items())),
    }
    out = Path(args.out)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out.exists():
        if out.read_text() != serialized:
            raise ValueError("existing official-RULER formal protocol differs")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(serialized)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "protocol": str(out.resolve()),
                "protocol_sha256": sha256_file(out),
                "rows_per_model": payload["rows_per_model"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

