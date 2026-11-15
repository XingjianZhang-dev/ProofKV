#!/usr/bin/env python3
"""Freeze the official-baseline endpoint before any formal model output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from audit_official_longbench_v10_baselines import (
    CAPACITIES,
    CONFIGURATIONS,
    PRIMARY_ENDPOINT,
    PRIMARY_WEIGHTING,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--ruler-audit", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--preflight-audit", required=True)
    parser.add_argument("--preflight-audit-implementation", required=True)
    parser.add_argument("--formal-output-dir", action="append", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    parser.add_argument("--permutation-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def artifact_record(path: Path) -> dict[str, str]:
    require_file(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def stable_contract(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "status",
            "protocol",
            "selection_manifest_sha256",
            "controller_sha256",
            "ruler_audit_sha256",
            "runner_implementation_sha256",
            "audit_implementation_sha256",
            "chain_implementation_sha256",
            "preflight_audit_sha256",
            "preflight_audit_implementation_sha256",
            "models",
            "capacities",
            "bootstrap_replicates",
            "permutation_replicates",
            "seed",
            "precommitted_primary_endpoint",
            "primary_equal_weighting",
            "formal_output_directories",
            "formal_outputs_unobserved_at_freeze",
        )
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    if args.permutation_replicates < 1_000:
        raise ValueError("at least 1000 permutation replicates are required")

    selection_path = Path(args.selection_manifest)
    controller_path = Path(args.controller)
    ruler_audit_path = Path(args.ruler_audit)
    runner_path = Path(args.runner)
    audit_path = Path(args.audit)
    chain_path = Path(args.chain)
    preflight_audit_path = Path(args.preflight_audit)
    preflight_implementation_path = Path(
        args.preflight_audit_implementation
    )
    for path in (
        selection_path,
        controller_path,
        ruler_audit_path,
        runner_path,
        audit_path,
        chain_path,
        preflight_audit_path,
        preflight_implementation_path,
    ):
        require_file(path)

    ruler_audit = json.loads(ruler_audit_path.read_text())
    if ruler_audit.get("status") != "PASS":
        raise ValueError("official RULER audit has not passed")
    preflight_audit = json.loads(preflight_audit_path.read_text())
    if (
        preflight_audit.get("status") != "PASS"
        or preflight_audit.get("protocol")
        != "official_longbench_v10_current_code_smoke_preflight_v1"
        or len(preflight_audit.get("runs", [])) != 2
        or sum(
            int(run.get("rows", 0))
            for run in preflight_audit.get("runs", [])
        )
        != 40
    ):
        raise ValueError("current-code official-baseline preflight did not pass")
    runner_sha256 = sha256_file(runner_path)
    if preflight_audit.get("runner_implementation_sha256") != runner_sha256:
        raise ValueError("preflight runner hash differs from current runner")
    selection = json.loads(selection_path.read_text())
    output_dirs = [Path(value) for value in args.formal_output_dir]
    if len(output_dirs) != 2:
        raise ValueError("exactly two formal output directories are required")

    payload = {
        "status": "FROZEN",
        "protocol": "official_longbench_v10_model_compatible_formal_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256_file(selection_path),
        "controller": str(controller_path.resolve()),
        "controller_sha256": sha256_file(controller_path),
        "ruler_audit": str(ruler_audit_path.resolve()),
        "ruler_audit_sha256": sha256_file(ruler_audit_path),
        "runner": str(runner_path.resolve()),
        "runner_implementation_sha256": runner_sha256,
        "audit": str(audit_path.resolve()),
        "audit_implementation_sha256": sha256_file(audit_path),
        "chain": str(chain_path.resolve()),
        "chain_implementation_sha256": sha256_file(chain_path),
        "preflight_audit": str(preflight_audit_path.resolve()),
        "preflight_audit_sha256": sha256_file(preflight_audit_path),
        "preflight_audit_implementation": str(
            preflight_implementation_path.resolve()
        ),
        "preflight_audit_implementation_sha256": sha256_file(
            preflight_implementation_path
        ),
        "models": {
            model: {
                "official_method": values["method"],
                "official_commit": values["commit"],
            }
            for model, values in CONFIGURATIONS.items()
        },
        "tasks": [str(value) for value in selection["tasks"]],
        "samples_per_task": int(selection["max_samples_per_task"]),
        "capacities": list(CAPACITIES),
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "seed": args.seed,
        "precommitted_primary_endpoint": PRIMARY_ENDPOINT,
        "primary_equal_weighting": PRIMARY_WEIGHTING,
        "capacity_family_correction": "Holm over eight model-capacity tests",
        "formal_output_directories": [
            str(path.resolve()) for path in output_dirs
        ],
        "formal_outputs_unobserved_at_freeze": True,
    }

    out = Path(args.out)
    if out.is_file():
        existing = json.loads(out.read_text())
        if stable_contract(existing) != stable_contract(payload):
            raise ValueError("existing frozen protocol does not match current code")
        print(
            json.dumps(
                {
                    "status": "FROZEN",
                    "idempotent": True,
                    "out": str(out.resolve()),
                },
                sort_keys=True,
            )
        )
        return

    formal_artifacts = ("results.jsonl", "metrics.json", "run_manifest.json")
    observed = [
        str(path / name)
        for path in output_dirs
        for name in formal_artifacts
        if (path / name).exists()
    ]
    if observed:
        raise ValueError(
            "formal outputs already exist before protocol freeze: "
            + ", ".join(observed)
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "idempotent": False,
                "out": str(out.resolve()),
                "sha256": sha256_file(out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
