#!/usr/bin/env python
"""Verify that official-RULER runs used the pre-output frozen implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-protocol", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(
    protocol_path: Path,
    run_dirs: list[Path],
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol.get("status") != "FROZEN"
        or protocol.get("protocol")
        != "proofkv_v10_official_ruler_formal_protocol_v1"
        or protocol.get("frozen_before_any_formal_output") is not True
    ):
        raise ValueError("official-RULER formal protocol is not frozen")
    frozen_implementation = protocol.get("implementation_sha256")
    if not isinstance(frozen_implementation, dict) or not frozen_implementation:
        raise ValueError("formal protocol has no frozen implementation hashes")

    drifted_files = []
    missing_files = []
    for value, expected_hash in sorted(frozen_implementation.items()):
        path = Path(value)
        if not path.is_file():
            missing_files.append(value)
        elif sha256_file(path) != expected_hash:
            drifted_files.append(value)
    if missing_files or drifted_files:
        raise ValueError(
            "frozen implementation drift: "
            f"missing={missing_files}, drifted={drifted_files}"
        )

    expected_dirs = {
        str(Path(value).resolve())
        for value in protocol["formal_output_directories"]
    }
    observed_dirs = {str(path.resolve()) for path in run_dirs}
    if observed_dirs != expected_dirs:
        raise ValueError("formal run directories differ from frozen protocol")

    manifests = []
    runtime_maps = []
    for run_dir in sorted(run_dirs, key=lambda path: str(path.resolve())):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("selection_manifest_sha256")
            != protocol["formal_manifest_sha256"]
            or manifest.get("risk_controller_sha256")
            != protocol["controller_sha256"]
        ):
            raise ValueError(f"selection/controller binding differs: {run_dir}")
        runtime = manifest.get("implementation_sha256")
        if not isinstance(runtime, dict) or not runtime:
            raise ValueError(f"run manifest has no implementation map: {run_dir}")
        extra = sorted(set(runtime) - set(frozen_implementation))
        mismatched = sorted(
            path
            for path, digest in runtime.items()
            if frozen_implementation.get(path) != digest
        )
        if extra or mismatched:
            raise ValueError(
                f"runtime implementation differs for {run_dir}: "
                f"extra={extra}, mismatched={mismatched}"
            )
        runtime_maps.append(runtime)
        manifests.append(
            {
                "run_dir": str(run_dir.resolve()),
                "run_manifest_sha256": sha256_file(manifest_path),
                "runtime_implementation_files": len(runtime),
            }
        )
    if any(runtime != runtime_maps[0] for runtime in runtime_maps[1:]):
        raise ValueError("model runs used different runtime implementations")

    return {
        "status": "PASS",
        "protocol": "proofkv_v10_official_ruler_protocol_binding_audit_v1",
        "formal_protocol": str(protocol_path.resolve()),
        "formal_protocol_sha256": sha256_file(protocol_path),
        "frozen_implementation_files": len(frozen_implementation),
        "runtime_implementation_files": len(runtime_maps[0]),
        "run_manifests": manifests,
        "audit_implementation_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> None:
    args = parse_args()
    payload = audit(
        Path(args.formal_protocol),
        [Path(value) for value in args.run_dir],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "frozen_implementation_files": payload[
                    "frozen_implementation_files"
                ],
                "runtime_implementation_files": payload[
                    "runtime_implementation_files"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
