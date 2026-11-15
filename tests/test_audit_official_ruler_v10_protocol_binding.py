from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_official_ruler_v10_protocol_binding import audit


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, list[Path], Path]:
    implementation = tmp_path / "runner.py"
    implementation.write_text("print('frozen')\n")
    formal_manifest = tmp_path / "formal_manifest.json"
    formal_manifest.write_text("{}\n")
    controller = tmp_path / "controller.json"
    controller.write_text("{}\n")
    run_dirs = [tmp_path / "qwen", tmp_path / "mistral"]
    runtime = {str(implementation): sha256(implementation)}
    for run_dir in run_dirs:
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "selection_manifest_sha256": sha256(formal_manifest),
                    "risk_controller_sha256": sha256(controller),
                    "implementation_sha256": runtime,
                }
            )
        )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "protocol": "proofkv_v10_official_ruler_formal_protocol_v1",
                "frozen_before_any_formal_output": True,
                "formal_manifest_sha256": sha256(formal_manifest),
                "controller_sha256": sha256(controller),
                "formal_output_directories": [str(path) for path in run_dirs],
                "implementation_sha256": runtime,
            }
        )
    )
    return protocol, run_dirs, implementation


def test_protocol_binding_passes(tmp_path: Path) -> None:
    protocol, run_dirs, _ = fixture(tmp_path)
    payload = audit(protocol, run_dirs)
    assert payload["status"] == "PASS"
    assert payload["runtime_implementation_files"] == 1


def test_protocol_binding_detects_runtime_drift(tmp_path: Path) -> None:
    protocol, run_dirs, implementation = fixture(tmp_path)
    implementation.write_text("print('changed')\n")
    with pytest.raises(ValueError, match="frozen implementation drift"):
        audit(protocol, run_dirs)


def test_protocol_binding_detects_manifest_mismatch(tmp_path: Path) -> None:
    protocol, run_dirs, _ = fixture(tmp_path)
    manifest = json.loads((run_dirs[0] / "run_manifest.json").read_text())
    manifest["implementation_sha256"][next(iter(manifest["implementation_sha256"]))] = (
        "0" * 64
    )
    (run_dirs[0] / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="runtime implementation differs"):
        audit(protocol, run_dirs)
