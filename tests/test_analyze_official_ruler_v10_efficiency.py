from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_official_ruler_v10_efficiency as ruler_efficiency


def test_specialization_rebinds_protocol_and_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = tmp_path / "wrapper.py"
    base = tmp_path / "base.py"
    wrapper.write_text("wrapper\n")
    base.write_text("base\n")
    monkeypatch.setattr(
        ruler_efficiency.base,
        "sha256_file",
        lambda path: {
            wrapper: "wrapper-sha",
            base: "base-sha",
        }[path],
    )
    payload = {
        "status": "PASS",
        "protocol": "proofkv_v10_physical_efficiency_audit_v1",
        "analysis_implementation_sha256": "base-sha",
        "input_runs": [
            {
                "capacities": [128, 256, 512, 1024],
                "rows": 1_974,
            },
            {
                "capacities": [128, 256, 512, 1024],
                "rows": 1_974,
            },
        ],
        "validation": {"completed_runs": 2, "rows": 3_948},
    }

    result = ruler_efficiency.specialize_payload(
        payload,
        wrapper,
        base,
    )

    assert result["status"] == "PASS"
    assert (
        result["protocol"]
        == "proofkv_v10_official_ruler_physical_efficiency_audit_v1"
    )
    assert result["base_protocol"] == (
        "proofkv_v10_physical_efficiency_audit_v1"
    )
    assert result["analysis_implementation_sha256"] == "wrapper-sha"
    assert result["base_analysis_implementation_sha256"] == "base-sha"
    assert result["validation"]["rows"] == 3_948


def test_specialization_rejects_longbench_capacities(tmp_path: Path) -> None:
    wrapper = tmp_path / "wrapper.py"
    base = tmp_path / "base.py"
    wrapper.write_text("wrapper\n")
    base.write_text("base\n")
    payload = {
        "status": "PASS",
        "protocol": "proofkv_v10_physical_efficiency_audit_v1",
        "analysis_implementation_sha256": "irrelevant",
        "input_runs": [
            {
                "capacities": [64, 128, 256, 512],
                "rows": 1_974,
            }
        ],
        "validation": {"completed_runs": 1, "rows": 1_974},
    }

    with pytest.raises(ValueError, match="capacities differ"):
        ruler_efficiency.specialize_payload(payload, wrapper, base)
