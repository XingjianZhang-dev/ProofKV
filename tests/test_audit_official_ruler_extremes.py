from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_official_ruler_extremes import audit_run, parse_run


def result(
    source_id: str,
    method: str,
    capacity: int | str,
    score: float,
    prediction: str,
) -> dict:
    return {
        "id": source_id,
        "task": "ruler_variable_tracking",
        "method": method,
        "capacity": capacity,
        "answers": ["AAA"],
        "prediction": prediction,
        "score": score,
        "actual_retained_tokens": 8000 if capacity == "full" else capacity,
    }


def write_run(tmp_path: Path, physical_score: float = 0.0) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        result("sample", "full", "full", 1.0, "AAA"),
        result("sample", "ruler_chain_adapter_v10", 128, 1.0, "AAA"),
    ]
    for method in (
        "streamingllm",
        "snapkv",
        "pyramidkv",
        "chunkkv_snapkv",
    ):
        rows.append(
            result(
                "sample",
                method,
                128,
                physical_score,
                "AAA" if physical_score else "wrong",
            )
        )
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return run_dir


def test_parse_run_requires_label_candidate_and_directory() -> None:
    assert parse_run("dev_q=ruler_chain_adapter_v10=/tmp/run") == (
        "dev_q",
        "ruler_chain_adapter_v10",
        Path("/tmp/run"),
    )
    with pytest.raises(ValueError):
        parse_run("dev_q=/tmp/run")


def test_extreme_audit_replays_metric_and_budget(tmp_path: Path) -> None:
    record = audit_run(
        "dev_q",
        "ruler_chain_adapter_v10",
        write_run(tmp_path),
        "ruler_variable_tracking",
        128,
        1,
        True,
        90.0,
    )
    assert record["complete_samples"] == 1
    assert record["score_means"]["full"] == 100.0
    assert record["score_means"]["streamingllm"] == 0.0
    assert record["score_means"]["ruler_chain_adapter_v10"] == 100.0
    assert record["all_physical_predictions_zero"] is True


def test_extreme_audit_rejects_nonzero_baseline_when_required(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="physical baseline is nonzero"):
        audit_run(
            "dev_q",
            "ruler_chain_adapter_v10",
            write_run(tmp_path, physical_score=1.0),
            "ruler_variable_tracking",
            128,
            1,
            True,
            90.0,
        )
