from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_longbench_v10_efficiency import load_and_validate_run


def write_run(tmp_path: Path, blank_score: float) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = {
        "model": "test-model",
        "tasks": ["task"],
        "methods": ["full", "confirmation_stable_proofkv_v10"],
        "capacities": [64, 128, 256, 512],
        "n_per_task": 1,
        "kv_bytes_per_token_bf16": 4.0,
        "truncated_inputs": 0,
        "selection_manifest_sha256": "selection",
    }
    manifest = {
        "implementation_sha256": {"runner": "implementation"},
        "scorer_checkpoint_sha256": {"scorer": "checkpoint"},
    }
    rows = []
    for method, capacity, retained, prediction, score in (
        ("full", "full", 10, "answer", 1.0),
        (
            "confirmation_stable_proofkv_v10",
            64,
            10,
            "\n\t",
            blank_score,
        ),
        ("confirmation_stable_proofkv_v10", 128, 10, "answer", 1.0),
        ("confirmation_stable_proofkv_v10", 256, 10, "answer", 1.0),
        ("confirmation_stable_proofkv_v10", 512, 10, "answer", 1.0),
    ):
        rows.append(
            {
                "task": "task",
                "id": "task:0",
                "method": method,
                "capacity": capacity,
                "score": score,
                "prediction": prediction,
                "prompt_tokens": 10,
                "actual_retained_tokens": retained,
                "actual_retained_ratio": 1.0,
                "actual_kv_bytes": 40.0,
                "selection_seconds": 0.25,
                "generation_seconds": 0.75,
                "seconds": 1.0,
                "peak_cuda_memory_gb": 1.0,
            }
        )
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return run_dir


def test_blank_zero_score_prediction_is_retained_and_reported(
    tmp_path: Path,
) -> None:
    _, rows, report = load_and_validate_run(write_run(tmp_path, 0.0), "confirmation_stable_proofkv_v10")

    assert len(rows) == 5
    assert report["empty_predictions"] == 1
    assert report["empty_string_predictions"] == 0
    assert report["whitespace_only_predictions"] == 1
    assert report["empty_predictions_by_task_method_capacity"] == [
        {
            "task": "task",
            "method": "confirmation_stable_proofkv_v10",
            "capacity": "64",
            "count": 1,
        }
    ]


def test_blank_nonzero_score_prediction_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="blank prediction has a nonzero score"):
        load_and_validate_run(
            write_run(tmp_path, 0.5),
            "confirmation_stable_proofkv_v10",
        )
