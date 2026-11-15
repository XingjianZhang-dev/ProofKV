from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_official_ruler_v10_statistics import paired_matrix
from freeze_official_ruler_v10_controller import (
    RISK_FEATURES,
    constant_utility_model,
)
from official_ruler_v10_common import (
    convert_official_row,
    official_fields,
    score_prediction,
)


def test_convert_official_niah_preserves_terminal_question_tail() -> None:
    converted = convert_official_row(
        "ruler_niah",
        {
            "index": 3,
            "input": (
                "Remember the text.\nThe key for blue-fox is 7319.\n"
                "What is the special magic number for blue-fox?"
            ),
            "outputs": ["7319"],
            "answer_prefix": " The special magic number is",
        },
    )

    assert converted["context"].endswith("7319.")
    assert converted["input"].startswith("What is")
    assert official_fields("ruler_niah", converted) == (
        converted["context"],
        converted["input"],
        converted["answer_prefix"],
    )


def test_convert_official_variable_tracking_uses_last_question() -> None:
    converted = convert_official_row(
        "ruler_variable_tracking",
        {
            "index": 0,
            "input": (
                "Question: this is an in-context demonstration.\n"
                "VAR AAA = 42\nVAR BBB = VAR AAA\n"
                "Question: Find all variables assigned 42."
            ),
            "outputs": ["AAA", "BBB"],
            "answer_prefix": " Answer: they are:",
        },
    )

    assert "in-context demonstration" in converted["context"]
    assert converted["input"] == "Question: Find all variables assigned 42."
    assert converted["answers"] == ["AAA", "BBB"]


def test_official_ruler_score_is_reference_coverage() -> None:
    assert score_prediction("AAA then BBB", ["AAA", "BBB"]) == 1.0
    assert score_prediction("AAA only", ["AAA", "BBB"]) == 0.5
    assert score_prediction("The answer is 7319.", ["7319"]) == 1.0


def test_constant_utility_model_forces_prospective_static_route() -> None:
    model = constant_utility_model()

    assert model["feature_names"] == RISK_FEATURES
    assert model["intercept"] == 1.0
    assert not any(model["coefficients"])
    assert all(scale == 1.0 for scale in model["feature_scale"])


def test_ruler_paired_matrix_uses_task_specific_fallbacks() -> None:
    rows = []
    for task, fallback, v10, baseline in (
        ("ruler_niah", "streamingllm", 1.0, 0.5),
        ("ruler_variable_tracking", "snapkv", 0.5, 0.75),
    ):
        for source_id in ("a", "b"):
            rows.extend(
                [
                    {
                        "task": task,
                        "id": f"{task}:{source_id}",
                        "method": "confirmation_stable_proofkv_v10",
                        "capacity": 64,
                        "score": v10,
                    },
                    {
                        "task": task,
                        "id": f"{task}:{source_id}",
                        "method": fallback,
                        "capacity": 64,
                        "score": baseline,
                    },
                ]
            )

    matrix = paired_matrix(
        rows,
        ["ruler_niah", "ruler_variable_tracking"],
        64,
        {
            "ruler_niah": "streamingllm",
            "ruler_variable_tracking": "snapkv",
        },
    )

    np.testing.assert_allclose(
        matrix,
        np.asarray([[50.0, 50.0], [-25.0, -25.0]]),
    )
