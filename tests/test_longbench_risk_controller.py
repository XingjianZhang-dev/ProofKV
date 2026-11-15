from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_longbench_structured_common import (
    calibrated_proof_method,
    risk_feature_value,
    risk_rule_uses_proof,
    validate_risk_controller,
)
from calibrate_longbench_risk_controller import (
    CAPACITIES,
    COVERAGE_METHODS,
    PHYSICAL_METHODS,
    choose_coverage_methods,
    choose_fallbacks,
    routed_macro_by_capacity_rules,
)


def _valid_controller() -> dict:
    return {
        "status": "PASS",
        "proof_method_by_capacity": {
            "64": "learned_and_or_recent_aux",
            "128": "learned_and_or_mixed_aux",
        },
        "final_risk_rule_by_capacity": {
            "64": {
                "feature": "overlay_confidence",
                "direction": "high",
                "threshold": 0.75,
            },
            "128": {
                "feature": "question_budget_fraction",
                "direction": "low",
                "threshold": 0.25,
            },
        },
        "fallback_by_model_capacity": {
            "qwen25:64": "streamingllm",
            "qwen25:128": "snapkv",
        },
    }


def test_risk_feature_values_match_calibration_definitions() -> None:
    partition = {
        "proof_core_tokens": 24,
        "coverage_auxiliary_tokens": 16,
        "overlay_confidence": 0.8,
    }

    assert risk_feature_value(
        "proof_core_fraction", partition, 64, 1000, 24
    ) == 24 / 64
    assert risk_feature_value(
        "coverage_auxiliary_fraction", partition, 64, 1000, 24
    ) == 16 / 64
    assert risk_feature_value(
        "context_log_tokens", partition, 64, 1000, 24
    ) == math.log1p(1000)
    assert risk_feature_value(
        "question_budget_fraction", partition, 64, 1000, 24
    ) == 24 / 64
    assert risk_feature_value(
        "overlay_confidence", partition, 64, 1000, 24
    ) == 0.8


@pytest.mark.parametrize(
    ("value", "direction", "threshold", "expected"),
    [
        (0.8, "high", 0.75, True),
        (0.75, "high", 0.75, True),
        (0.7, "high", 0.75, False),
        (0.2, "low", 0.25, True),
        (0.25, "low", 0.25, True),
        (0.3, "low", 0.25, False),
    ],
)
def test_risk_rule_boundary_is_inclusive(
    value: float,
    direction: str,
    threshold: float,
    expected: bool,
) -> None:
    assert risk_rule_uses_proof(value, direction, threshold) is expected


def test_risk_controller_validates_before_model_load() -> None:
    controller = _valid_controller()
    validate_risk_controller(controller, "qwen25", [64, 128])
    assert calibrated_proof_method(controller, 64) == (
        "learned_and_or_recent_aux"
    )
    assert calibrated_proof_method(controller, 128) == (
        "learned_and_or_mixed_aux"
    )


def test_coverage_residual_is_selected_separately_by_capacity() -> None:
    tasks = ["task-a", "task-b"]
    expected = {
        64: "learned_and_or_recent_aux",
        128: "learned_and_or_stratified_aux",
        256: "learned_and_or_graph_aux",
        512: "learned_and_or_mixed_aux",
    }
    base_rows = []
    coverage_rows = []
    for task in tasks:
        for capacity in CAPACITIES:
            for method in PHYSICAL_METHODS:
                base_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "capacity": capacity,
                        "score": (
                            0.5 if method == "streamingllm" else 0.1
                        ),
                    }
                )
            for method in COVERAGE_METHODS:
                coverage_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "capacity": capacity,
                        "score": 0.6 if method == expected[capacity] else 0.2,
                    }
                )
    base_by_model = {"model": base_rows}
    coverage_by_model = {"model": coverage_rows}
    fallbacks = choose_fallbacks(base_by_model, tasks)

    selected, records = choose_coverage_methods(
        coverage_by_model,
        base_by_model,
        tasks,
        fallbacks,
    )

    assert selected == expected
    assert len(records) == len(CAPACITIES) * len(COVERAGE_METHODS)


def test_risk_rules_are_applied_separately_by_capacity() -> None:
    observations = [
        {
            "model": "model",
            "task": "task",
            "id": f"id-{capacity}",
            "capacity": capacity,
            "proof_score": 10.0,
            "fallback_score": float(capacity / 64),
            "features": {"overlay_confidence": 0.9},
        }
        for capacity in CAPACITIES
    ]
    rules = {
        64: {
            "feature": "overlay_confidence",
            "direction": "high",
            "threshold": 0.5,
        },
        128: {
            "feature": "overlay_confidence",
            "direction": "low",
            "threshold": 0.5,
        },
        256: {
            "feature": "overlay_confidence",
            "direction": "high",
            "threshold": 0.5,
        },
        512: {
            "feature": "overlay_confidence",
            "direction": "low",
            "threshold": 0.5,
        },
    }

    score, coverage = routed_macro_by_capacity_rules(
        observations,
        rules,
    )

    assert score == (10.0 + 2.0 + 10.0 + 8.0) / 4
    assert coverage == 0.5


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(status="FAIL"), "calibration gate"),
        (
            lambda value: value["proof_method_by_capacity"].update(
                {"128": "learned_and_or_aux"}
            ),
            "proof method",
        ),
        (
            lambda value: value["final_risk_rule_by_capacity"]["128"].update(
                feature="postdecode_f1"
            ),
            "risk feature",
        ),
        (
            lambda value: value["final_risk_rule_by_capacity"]["128"].update(
                direction="sideways"
            ),
            "direction",
        ),
        (
            lambda value: value["final_risk_rule_by_capacity"]["128"].update(
                threshold=float("nan")
            ),
            "finite",
        ),
        (
            lambda value: value["fallback_by_model_capacity"].pop("qwen25:128"),
            "physical fallback",
        ),
    ],
)
def test_invalid_risk_controller_is_rejected(mutation, message: str) -> None:
    controller = _valid_controller()
    mutation(controller)
    with pytest.raises(ValueError, match=message):
        validate_risk_controller(controller, "qwen25", [64, 128])
