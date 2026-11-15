from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calibrate_longbench_utility_router import (
    RISK_FEATURES,
    evaluate_predictions,
    fit_ridge,
    predict,
)
from run_longbench_structured_common import (
    utility_feature_values,
    utility_prediction,
    validate_utility_controller,
)
from audit_longbench_utility_formal import (
    predicted_utility,
    predicted_utility_from_features,
)
from calibrate_longbench_domain_utility_router import (
    stable_candidate_gate,
)
from analyze_longbench_utility_formal_statistics import (
    domain_paired_delta_matrix,
)


CANDIDATES = (
    "learned_and_or_recent_aux",
    "learned_and_or_stratified_aux",
    "learned_and_or_graph_aux",
    "learned_and_or_mixed_aux",
)


def _utility_model() -> dict:
    coefficients = [0.0] * len(RISK_FEATURES)
    coefficients[0] = 2.0
    return {
        "feature_names": list(RISK_FEATURES),
        "feature_mean": [0.0] * len(RISK_FEATURES),
        "feature_scale": [1.0] * len(RISK_FEATURES),
        "coefficients": coefficients,
        "intercept": -0.5,
    }


def _valid_controller() -> dict:
    return {
        "status": "PASS",
        "predecode_only": True,
        "feature_names": list(RISK_FEATURES),
        "candidate_methods": list(CANDIDATES),
        "final_policy_by_model_capacity": {
            "qwen25:64": {
                "fallback_method": "streamingllm",
                "utility_margin": 0.25,
                "candidate_models": {
                    method: _utility_model() for method in CANDIDATES
                },
            }
        },
    }


def _valid_domain_controller() -> dict:
    controller = _valid_controller()
    policy = controller.pop("final_policy_by_model_capacity")["qwen25:64"]
    policy["eligible_candidate_methods"] = [
        "learned_and_or_mixed_aux",
    ]
    policy["candidate_stability"] = {
        method: {
            "eligible": method == "learned_and_or_mixed_aux",
        }
        for method in CANDIDATES
    }
    policy["stability_eligible_candidate_methods"] = [
        "learned_and_or_mixed_aux",
    ]
    controller.update(
        {
            "domain_conditioned": True,
            "tasks": ["narrativeqa", "qasper"],
            "final_policy_by_model_task_capacity": {
                "qwen25:narrativeqa:64": policy,
                "qwen25:qasper:64": {
                    **policy,
                    "eligible_candidate_methods": [],
                    "stability_eligible_candidate_methods": [],
                    "utility_margin": 1.0e9,
                    "candidate_stability": {
                        method: {"eligible": False}
                        for method in CANDIDATES
                    },
                },
            },
        }
    )
    return controller


def test_ridge_utility_model_round_trips_serialized_parameters() -> None:
    examples = []
    for index in range(32):
        x = np.zeros(len(RISK_FEATURES), dtype=np.float64)
        x[0] = index / 10
        x[1] = (index % 3) / 3
        examples.append({"x": x, "utility": 1.5 + 2.0 * x[0]})
    model = fit_ridge(examples, alpha=0.1)
    assert model["feature_names"] == list(RISK_FEATURES)
    assert model["training_examples"] == 32
    probe = np.zeros(len(RISK_FEATURES), dtype=np.float64)
    probe[0] = 1.25
    assert abs(predict(model, probe) - 4.0) < 0.1


def test_utility_margin_routes_only_predictions_above_threshold() -> None:
    records = [
        {
            "task": "a",
            "predicted_utility": 1.5,
            "proof_score": 8.0,
            "fallback_score": 3.0,
        },
        {
            "task": "a",
            "predicted_utility": -0.2,
            "proof_score": 0.0,
            "fallback_score": 4.0,
        },
        {
            "task": "b",
            "predicted_utility": 0.4,
            "proof_score": 9.0,
            "fallback_score": 2.0,
        },
    ]
    result = evaluate_predictions(records, margin=0.5)
    assert result["proof_routes"] == 1
    assert result["proof_route_coverage"] == 1 / 3
    assert result["task_deltas"] == {"a": 2.5, "b": 0.0}
    assert result["delta"] == 1.25


def test_runtime_utility_prediction_matches_serialized_linear_model() -> None:
    model = _utility_model()
    partition = {
        name: 0.0
        for name in RISK_FEATURES
        if name
        not in {
            "proof_core_fraction",
            "coverage_auxiliary_fraction",
            "context_log_tokens",
            "question_budget_fraction",
        }
    }
    partition.update(
        {
            "overlay_confidence": 0.75,
            "proof_core_tokens": 0,
            "coverage_auxiliary_tokens": 0,
        }
    )
    runtime = utility_prediction(model, partition, 64, 1000, 16)
    features = utility_feature_values(partition, 64, 1000, 16)
    audit = predicted_utility(
        model,
        {
            "capacity": 64,
            "context_tokens": 1000,
            "question_tokens": 16,
            "selection_partition": partition,
        },
    )
    audit_from_features = predicted_utility_from_features(
        model,
        features,
    )
    assert runtime == 1.0
    assert audit == runtime
    assert audit_from_features == runtime
    assert features["overlay_confidence"] == 0.75
    assert set(features) == set(RISK_FEATURES)


def test_runtime_validates_complete_model_specific_policy() -> None:
    validate_utility_controller(_valid_controller(), "qwen25", [64])


def test_runtime_validates_domain_stability_gate_policies() -> None:
    validate_utility_controller(
        _valid_domain_controller(),
        "qwen25",
        [64],
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(status="FAIL"), "calibration gate"),
        (
            lambda value: value["final_policy_by_model_capacity"][
                "qwen25:64"
            ].update(utility_margin=float("nan")),
            "non-finite utility margin",
        ),
        (
            lambda value: value["final_policy_by_model_capacity"][
                "qwen25:64"
            ]["candidate_models"]["learned_and_or_recent_aux"].update(
                feature_scale=[0.0] * len(RISK_FEATURES)
            ),
            "nonpositive utility feature scale",
        ),
    ],
)
def test_invalid_utility_controller_is_rejected(mutation, message: str) -> None:
    controller = _valid_controller()
    mutation(controller)
    with pytest.raises(ValueError, match=message):
        validate_utility_controller(controller, "qwen25", [64])


def test_invalid_domain_candidate_eligibility_is_rejected() -> None:
    controller = _valid_domain_controller()
    controller["final_policy_by_model_task_capacity"][
        "qwen25:qasper:64"
    ]["eligible_candidate_methods"] = ["not_a_candidate"]
    with pytest.raises(ValueError, match="eligible utility candidates"):
        validate_utility_controller(controller, "qwen25", [64])


def test_domain_gate_rejects_phase_unstable_candidate() -> None:
    task = "narrativeqa"
    prefix = [f"prefix-{index}" for index in range(10)]
    extension = [f"extension-{index}" for index in range(18)]
    physical = {}
    coverage = {}
    for sample_id in [*prefix, *extension]:
        for method in (
            "streamingllm",
            "snapkv",
            "pyramidkv",
            "chunkkv_snapkv",
        ):
            physical[(task, sample_id, method, 64)] = {"score": 0.5}
        for method in CANDIDATES:
            candidate_score = 0.4
            if method == "learned_and_or_mixed_aux":
                candidate_score = 0.6
            elif method == "learned_and_or_recent_aux":
                candidate_score = (
                    0.6 if sample_id.startswith("prefix") else 0.4
                )
            coverage[(task, sample_id, method, 64)] = {
                "score": candidate_score
            }

    eligible, summaries = stable_candidate_gate(
        coverage,
        physical,
        "qwen25",
        task,
        prefix,
        extension,
        64,
        "streamingllm",
    )

    assert eligible == ["learned_and_or_mixed_aux"]
    assert not summaries["learned_and_or_recent_aux"]["eligible"]
    assert summaries["learned_and_or_mixed_aux"]["eligible"]


def test_fixed_candidate_formal_pairing_preserves_task_sample_grain() -> None:
    rows = []
    for task_index, task in enumerate(("task_a", "task_b")):
        for sample_index in range(3):
            for method, score in (
                ("utility_calibrated_proof_coverage", 0.6),
                ("learned_and_or_mixed_aux", 0.4),
            ):
                rows.append(
                    {
                        "task": task,
                        "id": f"{task}:{sample_index}",
                        "method": method,
                        "capacity": 64,
                        "score": (
                            score
                            + 0.01 * task_index
                            + 0.001 * sample_index
                        ),
                    }
                )
    paired = domain_paired_delta_matrix(
        rows,
        ["task_a", "task_b"],
        "utility_calibrated_proof_coverage",
        {
            "task_a": "learned_and_or_mixed_aux",
            "task_b": "learned_and_or_mixed_aux",
        },
        64,
    )
    assert paired.shape == (2, 3)
    assert np.allclose(paired, 20.0)
