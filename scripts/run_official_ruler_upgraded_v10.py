#!/usr/bin/env python
"""Run the physical ProofKV-v10 matrix on frozen official RULER rows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import run_longbench_structured_upgraded_v10 as runner
from official_ruler_v10_common import (
    MAX_NEW_TOKENS,
    official_fields,
    score_prediction,
)


_BASE_IMPLEMENTATION_SHA256 = runner.implementation_sha256
_BASE_MAKE_PLAN = runner.make_plan
RULER_CHAIN_METHOD = "ruler_chain_adapter_v10"


def make_plan(
    method: str,
    *args: Any,
    **kwargs: Any,
):
    """Expose the existing visible assignment-chain compiler to v10 routing."""

    if method != RULER_CHAIN_METHOD:
        return _BASE_MAKE_PLAN(method, *args, **kwargs)
    output = _BASE_MAKE_PLAN("compass_adaptive", *args, **kwargs)
    plan = output[0]
    metadata = kwargs.get("selection_metadata_out")
    if metadata is not None:
        retained = len(plan.retained_indices)
        retained_set = set(map(int, plan.retained_indices))
        proof_core = set()
        for start, end in plan.spans:
            proof_core.update(range(int(start), int(end)))
        proof_core &= retained_set
        has_chain = bool(plan.spans)
        metadata.update(
            {
                "proof_core_tokens": len(proof_core),
                "coverage_auxiliary_tokens": 0,
                "overlay_confidence": float(has_chain),
                "overlay_top_proof_probability": float(has_chain),
                "overlay_proof_probability_margin": float(has_chain),
                "overlay_root_top_probability": float(has_chain),
                "overlay_root_probability_margin": float(has_chain),
                "overlay_terminal_top_probability": float(has_chain),
                "overlay_terminal_probability_margin": float(has_chain),
                "overlay_num_proofs": int(has_chain),
                "overlay_top_proof_spans": len(plan.spans),
                "external_adapter": RULER_CHAIN_METHOD,
                "external_adapter_source": (
                    "prompt-visible NIAH/VAR assignment chain compiler"
                ),
                "external_adapter_retained_tokens": retained,
            }
        )
    return output


def validate_upgraded_controller(
    controller: dict[str, Any],
    model_label: str,
    capacities: list[int],
) -> None:
    if (
        controller.get("status") != "PASS"
        or controller.get("predecode_only") is not True
        or controller.get("domain_conditioned") is not True
        or controller.get("feature_names") != list(runner.RISK_FEATURES)
    ):
        raise ValueError("official-RULER controller schema differs")
    architecture = controller.get("upgraded_architecture")
    if (
        not isinstance(architecture, dict)
        or architecture.get("status") != "PASS"
        or architecture.get("predecode_only") is not True
    ):
        raise ValueError("official-RULER architecture was not frozen")
    tasks = controller.get("tasks")
    policies = controller.get("final_policy_by_model_task_capacity")
    if not isinstance(tasks, list) or not tasks or not isinstance(policies, dict):
        raise ValueError("official-RULER controller domain is missing")
    valid_planners = {
        "learned_and_or_recent_aux",
        "learned_and_or_stratified_aux",
        "learned_and_or_graph_aux",
        "learned_and_or_mixed_aux",
        "learned_and_or_lexical_anchor25_aux",
        "learned_and_or_lexical_anchor50_aux",
        "learned_and_or_lexical_anchor75_aux",
        "learned_and_or_control_anchor25_aux",
        "learned_and_or_control_anchor50_aux",
        "learned_and_or_control_anchor75_aux",
        RULER_CHAIN_METHOD,
    }
    for task in tasks:
        for capacity in capacities:
            key = f"{model_label}:{task}:{capacity}"
            policy = policies.get(key)
            if not isinstance(policy, dict):
                raise ValueError(f"missing official-RULER policy: {key}")
            if policy.get("fallback_method") not in runner.BASELINE_METHODS:
                raise ValueError(f"invalid official-RULER fallback: {key}")
            eligible = policy.get("eligible_candidate_methods")
            models = policy.get("candidate_models")
            route = policy.get("upgraded_route")
            if (
                not isinstance(eligible, list)
                or len(eligible) != len(set(eligible))
                or not set(eligible).issubset(valid_planners)
                or not isinstance(models, dict)
                or not set(eligible).issubset(models)
            ):
                raise ValueError(f"invalid official-RULER candidates: {key}")
            if route is None:
                if eligible:
                    raise ValueError(f"fallback-only policy routes candidates: {key}")
            elif (
                not isinstance(route, dict)
                or route.get("route_type") != "static_anchor"
                or eligible != [route.get("method")]
            ):
                raise ValueError(f"official-RULER route differs: {key}")
            for method in eligible:
                utility_model = models[method]
                names = utility_model.get("feature_names")
                means = utility_model.get("feature_mean")
                scales = utility_model.get("feature_scale")
                coefficients = utility_model.get("coefficients")
                if (
                    names != list(runner.RISK_FEATURES)
                    or not isinstance(means, list)
                    or not isinstance(scales, list)
                    or not isinstance(coefficients, list)
                    or not (
                        len(means)
                        == len(scales)
                        == len(coefficients)
                        == len(names)
                    )
                ):
                    raise ValueError(f"invalid utility model: {key}/{method}")
                if (
                    any(
                        not math.isfinite(float(value))
                        for value in [
                            *means,
                            *scales,
                            *coefficients,
                            utility_model.get("intercept"),
                        ]
                    )
                    or any(float(value) <= 0.0 for value in scales)
                ):
                    raise ValueError(f"invalid utility scale: {key}/{method}")


def implementation_sha256() -> dict[str, str]:
    output = _BASE_IMPLEMENTATION_SHA256()
    for path in [
        Path(__file__).resolve(),
        Path(__file__).with_name("official_ruler_v10_common.py").resolve(),
    ]:
        output[str(path)] = runner.sha256_file(path)
    return output


def main() -> None:
    runner.METHODS.add(RULER_CHAIN_METHOD)
    runner.SELECTOR_METHODS.add(RULER_CHAIN_METHOD)
    runner.official_fields = official_fields
    runner.best_qa_f1 = score_prediction
    runner.MAX_NEW_TOKENS.update(MAX_NEW_TOKENS)
    runner.implementation_sha256 = implementation_sha256
    runner.make_plan = make_plan
    runner.validate_upgraded_controller = validate_upgraded_controller
    runner.main()


if __name__ == "__main__":
    main()
