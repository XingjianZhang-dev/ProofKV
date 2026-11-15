from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_longbench_risk_formal_statistics import (
    holm_adjust,
    paired_inference,
)
from analyze_longbench_utility_formal_statistics import (
    domain_paired_delta_matrix,
)


def test_paired_inference_uses_equal_task_weighting() -> None:
    deltas = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
    )

    result = paired_inference(
        deltas,
        bootstrap_replicates=2000,
        permutation_replicates=2000,
        rng=np.random.default_rng(17),
    )

    assert result["paired_task_macro_delta"] == 2.0
    assert result["bootstrap_95_ci_low"] == 2.0
    assert result["bootstrap_95_ci_high"] == 2.0
    assert result["pairs"] == 8
    assert result["two_sided_sign_flip_p"] < 0.05


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    records = [
        {"two_sided_sign_flip_p": 0.01},
        {"two_sided_sign_flip_p": 0.04},
        {"two_sided_sign_flip_p": 0.03},
    ]

    holm_adjust(records)

    assert records[0]["holm_adjusted_p"] == 0.03
    assert records[1]["holm_adjusted_p"] == 0.06
    assert records[2]["holm_adjusted_p"] == 0.06


def test_domain_paired_matrix_uses_each_tasks_frozen_fallback() -> None:
    rows = []
    for task, fallback, utility, physical in (
        ("a", "streamingllm", 0.6, 0.5),
        ("b", "snapkv", 0.4, 0.7),
    ):
        for sample_id in ("0", "1"):
            rows.extend(
                [
                    {
                        "task": task,
                        "id": sample_id,
                        "method": "utility_calibrated_proof_coverage",
                        "capacity": 64,
                        "score": utility,
                    },
                    {
                        "task": task,
                        "id": sample_id,
                        "method": fallback,
                        "capacity": 64,
                        "score": physical,
                    },
                ]
            )

    matrix = domain_paired_delta_matrix(
        rows,
        ["a", "b"],
        "utility_calibrated_proof_coverage",
        {"a": "streamingllm", "b": "snapkv"},
        64,
    )

    np.testing.assert_allclose(
        matrix,
        np.asarray([[10.0, 10.0], [-30.0, -30.0]]),
    )
