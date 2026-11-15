from __future__ import annotations

import numpy as np

from analyze_fair_contract_external_20260726 import (
    EXTERNAL_TASKS,
    clustered_inference,
    holm_adjust,
)


def test_clustered_inference_preserves_equal_cluster_weighting() -> None:
    clusters = {
        ("qwen", "task_a", 128): np.asarray([0.1, 0.1]),
        ("mistral", "task_b", 128): np.asarray([0.3]),
    }
    result = clustered_inference(
        clusters,
        bootstrap_replicates=2_000,
        permutation_replicates=2_000,
        rng=np.random.default_rng(7),
    )
    assert result["delta_f1_points"] == 20.0
    assert result["clusters"] == 2
    assert result["paired_examples"] == 3


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    records = [
        {"permutation_p_two_sided": 0.01},
        {"permutation_p_two_sided": 0.04},
    ]
    holm_adjust(records)
    assert records[0]["holm_p_two_sided"] == 0.02
    assert records[1]["holm_p_two_sided"] == 0.04


def test_external_tasks_exclude_every_training_family() -> None:
    assert EXTERNAL_TASKS == ("qasper", "multifieldqa_en")
