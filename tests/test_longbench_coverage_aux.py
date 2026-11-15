from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from proofkv.prefill_selectors import GuidepostPlan
from run_hotpot_compact_qa import (
    _augment_plan_with_coverage,
    _coverage_chunks,
    parse_args,
)


def _base_plan() -> GuidepostPlan:
    return GuidepostPlan(
        guidepost_indices=[10, 11],
        retained_indices=[0, 1, 2, 3, 10, 11, *range(90, 100)],
        spans=[(10, 12)],
        retained_ratio=14 / 100,
    )


def test_hotpot_runner_cli_parses_upgrade_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_hotpot_compact_qa.py", "--out-dir", "/tmp/test-output"],
    )

    args = parse_args()

    assert args.learned_dense_core_count == 4
    assert args.learned_edge_batch_size == 256


def test_recent_coverage_fills_only_unused_budget() -> None:
    plan, auxiliary = _augment_plan_with_coverage(
        _base_plan(),
        seq_len=100,
        guidepost_budget=10,
        sink_size=4,
        recent_size=10,
        evidence_end=90,
        mode="recent",
        root_spans=[],
        graph_spans=[],
    )

    assert plan.guidepost_indices == [10, 11, *range(82, 90)]
    assert auxiliary == list(range(82, 90))
    assert len(plan.retained_indices) == 24


def test_mixed_coverage_is_deterministic_and_preserves_core() -> None:
    arguments = dict(
        base_plan=_base_plan(),
        seq_len=100,
        guidepost_budget=18,
        sink_size=4,
        recent_size=10,
        evidence_end=90,
        mode="mixed",
        root_spans=[(20, 28), (50, 58)],
        graph_spans=[(30, 38), (70, 78)],
    )
    first, first_auxiliary = _augment_plan_with_coverage(**arguments)
    second, second_auxiliary = _augment_plan_with_coverage(**arguments)

    assert first == second
    assert first_auxiliary == second_auxiliary
    assert {10, 11} <= set(first.guidepost_indices)
    assert len(first.guidepost_indices) == 18
    assert len(first_auxiliary) == 16
    assert not (set(first_auxiliary) & {0, 1, 2, 3, *range(90, 100)})


def test_mixed_coverage_splits_long_spans_before_round_robin() -> None:
    chunks = _coverage_chunks(
        mode="mixed",
        evidence_start=4,
        evidence_end=196,
        remaining_budget=64,
        root_spans=[(20, 84)],
        graph_spans=[(100, 164)],
        chunk_size=16,
    )

    assert chunks[0] == list(range(20, 36))
    assert chunks[1] == list(range(100, 116))
    assert chunks[2] == list(range(180, 196))[::-1]
    assert chunks[3] == list(range(4, 20))
    assert all(len(chunk) <= 16 for chunk in chunks)

    plan, auxiliary = _augment_plan_with_coverage(
        GuidepostPlan(
            guidepost_indices=[],
            retained_indices=[0, 1, 2, 3, *range(196, 200)],
            spans=[],
            retained_ratio=8 / 200,
        ),
        seq_len=200,
        guidepost_budget=64,
        sink_size=4,
        recent_size=4,
        evidence_end=196,
        mode="mixed",
        root_spans=[(20, 84)],
        graph_spans=[(100, 164)],
    )

    assert len(auxiliary) == 64
    assert len(plan.guidepost_indices) == 64
    assert len(set(auxiliary) & set(range(20, 36))) == 16
    assert len(set(auxiliary) & set(range(100, 116))) == 16
    assert len(set(auxiliary) & set(range(180, 196))) == 16
    assert len(set(auxiliary) & set(range(4, 20))) == 16


@pytest.mark.parametrize(
    "mode", ["recent", "stratified", "graph", "mixed"]
)
@pytest.mark.parametrize("guidepost_budget", [2, 3, 16, 64, 160])
def test_coverage_modes_preserve_exact_budget_and_disjointness(
    mode: str,
    guidepost_budget: int,
) -> None:
    base = GuidepostPlan(
        guidepost_indices=[10, 11],
        retained_indices=[0, 1, 2, 3, 10, 11, *range(190, 200)],
        spans=[(10, 12)],
        retained_ratio=16 / 200,
    )
    plan, auxiliary = _augment_plan_with_coverage(
        base,
        seq_len=200,
        guidepost_budget=guidepost_budget,
        sink_size=4,
        recent_size=10,
        evidence_end=190,
        mode=mode,
        root_spans=[(20, 84), (120, 136)],
        graph_spans=[(90, 118), (150, 175)],
    )

    stream = {0, 1, 2, 3, *range(190, 200)}
    assert len(plan.guidepost_indices) == guidepost_budget
    assert len(auxiliary) == guidepost_budget - 2
    assert {10, 11} <= set(plan.guidepost_indices)
    assert set(auxiliary).isdisjoint(stream | {10, 11})
    assert len(plan.guidepost_indices) == len(set(plan.guidepost_indices))
    assert set(plan.retained_indices) == stream | set(plan.guidepost_indices)
    assert plan.retained_indices == sorted(plan.retained_indices)
