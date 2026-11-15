import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_kvpress_natural_qa import fixed_budget_targets


def test_fixed_budget_deducts_uncompressed_question_tokens() -> None:
    retained_total, retained_context, effective_budget = fixed_budget_targets(
        context_tokens=1000,
        question_tokens=60,
        budget=0.08,
        budget_tokens=512,
    )

    assert retained_total == 512
    assert retained_context == 452
    assert effective_budget == 0.452
    assert retained_context + 60 == retained_total


def test_ratio_budget_uses_total_prefill_scope() -> None:
    retained_total, retained_context, effective_budget = fixed_budget_targets(
        context_tokens=940,
        question_tokens=60,
        budget=0.08,
        budget_tokens=None,
    )

    assert retained_total == 80
    assert retained_context == 20
    assert effective_budget == 20 / 940


def test_budget_never_requests_more_context_than_exists() -> None:
    retained_total, retained_context, effective_budget = fixed_budget_targets(
        context_tokens=32,
        question_tokens=12,
        budget=0.08,
        budget_tokens=512,
    )

    assert retained_total == 44
    assert retained_context == 32
    assert effective_budget == 1.0
