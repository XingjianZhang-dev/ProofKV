from __future__ import annotations

from proofkv.proofcache.index import EvidenceIndexer
from proofkv.proofcache.schema import ProofOverlay
from run_hotpot_compact_qa_upgraded_v10 import METHODS, make_plan
from run_longbench_structured_upgraded_v10 import (
    SELECTOR_METHODS as LONGBENCH_SELECTOR_METHODS,
)


def _character_offsets(text: str) -> list[tuple[int, int]]:
    return [(index, index + 1) for index in range(len(text))]


def test_learned_root_control_is_registered_for_natural_and_longbench_runs() -> None:
    assert "learned_root_topk" in METHODS
    assert "learned_root_topk" in LONGBENCH_SELECTOR_METHODS


def test_learned_root_control_uses_independent_root_scores_under_same_budget() -> None:
    prefix = "P" * 24
    low = "low relevance sentence"
    separator = "S" * 12
    high = "high relevance sentence"
    suffix = "T" * 24
    question = "Which sentence is relevant?"
    prompt = f"{prefix}{low}{separator}{high}{suffix}{question}"
    low_start = len(prefix)
    high_start = low_start + len(low) + len(separator)
    spans = [
        (low_start, low_start + len(low), low),
        (high_start, high_start + len(high), high),
    ]
    index = EvidenceIndexer().build("root-control", spans, len(prompt))
    overlay = ProofOverlay(
        query=question,
        proofs=(),
        root_scores={"s0": 0.10, "s1": 0.95},
        confidence=0.95,
    )
    source = {"id": "root-control", "question": question}
    sentences = [
        {
            "title": f"Document {idx}",
            "text": text,
            "sent_id": idx,
            "gold": idx == 1,
            "char_span": (start, end),
        }
        for idx, (start, end, text) in enumerate(spans)
    ]
    metadata: dict[str, object] = {}
    plan, _, _ = make_plan(
        "learned_root_topk",
        source,
        prompt,
        question,
        sentences,
        _character_offsets(prompt),
        budget=0.50,
        learned_bundle=("cross", object(), object()),
        learned_overlay_cache={"root-control": (overlay, index)},
        selection_metadata_out=metadata,
    )
    retained = set(plan.retained_indices)
    assert set(range(high_start, high_start + len(high))) <= retained
    assert metadata["proof_core_span_ids"] == ["s1"]
