from __future__ import annotations

from eval_natural_route_coverage import SINK, adaptive_budget_parts
from run_hotpot_compact_qa import METHODS, make_plan


def _character_offsets(text: str) -> list[tuple[int, int]]:
    return [(index, index + 1) for index in range(len(text))]


def test_bm25_plan_retains_query_matching_evidence_under_fixed_budget() -> None:
    irrelevant_a = "Gardening notes discuss roses, soil, and watering schedules. " * 3
    relevant = "Marie Curie discovered polonium and conducted radiation research. "
    irrelevant_b = "A cookbook describes bread, soup, and seasonal vegetables. " * 3
    question = "Who discovered polonium?"
    parts = [irrelevant_a, relevant, irrelevant_b]
    prompt = "\n".join(parts) + f"\nQuestion: {question}"
    starts = []
    cursor = 0
    for part in parts:
        starts.append((cursor, cursor + len(part)))
        cursor += len(part) + 1
    sentences = [
        {
            "title": title,
            "text": text,
            "sent_id": index,
            "gold": index == 1,
            "char_span": span,
        }
        for index, (title, text, span) in enumerate(
            zip(("Garden", "Polonium", "Cooking"), parts, starts)
        )
    ]
    source = {"id": "retrieval-test", "question": question}
    plan, sentence_spans, gold_ids = make_plan(
        "bm25_topk",
        source,
        prompt,
        question,
        sentences,
        _character_offsets(prompt),
        budget=0.30,
    )
    retained = set(plan.retained_indices)
    relevant_tokens = set(range(*sentence_spans[1]))
    assert gold_ids == [1]
    assert relevant_tokens <= retained


def test_all_nonlearned_retrieval_baselines_obey_the_same_retained_budget() -> None:
    parts = [
        "Alpha links to Beta. " * 5,
        "Beta links to Gamma. " * 5,
        "Unrelated weather report. " * 8,
    ]
    question = "How does Alpha reach Gamma?"
    prompt = "\n".join(parts) + f"\nQuestion: {question}"
    starts = []
    cursor = 0
    for part in parts:
        starts.append((cursor, cursor + len(part)))
        cursor += len(part) + 1
    sentences = [
        {
            "title": f"Document {index}",
            "text": text,
            "sent_id": index,
            "gold": index < 2,
            "char_span": span,
        }
        for index, (text, span) in enumerate(zip(parts, starts))
    ]
    source = {"id": "budget-test", "question": question}
    query_start = len(prompt) - len(question)
    _, recent_budget, guidepost_budget = adaptive_budget_parts(
        len(prompt), 0.25, query_start
    )
    maximum_retained = min(
        len(prompt), SINK + recent_budget + guidepost_budget
    )
    for method in (
        "bm25_topk",
        "dense_hash_topk",
        "reranker_topk",
        "similarity_graph_topk",
    ):
        plan, _, _ = make_plan(
            method,
            source,
            prompt,
            question,
            sentences,
            _character_offsets(prompt),
            budget=0.25,
        )
        assert len(plan.retained_indices) <= maximum_retained


def test_frontier_and_or_variants_are_registered_as_learned_methods() -> None:
    assert "learned_frontier_and_or" in METHODS
    assert "learned_frontier_and_or_aux" in METHODS
