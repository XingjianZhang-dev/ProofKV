"""Iterative retrieval: multi-hop dense retrieval with ProofKV's own scorer as the encoder.

Retriever score = the frozen scorer's root-relevance head P(root | query, span) — the same
query-conditioned relevance signal ProofKV uses for its roots, so representation quality is
matched. Hop 1 scores every span against the question; the top-k new spans are appended to
the query (each truncated to `per_span_cap` encoder tokens; the appended part is capped at
`total_cap` so the candidate span always keeps room inside the encoder's 512-token window);
the expanded query re-scores all spans; up to `max_hops` hops. The union of retrieved spans is
ordered by the score at which each was retrieved (spec: "union in score order"); the hop index
is recorded so hop-order can be analysed post hoc. No typed relations, no edge scoring, no
completeness objective, no beam search.
"""
from __future__ import annotations
from typing import Any, Sequence
import torch
from proofkv.proofcache.model import score_roots


def _truncate(tokenizer: Any, text: str, cap: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= cap:
        return text
    return tokenizer.decode(ids[:cap])


@torch.inference_mode()
def iterative_retrieval_priority(
    scorer: Any, scorer_tokenizer: Any, question: str, spans: Sequence[Any],
    k: int = 2, max_hops: int = 4, per_span_cap: int = 64, total_cap: int = 256,
) -> tuple[list[str], dict[str, Any]]:
    texts = [s.text for s in spans]
    ids = [s.span_id for s in spans]
    retrieved: dict[str, tuple[float, int]] = {}
    appended: list[str] = []
    query = question
    hops: list[dict[str, Any]] = []
    for hop in range(1, max_hops + 1):
        scores = score_roots(scorer, scorer_tokenizer, query, texts)
        ranked = sorted(zip(scores, ids), key=lambda t: (-t[0], t[1]))
        new = [(s, i) for s, i in ranked if i not in retrieved][:k]
        hops.append({"hop": hop, "query_tokens": len(scorer_tokenizer(query, add_special_tokens=False)["input_ids"]),
                     "retrieved": [(i, float(s)) for s, i in new], "top5": [(i, float(s)) for s, i in ranked[:5]]})
        if not new:
            break
        for s, i in new:
            retrieved[i] = (float(s), hop)
            appended.append(_truncate(scorer_tokenizer, texts[ids.index(i)], per_span_cap))
        expansion = _truncate(scorer_tokenizer, " ".join(appended), total_cap)
        query = f"{question} {expansion}"
    score_order = sorted(retrieved, key=lambda i: (-retrieved[i][0], retrieved[i][1], i))
    hop_order = sorted(retrieved, key=lambda i: (retrieved[i][1], -retrieved[i][0], i))
    return score_order, {"k": k, "max_hops": max_hops, "per_span_cap": per_span_cap, "total_cap": total_cap,
                         "hops": hops, "score_order": score_order, "hop_order": hop_order,
                         "retrieved_scores": {i: retrieved[i][0] for i in retrieved}, "retrieved_hops": {i: retrieved[i][1] for i in retrieved}}
