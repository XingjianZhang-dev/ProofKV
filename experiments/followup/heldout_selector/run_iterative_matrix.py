#!/usr/bin/env python
"""Driver: inject ITERATIVE_RETRIEVAL methods into the frozen matrix script scripts/run_hotpot_compact_qa.py
and run its main() unchanged, so prompt path, budget accounting (adaptive_budget_parts), span pool,
_plan_from_spans compiler, compact-prefill decoder, results schema and run_manifest are exactly ProofKV's.

New method names: learned_iterative_retrieval_k{K}_h{H}. They take the learned_bundle (frozen scorer) like
every other learned_* method, compute the retrieval priority with the scorer's root head, and then follow the
same code path as learned_root_topk: select_by_priority(priority, by_id, adaptive_gp) -> _plan_from_spans.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
IMPL = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(IMPL / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_hotpot_compact_qa as rq
from iterative_retrieval import iterative_retrieval_priority

PATTERN = re.compile(r"^learned_iterative_retrieval_k(\d+)_h(\d+)(_hoporder)?(_tokenfill)?$")
ROOT_TOKENFILL = "learned_root_topk_tokenfill"
_original_make_plan = rq.make_plan


def make_plan(method, source, prompt, query, sentences, offsets, budget, learned_bundle=None, terminal_diversity_weight=0.05,
              partial_proof_weight=0.25, repair_halo=0, cached_edge_non_none_floor=None, learned_dense_core_count=16,
              learned_edge_batch_size=64, learned_overlay_cache=None, selection_metadata_out=None, include_selection_indices=False):
    m = PATTERN.match(method)
    if not m and method != ROOT_TOKENFILL:
        return _original_make_plan(method, source, prompt, query, sentences, offsets, budget, learned_bundle, terminal_diversity_weight,
                                   partial_proof_weight, repair_halo, cached_edge_non_none_floor, learned_dense_core_count,
                                   learned_edge_batch_size, learned_overlay_cache, selection_metadata_out, include_selection_indices)
    k, hops = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    tokenfill = (method == ROOT_TOKENFILL) or bool(m and m.group(4))
    if learned_bundle is None:
        raise ValueError(f"{method} requires --scorer-checkpoint")
    scorer_kind, scorer, scorer_tokenizer = learned_bundle
    if scorer_kind != "cross":
        raise ValueError("iterative retrieval uses the cross-encoder root head")
    seq_len = len(offsets)
    sentence_spans = [rq.token_span(offsets, *s["char_span"]) for s in sentences]
    gold_ids = [i for i, s in enumerate(sentences) if s["gold"]]
    query_start, _ = rq.token_span(offsets, len(prompt) - len(query), len(prompt))
    _, adaptive_recent, adaptive_gp = rq.adaptive_budget_parts(seq_len, budget, query_start)
    evidence_spans = [
        rq.EvidenceSpan(f"s{i}", a, b, f"[{s['title']}] {s['text']}", cues=rq.extract_cues(f"{s['title']} {s['text']}"),
                        embedding=rq.hashed_embedding(f"{s['title']} {s['text']}", 64), metadata={"title": s["title"], "sentence_index": s["sent_id"]})
        for i, ((a, b), s) in enumerate(zip(sentence_spans, sentences))]
    index = rq.EvidenceIndexer().build(str(source["id"]), evidence_spans, seq_len)
    if method == ROOT_TOKENFILL:
        from proofkv.proofcache.model import score_roots
        with __import__("torch").inference_mode():
            root = score_roots(scorer, scorer_tokenizer, str(source["question"]), [sp.text for sp in index.spans])
        priority = [sid for _, sid in sorted(zip(root, [sp.span_id for sp in index.spans]), key=lambda t: (-t[0], t[1]))]
        trace = {"retrieved_hops": {}, "retrieved_scores": {sid: float(sc) for sc, sid in zip(root, [sp.span_id for sp in index.spans])}}
    else:
        priority, trace = iterative_retrieval_priority(scorer, scorer_tokenizer, str(source["question"]), index.spans, k=k, max_hops=hops)
        if m.group(3):
            priority = list(trace["hop_order"])
    # span-atomic fill (select_by_priority, as learned_root_topk in the frozen script) or token-level fill
    # (all spans in priority order, _plan_from_spans truncates at the exact budget, as learned_topk_paths does)
    selected_ids = list(priority) if tokenfill else list(rq.select_by_priority(priority, index.by_id, adaptive_gp))
    selected_spans = [(max(0, index.by_id[i].token_start - repair_halo), min(seq_len, index.by_id[i].token_end + repair_halo)) for i in selected_ids]
    plan = rq._plan_from_spans(seq_len, selected_spans, adaptive_gp, rq.SINK, adaptive_recent)
    if selection_metadata_out is not None:
        selection_metadata_out.update({"retrieval_priority_span_ids": priority, "retrieval_selected_span_ids": selected_ids,
                                       "retrieval_selected_span_count": len(selected_ids), "retrieval_hops": trace["retrieved_hops"],
                                       "retrieval_scores": trace["retrieved_scores"], "retrieval_k": k, "retrieval_max_hops": hops, "token_level_fill": tokenfill})
    return plan, sentence_spans, gold_ids


rq.make_plan = make_plan
for k in (1, 2, 4):
    for h in (1, 2, 3, 4):
        rq.METHODS.add(f"learned_iterative_retrieval_k{k}_h{h}")
        rq.METHODS.add(f"learned_iterative_retrieval_k{k}_h{h}_hoporder")
        rq.METHODS.add(f"learned_iterative_retrieval_k{k}_h{h}_tokenfill")
        rq.METHODS.add(f"learned_iterative_retrieval_k{k}_h{h}_hoporder_tokenfill")
rq.METHODS.add(ROOT_TOKENFILL)

if __name__ == "__main__":
    rq.main()
