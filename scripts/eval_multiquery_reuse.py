#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoTokenizer

try:
    import run_physical_kv_replay as physical
except ImportError:
    from scripts import run_physical_kv_replay as physical

from proofkv.eval.metrics import contains_answer
from proofkv.proofcache.datasets import EvidenceExampleSpan
from proofkv.proofcache.index import EvidenceIndexer, extract_cues, hashed_embedding
from proofkv.proofcache.model import (
    CachedTypedRelationScorer,
    TypedRelationScorer,
    build_cached_overlay,
    build_learned_overlay,
    encode_cached_spans,
)
from proofkv.proofcache.schema import EvidenceSpan, PrecisionTier, ProofOverlay
from proofkv.proofcache.selector import AndOrProofSelector, ProofSearchConfig, TypedProofSearcher
from proofkv.proofcache.topology import TopologyConfig, make_topology_example
from proofkv.utils.io import append_jsonl


@dataclass(frozen=True)
class QueryRecord:
    question: str
    answer: str
    gold_proofs: tuple[frozenset[str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence Index + Proof Overlay multi-query reuse experiment.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--scorer-checkpoint")
    parser.add_argument("--cached-scorer-checkpoint")
    parser.add_argument("--query-counts", default="1,2,4,8,16")
    parser.add_argument("--context-length", type=int, default=8000)
    parser.add_argument("--budget-ratio", type=float, default=0.08)
    parser.add_argument("--selector", choices=("and_or", "topk"), default="and_or")
    parser.add_argument("--candidate-strategy", choices=("root", "frontier"), default="frontier")
    parser.add_argument("--cached-edge-non-none-floor", type=float, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--physical", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--correlation-aware", action="store_true")
    parser.add_argument("--terminal-diversity-weight", type=float, default=0.05)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def token_span(offsets: Sequence[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    hits = [idx for idx, (start, end) in enumerate(offsets) if end > char_start and start < char_end and end > start]
    return (min(hits), max(hits) + 1) if hits else (0, 0)


def make_shared_document(max_queries: int, seed: int) -> tuple[list[EvidenceExampleSpan], list[QueryRecord]]:
    spans: list[EvidenceExampleSpan] = []
    queries: list[QueryRecord] = []
    for query_idx in range(max_queries):
        example = make_topology_example(
            TopologyConfig(
                hops=2 + query_idx % 3,
                branch_factor=2,
                alternative_proofs=1 + int(query_idx % 4 == 0),
                distractor_chains=1,
                paraphrase=bool(query_idx % 2),
                fact_reorder=True,
            ),
            seed=seed,
            sample_index=query_idx,
        )
        mapping = {span.span_id: f"q{query_idx}_{span.span_id}" for span in example.spans}
        for span in example.spans:
            spans.append(
                EvidenceExampleSpan(
                    mapping[span.span_id], span.title, span.text, span.kind,
                    {**span.metadata, "query_owner": query_idx},
                )
            )
        queries.append(
            QueryRecord(
                example.question,
                example.answer,
                tuple(frozenset(mapping[span_id] for span_id in proof) for proof in example.gold_proofs),
            )
        )
    random.Random(seed).shuffle(spans)
    return spans, queries


def render_context(spans: Sequence[EvidenceExampleSpan]) -> tuple[str, list[tuple[int, int]]]:
    pieces = []
    char_spans = []
    cursor = 0
    for span in spans:
        line = f"[{span.title}] {span.text}\n"
        pieces.append(line)
        char_spans.append((cursor, cursor + len(line)))
        cursor += len(line)
    return "".join(pieces), char_spans


def main() -> None:
    args = parse_args()
    if args.scorer_checkpoint and args.cached_scorer_checkpoint:
        raise ValueError("provide at most one scorer checkpoint")
    query_counts = [int(value) for value in args.query_counts.split(",")]
    max_queries = max(query_counts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "queries.jsonl"
    if results_path.exists() or (out_dir / "metrics.json").exists():
        raise FileExistsError(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    source_spans, queries = make_shared_document(max_queries, args.seed)
    # Add unrelated filler until the requested context scale is reached.
    filler_idx = 0
    while True:
        context, char_spans = render_context(source_spans)
        current_tokens = len(tokenizer(context, add_special_tokens=False)["input_ids"])
        if current_tokens >= int(0.95 * args.context_length):
            break
        source_spans.append(
            EvidenceExampleSpan(
                f"global_filler_{filler_idx}",
                "Unrelated archive",
                f"Archive entry {filler_idx} contains unrelated value {10_000 + filler_idx} and no route evidence.",
                source_spans[0].kind,
                {"role": "global_filler"},
            )
        )
        filler_idx += 1
    context, char_spans = render_context(source_spans)
    context_encoding = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
    context_offsets = [(int(start), int(end)) for start, end in context_encoding["offset_mapping"]]
    evidence_spans = []
    for source, chars in zip(source_spans, char_spans):
        start, end = token_span(context_offsets, *chars)
        text = f"[{source.title}] {source.text}"
        evidence_spans.append(
            EvidenceSpan(
                source.span_id, start, end, text, source.kind,
                extract_cues(text), hashed_embedding(text, 64),
                {**source.metadata, "title": source.title},
            )
        )
    index_start = time.perf_counter()
    index = EvidenceIndexer().build("shared-multiquery-context", evidence_spans, len(context_offsets))
    index_ms = (time.perf_counter() - index_start) * 1000
    by_id = index.by_id

    learned_model = learned_tokenizer = None
    cached_model = cached_tokenizer = span_embeddings = None
    if args.scorer_checkpoint:
        learned_model, learned_tokenizer = TypedRelationScorer.load_checkpoint(
            args.scorer_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"
        )
    if args.cached_scorer_checkpoint:
        cached_model, cached_tokenizer = CachedTypedRelationScorer.load_checkpoint(
            args.cached_scorer_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        offline_started = time.perf_counter()
        span_embeddings = encode_cached_spans(
            cached_model, cached_tokenizer, [span.text for span in index.spans]
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cached_embedding_ms = (time.perf_counter() - offline_started) * 1000
    else:
        cached_embedding_ms = 0.0
    llm = None
    if args.physical:
        llm, _ = physical.load_model_tokenizer(
            args.model, dtype="bfloat16", attn_implementation="sdpa", device_map="auto"
        )
    selector = AndOrProofSelector(
        correlation_aware=args.correlation_aware,
        terminal_diversity_weight=(
            args.terminal_diversity_weight if args.correlation_aware else 0.0
        ),
    )
    heuristic_searcher = TypedProofSearcher()
    learned_search_config = ProofSearchConfig(
        root_top_k=32,
        beam_size=48,
        max_hops=8,
        max_proofs=16,
        min_spans_per_proof=2,
        terminal_probability_floor=0.20,
    )
    kv_values_per_token = 28 * 2 * 4 * 128
    kv_bytes_per_token = kv_values_per_token * 2
    union_context_tokens: set[int] = set()
    previous_route_tokens: set[int] | None = None
    query_rows: list[dict[str, Any]] = []

    for query_idx, query_record in enumerate(queries):
        if torch.cuda.is_available() and (cached_model is not None or learned_model is not None):
            torch.cuda.synchronize()
        overlay_start = time.perf_counter()
        if cached_model is not None:
            overlay, learned_index = build_cached_overlay(
                cached_model,
                cached_tokenizer,
                query_record.question,
                index,
                span_embeddings,
                candidate_span_count=64,
                search_config=learned_search_config,
                candidate_strategy=args.candidate_strategy,
                edge_non_none_floor=args.cached_edge_non_none_floor,
            )
            selection_index = learned_index
        elif learned_model is not None:
            overlay, learned_index = build_learned_overlay(
                learned_model,
                learned_tokenizer,
                query_record.question,
                index,
                candidate_span_count=64,
                search_config=learned_search_config,
                candidate_strategy=args.candidate_strategy,
            )
            selection_index = learned_index
        else:
            overlay = heuristic_searcher.search(query_record.question, index)
            selection_index = index
        evidence_budget = int(round(len(context_offsets) * args.budget_ratio))
        if args.selector == "topk":
            selected_ids = []
            used_tokens = 0
            for proof in overlay.proofs:
                for span_id in proof.span_ids:
                    if span_id in selected_ids:
                        continue
                    cost = selection_index.by_id[span_id].token_count
                    if used_tokens + cost <= evidence_budget:
                        selected_ids.append(span_id)
                        used_tokens += cost
            selected_overlay = ProofOverlay(
                query=overlay.query,
                proofs=overlay.proofs,
                root_scores=overlay.root_scores,
                confidence=overlay.confidence,
                terminal_scores=overlay.terminal_scores,
                selected_proof_ids=tuple(
                    proof.proof_id
                    for proof in overlay.proofs
                    if set(proof.span_ids) <= set(selected_ids)
                ),
                selected_span_ids=tuple(selected_ids),
            )
        else:
            selected_overlay = selector.select(
                overlay,
                selection_index,
                byte_budget=evidence_budget * kv_bytes_per_token,
                values_per_token=kv_values_per_token,
                precision=PrecisionTier.BF16,
            )
        if torch.cuda.is_available() and (cached_model is not None or learned_model is not None):
            torch.cuda.synchronize()
        overlay_ms = (time.perf_counter() - overlay_start) * 1000
        selected_context_tokens = {
            token
            for span_id in selected_overlay.selected_span_ids
            for token in range(by_id[span_id].token_start, by_id[span_id].token_end)
        }
        new_context_tokens = selected_context_tokens - union_context_tokens
        union_context_tokens |= selected_context_tokens
        overlap = (
            len(selected_context_tokens & previous_route_tokens) / max(1, len(selected_context_tokens | previous_route_tokens))
            if previous_route_tokens is not None else 0.0
        )
        previous_route_tokens = selected_context_tokens
        selected_ids = set(selected_overlay.selected_span_ids)
        pcr = float(any(proof <= selected_ids for proof in query_record.gold_proofs))

        full_seconds = route_seconds = None
        full_correctness = route_correctness = None
        prompt_tokens = len(context_offsets)
        if llm is not None:
            query_text = f"\nQuestion: {query_record.question}\nAnswer:"
            prompt = context + query_text
            encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to(next(llm.parameters()).device)
            prompt_tokens = int(input_ids.shape[1])
            query_start = len(context_offsets)
            retained = sorted(
                set(range(min(4, prompt_tokens)))
                | selected_context_tokens
                | set(range(query_start, prompt_tokens))
            )
            started = time.perf_counter()
            full_text, full_cache, _ = physical._greedy_full_decode(
                llm, tokenizer, input_ids, args.max_new_tokens
            )
            full_seconds = time.perf_counter() - started
            del full_cache
            physical._release_cuda()
            started = time.perf_counter()
            route_text, route_cache, _, _, _ = physical._compact_prefill_decode(
                llm, tokenizer, input_ids, retained, prompt_tokens, args.max_new_tokens
            )
            route_seconds = time.perf_counter() - started
            del route_cache
            physical._release_cuda()
            full_correctness = contains_answer(full_text, query_record.answer)
            route_correctness = contains_answer(route_text, query_record.answer)

        row = {
            "query_index": query_idx + 1,
            "question": query_record.question,
            "answer": query_record.answer,
            "pcr": pcr,
            "overlay_confidence": selected_overlay.confidence,
            "overlay_ms": overlay_ms,
            "route_context_tokens": len(selected_context_tokens),
            "incremental_context_tokens": len(new_context_tokens),
            "union_context_tokens": len(union_context_tokens),
            "union_context_ratio": len(union_context_tokens) / len(context_offsets),
            "previous_overlay_jaccard": overlap,
            "prompt_tokens": prompt_tokens,
            "full_seconds": full_seconds,
            "route_seconds": route_seconds,
            "full_correctness": full_correctness,
            "route_correctness": route_correctness,
        }
        query_rows.append(row)
        append_jsonl(results_path, row)

    records = []
    for count in query_counts:
        values = query_rows[:count]
        records.append(
            {
                "queries": count,
                "index_build_ms": index_ms,
                "offline_embedding_ms": cached_embedding_ms,
                "amortized_offline_index_ms": (index_ms + cached_embedding_ms) / count,
                "pcr": sum(value["pcr"] for value in values) / count,
                "mean_overlay_ms": sum(value["overlay_ms"] for value in values) / count,
                "mean_pairwise_overlay_jaccard": sum(value["previous_overlay_jaccard"] for value in values[1:]) / max(1, count - 1),
                "union_context_ratio": values[-1]["union_context_ratio"],
                "mean_incremental_context_tokens": sum(value["incremental_context_tokens"] for value in values) / count,
                "amortized_full_seconds": None if not args.physical else sum(value["full_seconds"] for value in values) / count,
                "amortized_route_seconds": None if not args.physical else sum(value["route_seconds"] for value in values) / count,
                "full_accuracy": None if not args.physical else sum(value["full_correctness"] for value in values) / count,
                "route_accuracy": None if not args.physical else sum(value["route_correctness"] for value in values) / count,
            }
        )
    payload = {
        "context_tokens": len(context_offsets),
        "queries": max_queries,
        "budget_ratio": args.budget_ratio,
        "index_build_ms": index_ms,
        "offline_embedding_ms": cached_embedding_ms,
        "learned_scorer": bool(args.scorer_checkpoint),
        "cached_scorer": bool(args.cached_scorer_checkpoint),
        "correlation_aware": args.correlation_aware,
        "selector": args.selector,
        "candidate_strategy": args.candidate_strategy,
        "physical": args.physical,
        "records": records,
        "notes": {"kvzip": "formal KVzip comparison requires its official implementation and is not claimed by this run"},
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "records": records}))


if __name__ == "__main__":
    main()
