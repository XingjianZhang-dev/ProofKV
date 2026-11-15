#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import load_dataset
from transformers import AutoTokenizer

from proofkv.prefill_selectors import compass_prefill_guideposts
from proofkv.proofcache.datasets import EvidenceExample, from_2wiki, from_hotpot, from_musique
from proofkv.proofcache.index import EvidenceIndexer, extract_cues, hashed_embedding, lexical_terms
from proofkv.proofcache.schema import EvidenceSpan, PrecisionTier, ProofOverlay
from proofkv.proofcache.selector import (
    AnytimeProofSelector,
    AndOrProofSelector,
    ProofSearchConfig,
    TypedProofSearcher,
)
from proofkv.utils.io import append_jsonl


SINK = 4
TWO_WIKI_TRAIN = "https://huggingface.co/datasets/somebody-had-to-do-it/2WikiMultihopQA/resolve/main/train.parquet"
METHODS = {
    "bm25",
    "dense_hash_proxy",
    "reranker_proxy",
    "graph_similarity_proxy",
    "v1_compass",
    "typed_single_path",
    "typed_topk_paths",
    "and_or_proof",
    "oracle_proof",
    "learned_single_path",
    "learned_topk_paths",
    "learned_and_or",
    "learned_and_or_aux",
    "learned_and_or_diverse",
    "learned_and_or_diverse_aux",
    "learned_anytime",
    "learned_anytime_aux",
    "cached_single_path",
    "cached_topk_paths",
    "cached_and_or",
    "cached_and_or_aux",
    "cached_and_or_diverse",
    "cached_and_or_diverse_aux",
    "cached_anytime",
    "cached_anytime_aux",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified selector-only evidence/proof evaluator.")
    parser.add_argument("--dataset", choices=["hotpotqa", "2wiki", "musique"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-file")
    parser.add_argument("--unlock-locked-test", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--budgets", default="128,256,512,1024,0.02,0.04,0.08,0.16")
    parser.add_argument("--methods", default=",".join(sorted(METHODS)))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--scorer-checkpoint")
    parser.add_argument("--cached-scorer-checkpoint")
    parser.add_argument("--scorer-device", default="cuda")
    parser.add_argument("--learned-candidate-count", type=int, default=32)
    parser.add_argument("--learned-candidate-strategy", choices=("root", "frontier"), default="root")
    parser.add_argument("--learned-max-hops", type=int, default=3)
    parser.add_argument("--learned-beam-size", type=int, default=8)
    parser.add_argument("--learned-max-proofs", type=int, default=8)
    parser.add_argument("--learned-dense-core-count", type=int, default=16)
    parser.add_argument("--learned-edge-batch-size", type=int, default=64)
    parser.add_argument("--terminal-diversity-weight", type=float, default=0.05)
    parser.add_argument("--partial-proof-weight", type=float, default=0.25)
    parser.add_argument("--cached-relation-prior-weight", type=float, default=0.0)
    parser.add_argument("--cached-edge-non-none-floor", type=float, default=None)
    parser.add_argument(
        "--learned-normalize-path-probability",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def token_span(offsets: Sequence[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    hits = [idx for idx, (start, end) in enumerate(offsets) if end > char_start and start < char_end and end > start]
    return (min(hits), max(hits) + 1) if hits else (0, 0)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def load_examples(args: argparse.Namespace) -> list[EvidenceExample]:
    if args.dataset == "hotpotqa":
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
        rows = [dataset[idx] for idx in range(len(dataset))]
        adapter = from_hotpot
    elif args.dataset == "2wiki":
        if args.split != "train":
            raise ValueError("v1 freeze permits only 2Wiki train until a calibration partition is materialized")
        dataset = load_dataset("parquet", data_files={"train": TWO_WIKI_TRAIN}, split="train")
        rows = [dataset[idx] for idx in range(len(dataset))]
        adapter = from_2wiki
    else:
        if not args.unlock_locked_test:
            raise ValueError("MuSiQue is locked; pass --unlock-locked-test only after selector freeze and gate approval")
        if not args.data_file:
            raise ValueError("MuSiQue requires a frozen local JSONL/Parquet data file")
        suffix = Path(args.data_file).suffix.lower()
        kind = "parquet" if suffix == ".parquet" else "json"
        dataset = load_dataset(kind, data_files={args.split: args.data_file}, split=args.split)
        rows = [dataset[idx] for idx in range(len(dataset))]
        adapter = from_musique
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    return [adapter(rows[idx], args.split) for idx in order[: args.max_samples]]


def parse_budget(spec: str, seq_len: int) -> tuple[str, int]:
    value = float(spec)
    if value < 1.0:
        return f"{value:.4f}", max(SINK + 1, int(round(seq_len * value)))
    return str(int(value)), max(SINK + 1, int(value))


def select_by_priority(priority: Iterable[str], by_id: dict[str, EvidenceSpan], budget: int) -> tuple[str, ...]:
    selected: list[str] = []
    used_tokens: set[int] = set()
    for span_id in priority:
        tokens = set(range(by_id[span_id].token_start, by_id[span_id].token_end))
        if len(used_tokens | tokens) > budget:
            continue
        selected.append(span_id)
        used_tokens |= tokens
    return tuple(selected)


def bm25_priority(question: str, spans: Sequence[EvidenceSpan]) -> list[str]:
    query = lexical_terms(question)
    df = Counter(term for span in spans for term in set(span.cues.get("terms", ())))
    average_length = sum(len(span.cues.get("terms", ())) for span in spans) / max(1, len(spans))
    rows = []
    for span in spans:
        terms = list(span.cues.get("terms", ()))
        tf = Counter(terms)
        length_norm = 1 - 0.75 + 0.75 * len(terms) / max(1.0, average_length)
        score = 0.0
        for term in query:
            idf = math.log(1 + (len(spans) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (tf[term] * 2.5) / (tf[term] + 1.5 * length_norm) if tf[term] else 0.0
        rows.append((score, span.span_id))
    return [span_id for _, span_id in sorted(rows, key=lambda row: (-row[0], row[1]))]


def dense_priority(question: str, spans: Sequence[EvidenceSpan]) -> list[str]:
    query_embedding = hashed_embedding(question, len(spans[0].embedding) if spans else 64)
    return [
        span_id
        for _, span_id in sorted(
            ((cosine(query_embedding, span.embedding), span.span_id) for span in spans),
            key=lambda row: (-row[0], row[1]),
        )
    ]


def reranker_priority(question: str, spans: Sequence[EvidenceSpan]) -> list[str]:
    bm25_rank = {span_id: idx for idx, span_id in enumerate(bm25_priority(question, spans))}
    dense_rank = {span_id: idx for idx, span_id in enumerate(dense_priority(question, spans))}
    question_terms = set(lexical_terms(question))
    rows = []
    for span in spans:
        title = str(span.metadata.get("title", ""))
        title_overlap = len(question_terms & set(lexical_terms(title)))
        reciprocal = 1 / (60 + bm25_rank[span.span_id]) + 1 / (60 + dense_rank[span.span_id])
        rows.append((reciprocal + 0.01 * title_overlap, span.span_id))
    return [span_id for _, span_id in sorted(rows, key=lambda row: (-row[0], row[1]))]


def graph_similarity_priority(question: str, spans: Sequence[EvidenceSpan]) -> list[str]:
    query_embedding = hashed_embedding(question, len(spans[0].embedding) if spans else 64)
    relevance = {span.span_id: max(0.0, cosine(query_embedding, span.embedding)) for span in spans}
    scores = dict(relevance)
    for _ in range(5):
        updated = {}
        for span in spans:
            neighbors = sorted(
                (
                    (max(0.0, cosine(span.embedding, other.embedding)), other.span_id)
                    for other in spans
                    if other.span_id != span.span_id
                ),
                reverse=True,
            )[:4]
            propagated = sum(similarity * scores[other_id] for similarity, other_id in neighbors)
            normalizer = sum(similarity for similarity, _ in neighbors) or 1.0
            updated[span.span_id] = relevance[span.span_id] + 0.35 * propagated / normalizer
        scores = updated
    return sorted(scores, key=lambda span_id: (-scores[span_id], span_id))


def proof_priority(overlay: ProofOverlay) -> list[str]:
    ordered: list[str] = []
    for proof in overlay.proofs:
        for span_id in proof.span_ids:
            if span_id not in ordered:
                ordered.append(span_id)
    return ordered


def selector_metrics(
    example: EvidenceExample,
    selected_span_ids: Sequence[str],
    by_id: dict[str, EvidenceSpan],
    retained_tokens: set[int],
    total_tokens: int,
    kv_bytes_per_token: int,
) -> dict[str, float]:
    selected = set(selected_span_ids)
    complete_proofs = sum(proof <= selected for proof in example.gold_proofs)
    pcr = float(complete_proofs > 0)
    gold_ids = example.gold_span_ids
    selected_evidence_tokens = {
        token for span_id in selected for token in range(by_id[span_id].token_start, by_id[span_id].token_end)
    }
    gold_tokens = {
        token for span_id in gold_ids for token in range(by_id[span_id].token_start, by_id[span_id].token_end)
    }
    true_tokens = selected_evidence_tokens & gold_tokens
    actual_bytes = len(retained_tokens) * kv_bytes_per_token
    return {
        "pcr": pcr,
        "complete_proofs": float(complete_proofs),
        "proof_efficiency_per_mib": complete_proofs / max(actual_bytes / (1024**2), 1e-12),
        "route_precision": len(true_tokens) / max(1, len(selected_evidence_tokens)),
        "gold_token_recall": len(true_tokens) / max(1, len(gold_tokens)),
        "retained_ratio": len(retained_tokens) / max(1, total_tokens),
        "actual_kv_bytes": float(actual_bytes),
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["budget_spec"])].append(row)
    records = []
    metric_names = [
        "pcr", "core_pcr", "complete_proofs", "proof_efficiency_per_mib", "route_precision",
        "gold_token_recall", "retained_ratio", "actual_kv_bytes", "selector_ms", "index_build_ms",
    ]
    for (method, budget), values in sorted(groups.items()):
        record: dict[str, Any] = {"method": method, "budget_spec": budget, "n": len(values)}
        for metric in metric_names:
            record[metric] = sum(float(value[metric]) for value in values) / len(values)
        ordered_latency = sorted(float(value["selector_ms"]) for value in values)
        record["selector_p95_ms"] = ordered_latency[min(len(ordered_latency) - 1, math.ceil(0.95 * len(ordered_latency)) - 1)]
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = set(methods) - METHODS
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    budget_specs = [value.strip() for value in args.budgets.split(",") if value.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    if results_path.exists() or (out_dir / "metrics.json").exists():
        raise FileExistsError(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    examples = load_examples(args)
    indexer = EvidenceIndexer()
    learned_search_config = ProofSearchConfig(
        root_top_k=16,
        beam_size=args.learned_beam_size,
        max_hops=args.learned_max_hops,
        max_proofs=args.learned_max_proofs,
        normalize_path_probability=args.learned_normalize_path_probability,
    )
    searcher = TypedProofSearcher(learned_search_config)
    learned_model = learned_tokenizer = None
    if any(method.startswith("learned_") for method in methods):
        if not args.scorer_checkpoint:
            raise ValueError("learned methods require --scorer-checkpoint")
        from proofkv.proofcache.model import TypedRelationScorer

        learned_model, learned_tokenizer = TypedRelationScorer.load_checkpoint(
            args.scorer_checkpoint, device=args.scorer_device
        )
    cached_model = cached_tokenizer = None
    if any(method.startswith("cached_") for method in methods):
        if not args.cached_scorer_checkpoint:
            raise ValueError("cached methods require --cached-scorer-checkpoint")
        from proofkv.proofcache.model import CachedTypedRelationScorer

        cached_model, cached_tokenizer = CachedTypedRelationScorer.load_checkpoint(
            args.cached_scorer_checkpoint, device=args.scorer_device
        )
    kv_bytes_per_token = args.num_layers * 2 * args.num_kv_heads * args.head_dim * 2
    rows: list[dict[str, Any]] = []

    for example in examples:
        context, char_spans = example.render_context()
        query = f"\nQuestion: {example.question}\nAnswer:"
        prompt = context + query
        encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [(int(start), int(end)) for start, end in encoding["offset_mapping"]]
        token_spans = [token_span(offsets, *char_span) for char_span in char_spans]
        evidence_spans = [
            EvidenceSpan(
                span_id=span.span_id,
                token_start=token_start,
                token_end=token_end,
                text=f"[{span.title}] {span.text}",
                kind=span.kind,
                cues=extract_cues(f"{span.title} {span.text}"),
                embedding=hashed_embedding(f"{span.title} {span.text}", 64),
                metadata={**span.metadata, "title": span.title},
            )
            for span, (token_start, token_end) in zip(example.spans, token_spans)
        ]
        index = indexer.build(example.sample_id, evidence_spans, len(offsets))
        by_id = index.by_id
        query_start, _ = token_span(offsets, len(context), len(prompt))
        query_tokens = set(range(query_start, len(offsets)))
        sink_tokens = set(range(min(SINK, len(offsets))))
        overlay = searcher.search(example.question, index)
        learned_overlay = learned_index = None
        learned_scorer_ms = 0.0
        if learned_model is not None:
            import torch
            from proofkv.proofcache.model import build_learned_overlay

            if torch.cuda.is_available() and args.scorer_device.startswith("cuda"):
                torch.cuda.synchronize()
            learned_start = time.perf_counter()
            learned_overlay, learned_index = build_learned_overlay(
                learned_model, learned_tokenizer, example.question, index,
                candidate_span_count=args.learned_candidate_count,
                search_config=learned_search_config,
                candidate_strategy=args.learned_candidate_strategy,
                dense_core_count=args.learned_dense_core_count,
                edge_batch_size=args.learned_edge_batch_size,
            )
            if torch.cuda.is_available() and args.scorer_device.startswith("cuda"):
                torch.cuda.synchronize()
            learned_scorer_ms = (time.perf_counter() - learned_start) * 1000
        cached_overlay = cached_index = None
        cached_index_ms = cached_overlay_ms = 0.0
        if cached_model is not None:
            import torch
            from proofkv.proofcache.model import build_cached_overlay, encode_cached_spans

            if torch.cuda.is_available() and args.scorer_device.startswith("cuda"):
                torch.cuda.synchronize()
            cached_start = time.perf_counter()
            cached_embeddings = encode_cached_spans(
                cached_model, cached_tokenizer, [span.text for span in index.spans]
            )
            if torch.cuda.is_available() and args.scorer_device.startswith("cuda"):
                torch.cuda.synchronize()
            cached_index_ms = (time.perf_counter() - cached_start) * 1000
            overlay_start = time.perf_counter()
            cached_overlay, cached_index = build_cached_overlay(
                cached_model, cached_tokenizer, example.question, index, cached_embeddings,
                candidate_span_count=args.learned_candidate_count,
                search_config=learned_search_config,
                candidate_strategy=args.learned_candidate_strategy,
                relation_prior_weight=args.cached_relation_prior_weight,
                edge_non_none_floor=args.cached_edge_non_none_floor,
            )
            if torch.cuda.is_available() and args.scorer_device.startswith("cuda"):
                torch.cuda.synchronize()
            cached_overlay_ms = (time.perf_counter() - overlay_start) * 1000

        for budget_input in budget_specs:
            budget_spec, total_budget = parse_budget(budget_input, len(offsets))
            mandatory_tokens = query_tokens | sink_tokens
            evidence_budget = max(0, total_budget - len(mandatory_tokens))
            for method in methods:
                start = time.perf_counter()
                core_selected_ids = None
                method_confidence = None
                if method == "bm25":
                    selected_ids = select_by_priority(bm25_priority(example.question, index.spans), by_id, evidence_budget)
                elif method == "dense_hash_proxy":
                    selected_ids = select_by_priority(dense_priority(example.question, index.spans), by_id, evidence_budget)
                elif method == "reranker_proxy":
                    selected_ids = select_by_priority(reranker_priority(example.question, index.spans), by_id, evidence_budget)
                elif method == "graph_similarity_proxy":
                    selected_ids = select_by_priority(graph_similarity_priority(example.question, index.spans), by_id, evidence_budget)
                elif method == "v1_compass":
                    plan = compass_prefill_guideposts(
                        prompt, query, offsets, evidence_budget, SINK, len(query_tokens),
                        block_size=16, halo=4, propagation_rounds=5,
                    )
                    retained = set(plan.retained_indices)
                    selected_ids = tuple(
                        span.span_id
                        for span in index.spans
                        if set(range(span.token_start, span.token_end)) <= retained
                    )
                elif method == "typed_single_path":
                    compositional = [proof for proof in overlay.proofs if len(proof.span_ids) >= 2]
                    priority = list((compositional or list(overlay.proofs))[0].span_ids) if overlay.proofs else []
                    selected_ids = select_by_priority(priority, by_id, evidence_budget)
                elif method == "typed_topk_paths":
                    selected_ids = select_by_priority(proof_priority(overlay), by_id, evidence_budget)
                elif method == "and_or_proof":
                    selected_overlay = AndOrProofSelector().select(
                        overlay,
                        index,
                        byte_budget=evidence_budget * kv_bytes_per_token,
                        values_per_token=args.num_layers * 2 * args.num_kv_heads * args.head_dim,
                        precision=PrecisionTier.BF16,
                    )
                    selected_ids = selected_overlay.selected_span_ids
                elif method == "oracle_proof":
                    gold_proofs = sorted(example.gold_proofs, key=lambda proof: (sum(by_id[x].token_count for x in proof), sorted(proof)))
                    priority = sorted(gold_proofs[0], key=lambda span_id: by_id[span_id].token_start) if gold_proofs else []
                    selected_ids = select_by_priority(priority, by_id, evidence_budget)
                elif method == "learned_single_path":
                    method_confidence = learned_overlay.confidence
                    compositional = [proof for proof in learned_overlay.proofs if len(proof.span_ids) >= 2]
                    priority = list((compositional or list(learned_overlay.proofs))[0].span_ids) if learned_overlay.proofs else []
                    selected_ids = select_by_priority(priority, by_id, evidence_budget)
                elif method == "learned_topk_paths":
                    method_confidence = learned_overlay.confidence
                    selected_ids = select_by_priority(proof_priority(learned_overlay), by_id, evidence_budget)
                elif method in {
                    "learned_and_or",
                    "learned_and_or_aux",
                    "learned_and_or_diverse",
                    "learned_and_or_diverse_aux",
                }:
                    diverse = "_diverse" in method
                    selected_overlay = AndOrProofSelector(
                        fill_auxiliary=method.endswith("_aux"),
                        correlation_aware=diverse,
                        terminal_diversity_weight=(args.terminal_diversity_weight if diverse else 0.0),
                    ).select(
                        learned_overlay,
                        learned_index,
                        byte_budget=evidence_budget * kv_bytes_per_token,
                        values_per_token=args.num_layers * 2 * args.num_kv_heads * args.head_dim,
                        precision=PrecisionTier.BF16,
                    )
                    selected_ids = selected_overlay.selected_span_ids + selected_overlay.auxiliary_span_ids
                    core_selected_ids = selected_overlay.selected_span_ids
                    method_confidence = selected_overlay.confidence
                elif method in {"learned_anytime", "learned_anytime_aux"}:
                    selected_overlay = AnytimeProofSelector(
                        fill_auxiliary=method.endswith("_aux"),
                        partial_proof_weight=args.partial_proof_weight,
                    ).select(
                        learned_overlay,
                        learned_index,
                        byte_budget=evidence_budget * kv_bytes_per_token,
                        values_per_token=args.num_layers * 2 * args.num_kv_heads * args.head_dim,
                        precision=PrecisionTier.BF16,
                    )
                    selected_ids = selected_overlay.selected_span_ids + selected_overlay.auxiliary_span_ids
                    core_selected_ids = selected_overlay.selected_span_ids
                    method_confidence = selected_overlay.confidence
                elif method == "cached_single_path":
                    method_confidence = cached_overlay.confidence
                    compositional = [proof for proof in cached_overlay.proofs if len(proof.span_ids) >= 2]
                    priority = list((compositional or list(cached_overlay.proofs))[0].span_ids) if cached_overlay.proofs else []
                    selected_ids = select_by_priority(priority, by_id, evidence_budget)
                elif method == "cached_topk_paths":
                    method_confidence = cached_overlay.confidence
                    selected_ids = select_by_priority(proof_priority(cached_overlay), by_id, evidence_budget)
                elif method in {
                    "cached_and_or",
                    "cached_and_or_aux",
                    "cached_and_or_diverse",
                    "cached_and_or_diverse_aux",
                }:
                    diverse = "_diverse" in method
                    selected_overlay = AndOrProofSelector(
                        fill_auxiliary=method.endswith("_aux"),
                        correlation_aware=diverse,
                        terminal_diversity_weight=(args.terminal_diversity_weight if diverse else 0.0),
                    ).select(
                        cached_overlay,
                        cached_index,
                        byte_budget=evidence_budget * kv_bytes_per_token,
                        values_per_token=args.num_layers * 2 * args.num_kv_heads * args.head_dim,
                        precision=PrecisionTier.BF16,
                    )
                    core_selected_ids = selected_overlay.selected_span_ids
                    selected_ids = selected_overlay.selected_span_ids + selected_overlay.auxiliary_span_ids
                    method_confidence = selected_overlay.confidence
                elif method in {"cached_anytime", "cached_anytime_aux"}:
                    selected_overlay = AnytimeProofSelector(
                        fill_auxiliary=method.endswith("_aux"),
                        partial_proof_weight=args.partial_proof_weight,
                    ).select(
                        cached_overlay,
                        cached_index,
                        byte_budget=evidence_budget * kv_bytes_per_token,
                        values_per_token=args.num_layers * 2 * args.num_kv_heads * args.head_dim,
                        precision=PrecisionTier.BF16,
                    )
                    core_selected_ids = selected_overlay.selected_span_ids
                    selected_ids = selected_overlay.selected_span_ids + selected_overlay.auxiliary_span_ids
                    method_confidence = selected_overlay.confidence
                else:  # pragma: no cover
                    raise ValueError(method)
                selector_ms = (time.perf_counter() - start) * 1000
                if method.startswith("learned_"):
                    selector_ms += learned_scorer_ms
                if method.startswith("cached_"):
                    selector_ms += cached_overlay_ms
                if method == "v1_compass":
                    retained_tokens = retained
                else:
                    retained_tokens = mandatory_tokens | {
                        token for span_id in selected_ids for token in range(by_id[span_id].token_start, by_id[span_id].token_end)
                    }
                row: dict[str, Any] = {
                    "dataset": example.dataset,
                    "split": example.split,
                    "sample_id": example.sample_id,
                    "question_type": example.question_type,
                    "method": method,
                    "budget_spec": budget_spec,
                    "budget_tokens": total_budget,
                    "prompt_tokens": len(offsets),
                    "selected_span_ids": list(selected_ids),
                    "selector_ms": selector_ms,
                    "index_build_ms": cached_index_ms if method.startswith("cached_") else 0.0,
                    "confidence": method_confidence,
                }
                row.update(
                    selector_metrics(
                        example, selected_ids, by_id, retained_tokens, len(offsets), kv_bytes_per_token
                    )
                )
                core_ids = set(selected_ids if core_selected_ids is None else core_selected_ids)
                row["core_pcr"] = float(any(proof <= core_ids for proof in example.gold_proofs))
                rows.append(row)
                append_jsonl(results_path, row)

    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "methods": methods,
        "budgets": budget_specs,
        "kv_bytes_per_token_bf16": kv_bytes_per_token,
        "records": aggregate(rows),
        "notes": {
            "proxy_boundary": "dense_hash, reranker, and graph_similarity are development proxies, not formal official baselines",
            "locked_test": args.dataset == "musique",
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "records": len(payload["records"]), "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
