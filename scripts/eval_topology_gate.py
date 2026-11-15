#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer
import torch

from proofkv.proofcache.index import EvidenceIndexer, extract_cues, hashed_embedding
from proofkv.proofcache.model import (
    CachedTypedRelationScorer,
    TypedRelationScorer,
    build_cached_overlay,
    build_learned_overlay,
    encode_cached_spans,
)
from proofkv.proofcache.schema import EvidenceIndex, EvidenceSpan, PrecisionTier, Relation, RelationType
from proofkv.proofcache.selector import AndOrProofSelector, ProofSearchConfig, TypedProofSearcher
from proofkv.proofcache.topology import TopologyConfig, make_topology_example
from proofkv.utils.io import append_jsonl

try:
    from eval_proofcache_selector import bm25_priority, select_by_priority, token_span
except ImportError:
    from scripts.eval_proofcache_selector import bm25_priority, select_by_priority, token_span


METHODS = {
    "bm25",
    "learned_single_path",
    "learned_topk_paths",
    "learned_and_or",
    "learned_and_or_aux",
    "relevance_same_count",
    "edge_shuffled",
    "random_edges",
    "broken_proof_same_count",
    "oracle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled proof-topology selector gate.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--scorer-checkpoint", default=None)
    parser.add_argument("--cached-scorer-checkpoint", default=None)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--budgets", default="128,256,0.08")
    parser.add_argument("--methods", default=",".join(sorted(METHODS)))
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--cached-relation-prior-weight", type=float, default=0.0)
    parser.add_argument("--cached-edge-non-none-floor", type=float, default=None)
    parser.add_argument("--learned-dense-core-count", type=int, default=16)
    parser.add_argument("--learned-edge-batch-size", type=int, default=64)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def config_for(index: int) -> tuple[TopologyConfig, str]:
    variant = index % 10
    robustness = (
        "base", "entity_collisions", "fact_reorder", "paraphrase", "marker_removal",
        "contradiction", "cycle", "broken_middle", "combined", "base",
    )[variant]
    return TopologyConfig(
        hops=(1, 2, 3, 4, 5, 6, 8)[index % 7],
        branch_factor=(1, 2, 4)[(index // 7) % 3],
        alternative_proofs=(1, 2, 4)[(index // 21) % 3],
        distractor_chains=(0, 4, 16)[(index // 63) % 3],
        # Cover every position even in the small n=128 development gate.
        evidence_position=("start", "middle", "end", "mixed")[(index // 10) % 4],
        entity_collisions=variant in {1, 8},
        fact_reorder=variant in {2, 8},
        paraphrase=variant in {3, 8},
        marker_removal=variant == 4,
        contradiction=variant in {5, 8},
        cycle=variant in {6, 8},
        broken_middle=variant in {7, 8},
    ), robustness


def proof_priority(overlay) -> list[str]:
    values = []
    for proof in overlay.proofs:
        for span_id in proof.span_ids:
            if span_id not in values:
                values.append(span_id)
    return values


def corrupt_index(index: EvidenceIndex, seed: int, random_types: bool) -> EvidenceIndex:
    rng = random.Random(seed)
    span_ids = [span.span_id for span in index.spans]
    relations = []
    for relation in index.relations:
        choices = [span_id for span_id in span_ids if span_id != relation.source]
        if not choices:
            continue
        target = rng.choice(choices)
        relation_type = rng.choice(list(RelationType)) if random_types else relation.relation_type
        relations.append(Relation(relation.source, target, relation_type, relation.probability))
    return EvidenceIndex(index.document_id, index.spans, tuple(relations), index.token_count, index.index_version + "+corrupt")


def parse_budget(spec: str, seq_len: int) -> tuple[str, int]:
    value = float(spec)
    return (f"{value:.4f}", max(1, int(round(seq_len * value)))) if value < 1 else (str(int(value)), int(value))


def pcr(example, selected: Sequence[str]) -> float:
    selected_set = set(selected)
    return float(any(proof <= selected_set for proof in example.gold_proofs))


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["axis"], row["axis_value"], row["budget_spec"], row["method"])].append(row)
    records = []
    for (axis, value, budget, method), items in sorted(groups.items()):
        latencies = sorted(item["selector_ms"] for item in items)
        records.append(
            {
                "axis": axis,
                "axis_value": value,
                "budget_spec": budget,
                "method": method,
                "n": len(items),
                "pcr": sum(item["pcr"] for item in items) / len(items),
                "oracle_fit_rate": sum(item["oracle_fit"] for item in items) / len(items),
                "pcr_given_oracle_fit": (
                    sum(item["pcr"] for item in items if item["oracle_fit"])
                    / max(1, sum(item["oracle_fit"] for item in items))
                ),
                "gold_recall": sum(item["gold_recall"] for item in items) / len(items),
                "route_precision": sum(item["route_precision"] for item in items) / len(items),
                "selector_p95_ms": latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)],
                "offline_index_p95_ms": sorted(item["offline_index_ms"] for item in items)[
                    min(len(items) - 1, math.ceil(0.95 * len(items)) - 1)
                ],
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if bool(args.scorer_checkpoint) == bool(args.cached_scorer_checkpoint):
        raise ValueError("provide exactly one of --scorer-checkpoint or --cached-scorer-checkpoint")
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = set(methods) - METHODS
    if unknown:
        raise ValueError(unknown)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.jsonl"
    if result_path.exists() or (out_dir / "metrics.json").exists():
        raise FileExistsError(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if args.cached_scorer_checkpoint:
        scorer, scorer_tokenizer = CachedTypedRelationScorer.load_checkpoint(
            args.cached_scorer_checkpoint, device="cuda"
        )
        cached_mode = True
    else:
        scorer, scorer_tokenizer = TypedRelationScorer.load_checkpoint(
            args.scorer_checkpoint, device="cuda"
        )
        cached_mode = False
    search_config = ProofSearchConfig(
        root_top_k=32, beam_size=48, max_hops=8, max_proofs=16,
        min_spans_per_proof=2, terminal_probability_floor=0.20,
    )
    searcher = TypedProofSearcher(search_config)
    rows = []

    for sample_index in range(args.max_samples):
        config, robustness = config_for(sample_index)
        example = make_topology_example(config, args.seed, sample_index)
        context, char_spans = example.render_context()
        query = f"\nQuestion: {example.question}\nAnswer:"
        prompt = context + query
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        evidence_spans = []
        for source, chars in zip(example.spans, char_spans):
            start, end = token_span(offsets, *chars)
            text = f"[{source.title}] {source.text}"
            evidence_spans.append(
                EvidenceSpan(
                    source.span_id, start, end, text, source.kind,
                    extract_cues(text), hashed_embedding(text, 64), source.metadata,
                )
            )
        index = EvidenceIndexer().build(example.sample_id, evidence_spans, len(offsets))
        by_id = index.by_id
        torch.cuda.synchronize()
        if cached_mode:
            offline_started = time.perf_counter()
            span_embeddings = encode_cached_spans(
                scorer, scorer_tokenizer, [span.text for span in index.spans]
            )
            torch.cuda.synchronize()
            offline_index_ms = (time.perf_counter() - offline_started) * 1000
            started = time.perf_counter()
            learned_overlay, learned_index = build_cached_overlay(
                scorer, scorer_tokenizer, example.question, index, span_embeddings,
                candidate_span_count=64, search_config=search_config,
                candidate_strategy="frontier",
                relation_prior_weight=args.cached_relation_prior_weight,
                edge_non_none_floor=args.cached_edge_non_none_floor,
            )
        else:
            offline_index_ms = 0.0
            started = time.perf_counter()
            learned_overlay, learned_index = build_learned_overlay(
                scorer, scorer_tokenizer, example.question, index,
                candidate_span_count=64, search_config=search_config,
                candidate_strategy="frontier",
                dense_core_count=args.learned_dense_core_count,
                edge_batch_size=args.learned_edge_batch_size,
            )
        torch.cuda.synchronize()
        learned_ms = (time.perf_counter() - started) * 1000
        shuffled_index = corrupt_index(learned_index, args.seed + sample_index, False)
        random_index = corrupt_index(learned_index, args.seed + sample_index + 999, True)
        shuffled_overlay = searcher.search(
            example.question, shuffled_index,
            root_scores=dict(learned_overlay.root_scores),
            terminal_scores=dict(learned_overlay.terminal_scores),
        )
        random_overlay = searcher.search(
            example.question, random_index,
            root_scores=dict(learned_overlay.root_scores),
            terminal_scores=dict(learned_overlay.terminal_scores),
        )

        for budget_input in args.budgets.split(","):
            budget_spec, total_budget = parse_budget(budget_input, len(offsets))
            query_start, _ = token_span(offsets, len(context), len(prompt))
            evidence_budget = max(0, total_budget - (len(offsets) - query_start) - 4)
            selections: dict[str, tuple[str, ...]] = {}
            selections["bm25"] = select_by_priority(
                bm25_priority(example.question, index.spans), by_id, evidence_budget
            )
            compositional = [proof for proof in learned_overlay.proofs if len(proof.span_ids) >= 2]
            selections["learned_single_path"] = select_by_priority(
                list((compositional or list(learned_overlay.proofs))[0].span_ids) if learned_overlay.proofs else [],
                by_id, evidence_budget,
            )
            selections["learned_topk_paths"] = select_by_priority(
                proof_priority(learned_overlay), by_id, evidence_budget
            )
            normal_core = AndOrProofSelector().select(
                learned_overlay, learned_index, evidence_budget * 2, 1, PrecisionTier.BF16
            )
            normal_aux = AndOrProofSelector(fill_auxiliary=True).select(
                learned_overlay, learned_index, evidence_budget * 2, 1, PrecisionTier.BF16
            )
            selections["learned_and_or"] = normal_core.selected_span_ids
            selections["learned_and_or_aux"] = normal_aux.selected_span_ids + normal_aux.auxiliary_span_ids
            shuffled_selected = AndOrProofSelector(fill_auxiliary=True).select(
                shuffled_overlay, shuffled_index, evidence_budget * 2, 1, PrecisionTier.BF16
            )
            selections["edge_shuffled"] = shuffled_selected.selected_span_ids + shuffled_selected.auxiliary_span_ids
            random_selected = AndOrProofSelector(fill_auxiliary=True).select(
                random_overlay, random_index, evidence_budget * 2, 1, PrecisionTier.BF16
            )
            selections["random_edges"] = random_selected.selected_span_ids + random_selected.auxiliary_span_ids
            normal_ids = list(selections["learned_and_or_aux"])
            relevance = sorted(learned_overlay.root_scores, key=lambda key: (-learned_overlay.root_scores[key], key))
            selections["relevance_same_count"] = select_by_priority(relevance, by_id, sum(by_id[x].token_count for x in normal_ids))[:len(normal_ids)]
            broken = list(normal_ids)
            for proof in example.gold_proofs:
                if proof <= set(broken):
                    victim = min(proof, key=lambda key: learned_overlay.root_scores.get(key, 0.0))
                    broken.remove(victim)
                    replacement = next((key for key in relevance if key not in broken and key not in example.gold_span_ids), None)
                    if replacement is not None:
                        broken.append(replacement)
                    break
            selections["broken_proof_same_count"] = tuple(broken)
            oracle_proof = min(example.gold_proofs, key=lambda proof: sum(by_id[x].token_count for x in proof))
            selections["oracle"] = select_by_priority(
                sorted(oracle_proof, key=lambda key: by_id[key].token_start), by_id, evidence_budget
            )
            oracle_fit = pcr(example, selections["oracle"])

            for method in methods:
                selected = selections[method]
                selected_tokens = {
                    token for span_id in selected for token in range(by_id[span_id].token_start, by_id[span_id].token_end)
                }
                gold_tokens = {
                    token for span_id in example.gold_span_ids for token in range(by_id[span_id].token_start, by_id[span_id].token_end)
                }
                base = {
                    "sample_id": example.sample_id,
                    "method": method,
                    "budget_spec": budget_spec,
                    "pcr": pcr(example, selected),
                    "oracle_fit": oracle_fit,
                    "gold_recall": len(selected_tokens & gold_tokens) / max(1, len(gold_tokens)),
                    "route_precision": len(selected_tokens & gold_tokens) / max(1, len(selected_tokens)),
                    "selector_ms": learned_ms if method != "bm25" and method != "oracle" else 0.0,
                    "offline_index_ms": offline_index_ms if method != "bm25" and method != "oracle" else 0.0,
                    "hops": config.hops,
                    "branch_factor": config.branch_factor,
                    "alternative_proofs": config.alternative_proofs,
                    "distractor_chains": config.distractor_chains,
                    "evidence_position": config.evidence_position,
                    "robustness": robustness,
                }
                axes = {
                    "overall": "all",
                    "hops": str(config.hops),
                    "branch_factor": str(config.branch_factor),
                    "alternative_proofs": str(config.alternative_proofs),
                    "distractor_chains": str(config.distractor_chains),
                    "evidence_position": config.evidence_position,
                    "robustness": robustness,
                }
                for axis, value in axes.items():
                    row = {**base, "axis": axis, "axis_value": value}
                    rows.append(row)
                    append_jsonl(result_path, row)
    payload = {
        "max_samples": args.max_samples,
        "budgets": args.budgets.split(","),
        "methods": methods,
        "scorer_checkpoint": args.scorer_checkpoint,
        "cached_scorer_checkpoint": args.cached_scorer_checkpoint,
        "records": aggregate(rows),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "records": len(payload["records"]), "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
