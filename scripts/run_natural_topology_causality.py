#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset

from proofkv.eval.generation import load_model_tokenizer
from proofkv.prefill_selectors import GuidepostPlan
from proofkv.proofcache.index import EvidenceIndexer, extract_cues, hashed_embedding
from proofkv.proofcache.model import TypedRelationScorer, build_learned_overlay
from proofkv.proofcache.schema import (
    EvidenceIndex,
    EvidenceSpan,
    PrecisionTier,
    Relation,
    RelationType,
)
from proofkv.proofcache.selector import AndOrProofSelector, ProofSearchConfig, TypedProofSearcher
from proofkv.utils.io import append_jsonl
from eval_natural_route_coverage import SINK, adaptive_budget_parts, token_span
from run_hotpot_compact_qa import (
    answer_scores,
    extract_short_answer,
    make_prompt,
    normalize_ircot_source,
    production_compact_decode,
)
from run_physical_kv_replay import _release_cuda


METHODS = (
    "proof_core",
    "relevance_same_bytes",
    "edge_shuffled",
    "random_edges",
    "broken_middle",
    "oracle_proof",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Natural MuSiQue topology-specific downstream causal interventions."
    )
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--exclude-results", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--sample-offset", type=int, default=128)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--intervention-seed", type=int, default=42823318)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(sample_id: str, seed: int) -> int:
    value = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(value[:8], "big")


def corrupt_index(index: EvidenceIndex, seed: int, random_types: bool) -> EvidenceIndex:
    rng = random.Random(seed)
    span_ids = [span.span_id for span in index.spans]
    relations: list[Relation] = []
    for relation in index.relations:
        candidates = [span_id for span_id in span_ids if span_id != relation.source]
        if not candidates:
            continue
        target = rng.choice(candidates)
        relation_type = rng.choice(list(RelationType)) if random_types else relation.relation_type
        relations.append(
            Relation(
                relation.source,
                target,
                relation_type,
                relation.probability,
                relation.evidence,
            )
        )
    return EvidenceIndex(
        index.document_id,
        index.spans,
        tuple(relations),
        index.token_count,
        index_version=index.index_version + ("+random" if random_types else "+shuffled"),
    )


def select_core(overlay: Any, index: EvidenceIndex, evidence_budget: int) -> tuple[str, ...]:
    selected = AndOrProofSelector(fill_auxiliary=False).select(
        overlay,
        index,
        byte_budget=evidence_budget * 2,
        values_per_token=1,
        precision=PrecisionTier.BF16,
    )
    return tuple(selected.selected_span_ids)


def exact_plan(
    seq_len: int,
    sink: int,
    recent: int,
    retained_total: int,
    index: EvidenceIndex,
    primary_ids: Iterable[str],
    filler_ids: Iterable[str],
    blocked_ids: set[str] | None = None,
) -> tuple[GuidepostPlan, dict[str, Any]]:
    blocked_ids = blocked_ids or set()
    stream = set(range(min(sink, seq_len))) | set(range(max(0, seq_len - recent), seq_len))
    target = min(seq_len, retained_total)
    guidepost_budget = max(0, target - len(stream))
    primary = [span_id for span_id in primary_ids if span_id not in blocked_ids]
    filler = [
        span_id
        for span_id in filler_ids
        if span_id not in blocked_ids and span_id not in primary
    ]
    ordered_ids = primary + filler
    ordered_tokens: list[int] = []
    primary_token_candidates: set[int] = set()
    for position, span_id in enumerate(ordered_ids):
        if span_id not in index.by_id:
            continue
        span = index.by_id[span_id]
        values = [
            token
            for token in range(max(sink, span.token_start), min(seq_len - recent, span.token_end))
            if token not in stream
        ]
        ordered_tokens.extend(values)
        if position < len(primary):
            primary_token_candidates.update(values)
    # Exact byte matching is completed with query-agnostic token order. This
    # padding is identical in definition across methods and never uses gold labels.
    ordered_tokens.extend(
        token
        for token in range(sink, max(sink, seq_len - recent))
        if token not in stream
    )
    selected_tokens: list[int] = []
    seen: set[int] = set()
    for token in ordered_tokens:
        if token in seen:
            continue
        selected_tokens.append(token)
        seen.add(token)
        if len(selected_tokens) >= guidepost_budget:
            break
    retained = sorted(stream | set(selected_tokens))
    if len(retained) != target:
        raise RuntimeError(f"exact token match failed: {len(retained)} != {target}")
    if retained[-1] != seq_len - 1:
        raise RuntimeError("final token not retained")
    spans = [
        (index.by_id[span_id].token_start, index.by_id[span_id].token_end)
        for span_id in primary
        if span_id in index.by_id
    ]
    plan = GuidepostPlan(
        guidepost_indices=sorted(selected_tokens),
        retained_indices=retained,
        spans=spans,
        retained_ratio=len(retained) / max(1, seq_len),
    )
    return plan, {
        "primary_span_ids": primary,
        "blocked_span_ids": sorted(blocked_ids),
        "primary_tokens_retained": len(primary_token_candidates & set(retained)),
        "padding_tokens_retained": len(set(retained) - stream - primary_token_candidates),
    }


def bootstrap_delta(
    left: list[float], right: list[float], samples: int, seed: int
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap inputs differ")
    deltas = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    means = []
    n = len(deltas)
    for _ in range(samples):
        means.append(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)))
    means.sort()
    low = means[max(0, int(0.025 * samples) - 1)]
    high = means[min(samples - 1, int(0.975 * samples))]
    return {
        "delta": statistics.fmean(deltas),
        "ci95_low": low,
        "ci95_high": high,
        "win_rate": sum(value > 0 for value in deltas) / n,
    }


def normalize_short(text: str) -> str:
    return " ".join(extract_short_answer(text).lower().split())


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.sample_offset < 0 or args.budget_tokens <= SINK:
        raise ValueError("invalid frozen protocol")
    data_path = Path(args.data_file)
    exclude_path = Path(args.exclude_results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.jsonl"
    metrics_path = out_dir / "metrics.json"
    manifest_path = out_dir / "run_manifest.json"
    if any(path.exists() for path in (result_path, metrics_path, manifest_path)):
        raise FileExistsError(out_dir)

    dataset = load_dataset("json", data_files={"validation": str(data_path)}, split="validation")
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    selected_indices = order[args.sample_offset : args.sample_offset + args.max_samples]
    if len(selected_indices) != args.max_samples:
        raise ValueError("insufficient selected examples")
    selected_ids = [
        str(normalize_ircot_source(dict(dataset[index]))["id"]) for index in selected_indices
    ]
    exclude_rows = [
        json.loads(line) for line in exclude_path.read_text().splitlines() if line.strip()
    ]
    excluded_ids = {str(row["id"]) for row in exclude_rows}
    overlap = sorted(set(selected_ids) & excluded_ids)
    if overlap:
        raise RuntimeError(f"formal causal slice overlaps prior downstream ids: {overlap[:3]}")
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("duplicate selected ids")
    manifest = {
        "arguments": {key: value for key, value in vars(args).items() if key != "out_dir"},
        "methods": list(METHODS),
        "dataset_sha256": file_sha256(data_path),
        "exclude_results_sha256": file_sha256(exclude_path),
        "selected_indices": selected_indices,
        "selected_ids": selected_ids,
        "excluded_unique_ids": len(excluded_ids),
        "overlap_count": len(overlap),
        "frozen_primary_gate": (
            "proof_core minus edge_shuffled answer F1 >= 0.03 and paired bootstrap "
            "two-sided 95% CI lower bound > 0"
        ),
        "claim_boundary": (
            "post-hoc disjoint downstream causal slice; the 500-example selector aggregate "
            "was previously observed, so this is not called an untouched locked test"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    model, tokenizer = load_model_tokenizer(
        args.model,
        dtype="bfloat16",
        attn_implementation=args.attn_implementation,
    )
    device = next(model.parameters()).device
    scorer, scorer_tokenizer = TypedRelationScorer.load_checkpoint(
        args.scorer_checkpoint, device=device
    )
    scorer.eval()
    search_config = ProofSearchConfig(
        root_top_k=16,
        beam_size=8,
        max_hops=8,
        max_proofs=8,
        normalize_path_probability=False,
    )
    searcher = TypedProofSearcher(search_config)
    all_rows: list[dict[str, Any]] = []

    for dataset_index in selected_indices:
        source = normalize_ircot_source(dict(dataset[dataset_index]))
        sample_id = str(source["id"])
        prompt, sentences, query = make_prompt(source, tokenizer)
        encoding = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = [(int(a), int(b)) for a, b in encoding.pop("offset_mapping")[0].tolist()]
        input_ids = encoding["input_ids"].to(device)
        sentence_spans = [token_span(offsets, *sentence["char_span"]) for sentence in sentences]
        gold_ids = [f"s{idx}" for idx, sentence in enumerate(sentences) if sentence["gold"]]
        spans = [
            EvidenceSpan(
                f"s{idx}",
                start,
                end,
                f"[{sentence['title']}] {sentence['text']}",
                cues=extract_cues(f"{sentence['title']} {sentence['text']}"),
                embedding=hashed_embedding(f"{sentence['title']} {sentence['text']}", 64),
                metadata={"title": sentence["title"], "sentence_index": sentence["sent_id"]},
            )
            for idx, ((start, end), sentence) in enumerate(zip(sentence_spans, sentences))
        ]
        index = EvidenceIndexer().build(sample_id, spans, len(offsets))
        query_start, _ = token_span(offsets, len(prompt) - len(query), len(prompt))
        retained_total, recent, evidence_budget = adaptive_budget_parts(
            len(offsets), min(1.0, args.budget_tokens / len(offsets)), query_start
        )
        torch.cuda.synchronize()
        selector_started = time.perf_counter()
        with torch.inference_mode():
            overlay, learned_index = build_learned_overlay(
                scorer,
                scorer_tokenizer,
                str(source["question"]),
                index,
                candidate_span_count=32,
                search_config=search_config,
                candidate_strategy="root",
                dense_core_count=4,
                edge_batch_size=256,
            )
        torch.cuda.synchronize()
        selector_seconds = time.perf_counter() - selector_started
        proof_ids = select_core(overlay, learned_index, evidence_budget)
        relevance_ids = tuple(
            sorted(overlay.root_scores, key=lambda key: (-overlay.root_scores[key], key))
        )
        sample_seed = stable_seed(sample_id, args.intervention_seed)
        shuffled_index = corrupt_index(learned_index, sample_seed, False)
        random_index = corrupt_index(learned_index, sample_seed + 1, True)
        shuffled_overlay = searcher.search(
            str(source["question"]),
            shuffled_index,
            root_scores=dict(overlay.root_scores),
            terminal_scores=dict(overlay.terminal_scores),
        )
        random_overlay = searcher.search(
            str(source["question"]),
            random_index,
            root_scores=dict(overlay.root_scores),
            terminal_scores=dict(overlay.terminal_scores),
        )
        shuffled_ids = select_core(shuffled_overlay, shuffled_index, evidence_budget)
        random_ids = select_core(random_overlay, random_index, evidence_budget)

        selected_proof_ids = set(
            AndOrProofSelector(fill_auxiliary=False)
            .select(
                overlay,
                learned_index,
                byte_budget=evidence_budget * 2,
                values_per_token=1,
                precision=PrecisionTier.BF16,
            )
            .selected_proof_ids
        )
        victims: set[str] = set()
        for proof in overlay.proofs:
            if proof.proof_id not in selected_proof_ids or len(proof.span_ids) < 2:
                continue
            # For 2-span natural proofs the terminal is the bridge target; for
            # longer paths remove the strict middle node.
            victim_index = len(proof.span_ids) // 2
            victims.add(proof.span_ids[victim_index])
        if not victims:
            compositional = next((proof for proof in overlay.proofs if len(proof.span_ids) >= 2), None)
            if compositional is not None:
                victims.add(compositional.span_ids[len(compositional.span_ids) // 2])
        broken_ids = tuple(span_id for span_id in proof_ids if span_id not in victims)

        filler_ids = tuple(
            sorted(
                (span.span_id for span in learned_index.spans),
                key=lambda value: hashlib.sha256(f"{sample_id}:{value}".encode()).hexdigest(),
            )
        )
        primary = {
            "proof_core": (proof_ids, set()),
            "relevance_same_bytes": (relevance_ids, set()),
            "edge_shuffled": (shuffled_ids, set()),
            "random_edges": (random_ids, set()),
            "broken_middle": (broken_ids, victims),
            "oracle_proof": (tuple(gold_ids), set()),
        }
        plans: dict[str, tuple[GuidepostPlan, dict[str, Any]]] = {}
        for method in METHODS:
            ids, blocked = primary[method]
            plans[method] = exact_plan(
                len(offsets),
                SINK,
                recent,
                retained_total,
                learned_index,
                ids,
                filler_ids,
                blocked,
            )
        lengths = {len(plan.retained_indices) for plan, _ in plans.values()}
        if lengths != {retained_total}:
            raise RuntimeError(f"within-example byte mismatch for {sample_id}: {lengths}")

        method_order = list(METHODS)
        random.Random(sample_seed + 2).shuffle(method_order)
        for method in method_order:
            plan, plan_metadata = plans[method]
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction, cache, _, warnings, generation_timing = production_compact_decode(
                model,
                tokenizer,
                input_ids,
                plan.retained_indices,
                int(input_ids.shape[1]),
                args.max_new_tokens,
                512,
            )
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - started
            del cache
            _release_cuda()
            short = extract_short_answer(prediction)
            em, f1, contains = answer_scores(short, str(source["answer"]))
            retained_set = set(plan.retained_indices)
            gold_tokens = {
                token
                for gold_id in gold_ids
                for token in range(
                    learned_index.by_id[gold_id].token_start,
                    learned_index.by_id[gold_id].token_end,
                )
            }
            gold_full = all(
                set(
                    range(
                        learned_index.by_id[gold_id].token_start,
                        learned_index.by_id[gold_id].token_end,
                    )
                )
                <= retained_set
                for gold_id in gold_ids
            )
            row = {
                "id": sample_id,
                "question_type": source["type"],
                "method": method,
                "answer": str(source["answer"]),
                "prediction": prediction,
                "short_prediction": short,
                "answer_em": em,
                "answer_f1": f1,
                "answer_contains": contains,
                "prompt_tokens": len(offsets),
                "retained_tokens": len(plan.retained_indices),
                "retained_ratio": plan.retained_ratio,
                "gold_token_recall": len(retained_set & gold_tokens) / max(1, len(gold_tokens)),
                "gold_proof_complete": float(gold_full),
                "selector_seconds_shared": selector_seconds,
                "generation_seconds": generation_seconds,
                "generation_timing": generation_timing,
                "route_sha256": hashlib.sha256(
                    json.dumps(plan.retained_indices, separators=(",", ":")).encode()
                ).hexdigest(),
                "broken_middle_applicable": bool(victims),
                "plan_metadata": plan_metadata,
                "warnings": warnings,
            }
            all_rows.append(row)
            append_jsonl(result_path, row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[str(row["method"])].append(row)
    records = []
    for method in METHODS:
        values = grouped[method]
        records.append(
            {
                "method": method,
                "n": len(values),
                "answer_em": statistics.fmean(row["answer_em"] for row in values),
                "answer_f1": statistics.fmean(row["answer_f1"] for row in values),
                "answer_contains": statistics.fmean(row["answer_contains"] for row in values),
                "gold_token_recall": statistics.fmean(row["gold_token_recall"] for row in values),
                "gold_proof_complete": statistics.fmean(row["gold_proof_complete"] for row in values),
                "retained_tokens_min": min(row["retained_tokens"] for row in values),
                "retained_tokens_max": max(row["retained_tokens"] for row in values),
                "generation_seconds_mean": statistics.fmean(row["generation_seconds"] for row in values),
            }
        )
    by_id_method = {
        (str(row["id"]), str(row["method"])): row for row in all_rows
    }
    comparisons = {}
    for opponent in ("relevance_same_bytes", "edge_shuffled", "random_edges", "broken_middle"):
        left = [by_id_method[(sample_id, "proof_core")]["answer_f1"] for sample_id in selected_ids]
        right = [by_id_method[(sample_id, opponent)]["answer_f1"] for sample_id in selected_ids]
        comparisons[f"proof_core_vs_{opponent}"] = bootstrap_delta(
            left,
            right,
            args.bootstrap_samples,
            stable_seed(opponent, args.intervention_seed),
        )
    oracle_left = [by_id_method[(sample_id, "oracle_proof")]["answer_f1"] for sample_id in selected_ids]
    proof_right = [by_id_method[(sample_id, "proof_core")]["answer_f1"] for sample_id in selected_ids]
    comparisons["oracle_proof_vs_proof_core"] = bootstrap_delta(
        oracle_left,
        proof_right,
        args.bootstrap_samples,
        args.intervention_seed + 99,
    )
    edge_gate = comparisons["proof_core_vs_edge_shuffled"]
    integrity = {
        "rows_expected": len(selected_ids) * len(METHODS),
        "rows_observed": len(all_rows),
        "unique_pairs": len(by_id_method),
        "prior_id_overlap": len(overlap),
        "all_exactly_matched_tokens": all(
            len({by_id_method[(sample_id, method)]["retained_tokens"] for method in METHODS}) == 1
            for sample_id in selected_ids
        ),
        "extreme_zero_f1_rows": sum(row["answer_f1"] == 0 for row in all_rows),
        "broken_middle_applicable_rate": statistics.fmean(
            float(by_id_method[(sample_id, "broken_middle")]["broken_middle_applicable"])
            for sample_id in selected_ids
        ),
    }
    integrity["passed"] = (
        integrity["rows_expected"] == integrity["rows_observed"]
        == integrity["unique_pairs"]
        and integrity["prior_id_overlap"] == 0
        and integrity["all_exactly_matched_tokens"]
    )
    payload = {
        "protocol": manifest,
        "records": records,
        "paired_bootstrap_f1": comparisons,
        "integrity": integrity,
        "primary_causal_gate": {
            "delta_at_least_0_03": edge_gate["delta"] >= 0.03,
            "ci95_low_above_zero": edge_gate["ci95_low"] > 0,
            "passed": edge_gate["delta"] >= 0.03 and edge_gate["ci95_low"] > 0,
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not integrity["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
