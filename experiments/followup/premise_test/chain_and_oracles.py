#!/usr/bin/env python
"""Premise test: gold-span coverage of every route, and equal-byte reference arms.

Re-runs the frozen natural MuSiQue topology slice (128 disjoint examples, exactly
512 retained tokens) through the *identical* prompt / index / frozen scorer /
exact-byte plan / compact-prefill decoder path as
scripts/run_natural_topology_causality.py, and adds:

  (a)  per-gold-span coverage measurement for every route (frozen arms, the
       span-atomic selector baselines, StreamingLLM-analytic, new arms);
  (b)  equal-byte reference arms in which every gold span keeps a fragment of its own
       tokens (UNIFORM_PARTIAL*), decoded through the same path.

Span granularity: a "span" is a whole MuSiQue paragraph (20 per example, 2-4
gold), NOT a sentence.  Nothing in the frozen scorer, policy or compiler is
modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

IMPL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(IMPL / "scripts"))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402

from proofkv.eval.generation import load_model_tokenizer  # noqa: E402
from proofkv.prefill_selectors import GuidepostPlan  # noqa: E402
from proofkv.proofcache.index import EvidenceIndexer, extract_cues, hashed_embedding  # noqa: E402
from proofkv.proofcache.model import TypedRelationScorer, build_learned_overlay, score_roots  # noqa: E402
from proofkv.proofcache.schema import EvidenceIndex, EvidenceSpan, PrecisionTier  # noqa: E402
from proofkv.proofcache.selector import AndOrProofSelector, ProofSearchConfig, TypedProofSearcher  # noqa: E402
from proofkv.utils.io import append_jsonl  # noqa: E402

from eval_natural_route_coverage import SINK, adaptive_budget_parts, token_span  # noqa: E402
from eval_proofcache_selector import (  # noqa: E402
    bm25_priority,
    dense_priority,
    graph_similarity_priority,
    reranker_priority,
    select_by_priority,
)
from run_hotpot_compact_qa import (  # noqa: E402
    answer_scores,
    extract_short_answer,
    make_prompt,
    normalize_ircot_source,
    production_compact_decode,
)
from run_natural_topology_causality import (  # noqa: E402
    METHODS as FROZEN_METHODS,
    bootstrap_delta,
    corrupt_index,
    exact_plan,
    file_sha256,
    select_core,
    stable_seed,
)
from run_physical_kv_replay import _release_cuda  # noqa: E402

FROZEN_METRICS = (
    str(Path(__file__).resolve().parents[3] / "results/paper/reference/musique_interventions_original_metrics.json")
)

# ---------------------------------------------------------------- arm catalogue
NEW_DECODED_ARMS = (
    "uniform_partial",            # literal spec: budget-proportional share per gold span, scorer within span
    "uniform_partial_head",       # literal spec, first-k tokens within span
    "uniform_partial_f25_scorer",
    "uniform_partial_f50_scorer",
    "uniform_partial_f75_scorer",
    "uniform_partial_f25_head",
    "uniform_partial_f50_head",
    "uniform_partial_f75_head",
)
COVERAGE_ONLY_ARMS = (
    "bm25_span_atomic",
    "reranker_proxy_span_atomic",
    "graph_similarity_proxy_span_atomic",
    "dense_hash_proxy_span_atomic",
    "bm25_exact512",
    "reranker_proxy_exact512",
    "graph_similarity_proxy_exact512",
    "dense_hash_proxy_exact512",
    "streamingllm_analytic_exact512",
)
DECODED_ARMS = tuple(FROZEN_METHODS) + NEW_DECODED_ARMS
ALL_ARMS = DECODED_ARMS + COVERAGE_ONLY_ARMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--sample-offset", type=int, default=128)
    parser.add_argument("--exclusion-offset", type=int, default=0)
    parser.add_argument("--exclusion-size", type=int, default=128)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--intervention-seed", type=int, default=42823318)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=100000)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--skip-decode", action="store_true", help="routes + coverage only (gate check)")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


# ---------------------------------------------------------------- helpers
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def route_hash(indices: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(indices), separators=(",", ":")).encode()).hexdigest()


def sentence_units(text: str) -> list[tuple[int, int]]:
    """Char spans of sentence units inside `text` (partition; whitespace attached to the left unit)."""
    units: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end = match.end()
        if end - start > 0:
            units.append((start, end))
            start = end
    if start < len(text):
        units.append((start, len(text)))
    if not units:
        units.append((0, len(text)))
    return units


def coverage_stats(retained: set[int], index: EvidenceIndex, gold_ids: Sequence[str]) -> dict[str, Any]:
    per_span = []
    for gold_id in gold_ids:
        span = index.by_id[gold_id]
        tokens = set(range(span.token_start, span.token_end))
        per_span.append(len(tokens & retained) / max(1, len(tokens)))
    gold_tokens = {
        token for gold_id in gold_ids for token in range(index.by_id[gold_id].token_start, index.by_id[gold_id].token_end)
    }
    touched = sum(value > 0 for value in per_span)
    return {
        "gold_span_count": len(gold_ids),
        "gold_spans_touched": touched,
        "all_gold_touched": float(touched == len(gold_ids)),
        "absent_gold_spans": len(gold_ids) - touched,
        "per_gold_span_retention": per_span,
        "mean_gold_span_retention": statistics.fmean(per_span) if per_span else 0.0,
        "min_gold_span_retention": min(per_span) if per_span else 0.0,
        "gold_token_recall": len(retained & gold_tokens) / max(1, len(gold_tokens)),
        "gold_proof_complete": float(all(value >= 1.0 for value in per_span)),
    }


def token_plan(
    seq_len: int,
    sink: int,
    recent: int,
    retained_total: int,
    primary_tokens: Sequence[int],
    blocked_tokens: set[int],
    spans: Sequence[tuple[int, int]],
    filler_token_order: Sequence[int] = (),
) -> tuple[GuidepostPlan, dict[str, Any]]:
    """Token-level analogue of the frozen exact_plan: primary tokens first, then the SAME
    query-agnostic padding as exact_plan (whole filler spans in the per-example hash order,
    then positional), excluding `blocked_tokens` from the padding."""
    stream = set(range(min(sink, seq_len))) | set(range(max(0, seq_len - recent), seq_len))
    target = min(seq_len, retained_total)
    guidepost_budget = max(0, target - len(stream))
    low, high = sink, max(sink, seq_len - recent)
    ordered: list[int] = [token for token in primary_tokens if low <= token < high and token not in stream]
    primary_set = set(ordered)
    ordered.extend(token for token in filler_token_order if low <= token < high and token not in stream and token not in blocked_tokens)
    ordered.extend(token for token in range(low, high) if token not in stream and token not in blocked_tokens)
    selected: list[int] = []
    seen: set[int] = set()
    for token in ordered:
        if token in seen:
            continue
        selected.append(token)
        seen.add(token)
        if len(selected) >= guidepost_budget:
            break
    retained = sorted(stream | set(selected))
    if len(retained) != target:
        raise RuntimeError(f"exact token match failed: {len(retained)} != {target}")
    if retained[-1] != seq_len - 1:
        raise RuntimeError("final token not retained")
    plan = GuidepostPlan(
        guidepost_indices=sorted(selected),
        retained_indices=retained,
        spans=list(spans),
        retained_ratio=len(retained) / max(1, seq_len),
    )
    return plan, {
        "primary_tokens_requested": len(primary_set),
        "primary_tokens_retained": len(primary_set & set(retained)),
        "blocked_tokens": len(blocked_tokens),
        "padding_tokens_retained": len(set(retained) - stream - primary_set),
    }


def proportional_shares(lengths: Sequence[int], budget: int) -> list[int]:
    """Largest-remainder allocation of `budget` tokens across spans, capped at each span length."""
    total = sum(lengths)
    if total <= budget:
        return list(lengths)
    raw = [budget * length / total for length in lengths]
    shares = [min(length, int(value)) for value, length in zip(raw, lengths)]
    remainder = budget - sum(shares)
    order = sorted(range(len(lengths)), key=lambda i: -(raw[i] - int(raw[i])))
    for i in order:
        if remainder <= 0:
            break
        if shares[i] < lengths[i]:
            shares[i] += 1
            remainder -= 1
    return shares


def sign_flip_pvalues(values: list[float], samples: int, seed: int) -> dict[str, float]:
    """Paired sign-flip test; one-sided (frozen definition, H1: mean>0) and two-sided."""
    observed = statistics.fmean(values)
    rng = random.Random(seed)
    exceed_one = exceed_two = 0
    for _ in range(samples):
        value = statistics.fmean(item if rng.random() < 0.5 else -item for item in values)
        exceed_one += value >= observed - 1e-15
        exceed_two += abs(value) >= abs(observed) - 1e-15
    return {
        "p_one_sided_greater": (exceed_one + 1) / (samples + 1),
        "p_two_sided": (exceed_two + 1) / (samples + 1),
    }


# ---------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    started_wall = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.jsonl"
    routes_path = out_dir / "routes.jsonl"
    metrics_path = out_dir / "metrics.json"
    manifest_path = out_dir / "run_manifest.json"
    if any(path.exists() for path in (result_path, metrics_path, manifest_path, routes_path)):
        raise FileExistsError(out_dir)

    data_path = Path(args.data_file)
    dataset = load_dataset("json", data_files={"validation": str(data_path)}, split="validation")
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    selected_indices = order[args.sample_offset : args.sample_offset + args.max_samples]
    exclusion_indices = order[args.exclusion_offset : args.exclusion_offset + args.exclusion_size]
    selected_ids = [str(normalize_ircot_source(dict(dataset[i]))["id"]) for i in selected_indices]
    excluded_ids = {str(normalize_ircot_source(dict(dataset[i]))["id"]) for i in exclusion_indices}
    overlap = sorted(set(selected_ids) & excluded_ids)
    frozen = json.loads(Path(FROZEN_METRICS).read_text())
    frozen_protocol = frozen["protocol"]
    if selected_indices != frozen_protocol["selected_indices"] or selected_ids != frozen_protocol["selected_ids"]:
        raise RuntimeError("reconstructed slice does not match the frozen manifest")
    if overlap or len(excluded_ids) != frozen_protocol["excluded_unique_ids"]:
        raise RuntimeError(f"exclusion reconstruction failed: overlap={len(overlap)} n={len(excluded_ids)}")

    code_files = {
        "experiments/followup/premise_test/chain_and_oracles.py": Path(__file__).resolve(),
        "scripts/run_natural_topology_causality.py": IMPL / "scripts/run_natural_topology_causality.py",
        "scripts/run_hotpot_compact_qa.py": IMPL / "scripts/run_hotpot_compact_qa.py",
        "scripts/eval_natural_route_coverage.py": IMPL / "scripts/eval_natural_route_coverage.py",
        "scripts/eval_proofcache_selector.py": IMPL / "scripts/eval_proofcache_selector.py",
        "proofkv/proofcache/model.py": IMPL / "proofkv/proofcache/model.py",
        "proofkv/proofcache/selector.py": IMPL / "proofkv/proofcache/selector.py",
        "proofkv/proofcache/index.py": IMPL / "proofkv/proofcache/index.py",
        "proofkv/proofcache/schema.py": IMPL / "proofkv/proofcache/schema.py",
        "proofkv/physical_kv.py": IMPL / "proofkv/physical_kv.py",
        "proofkv/prefill_selectors.py": IMPL / "proofkv/prefill_selectors.py",
    }
    implementation_sha256 = {key: file_sha256(path) for key, path in code_files.items()}
    scorer_dir = Path(args.scorer_checkpoint)
    scorer_sha256 = {
        str(path.relative_to(scorer_dir)): file_sha256(path) for path in sorted(scorer_dir.rglob("*")) if path.is_file()
    }
    import transformers

    manifest = {
        "experiment_id": "premise_test/chain_and_oracles",
        "purpose": "Partial-coverage test: per-gold-span coverage of every route and "
        "equal-byte UNIFORM_PARTIAL arms decoded through the frozen path.",
        "span_granularity": "whole MuSiQue paragraph (20 per example, 2-4 gold per question); NOT sentence-level",
        "arguments": {key: value for key, value in vars(args).items() if key != "out_dir"},
        "command_line": " ".join(sys.argv),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": args.model,
        "model_revision": args.model_revision,
        "scorer_checkpoint": str(scorer_dir),
        "scorer_checkpoint_sha256": scorer_sha256,
        "implementation_sha256": implementation_sha256,
        "software_package_manifest_sha256": file_sha256(IMPL / "MANIFEST.sha256"),
        "config_hash": sha256_text(json.dumps({"impl": implementation_sha256, "scorer": scorer_sha256, "args": vars(args)}, sort_keys=True)),
        "dataset_sha256": file_sha256(data_path),
        "frozen_reference_metrics": FROZEN_METRICS,
        "frozen_reference_metrics_sha256": file_sha256(Path(FROZEN_METRICS)),
        "selected_indices": selected_indices,
        "selected_ids": selected_ids,
        "exclusion_reconstruction": "seed-17 shuffle positions 0:128 (locked_v6_qwen25_musique_test_matrix_n128); "
        "original exclusion results.jsonl not in the frozen archive (sha256 481e6063...)",
        "excluded_unique_ids": len(excluded_ids),
        "overlap_count": len(overlap),
        "opened_after_submission": False,
        "opened_after_submission_note": "The 128-example disjoint slice was already opened for the submitted paper "
        "(TARGETED_GPU_EXPERIMENT_FREEZE_20260714.md, Group 2). No protected complement is opened here.",
        "frozen_policy": {
            "root_top_k": 16, "beam_size": 8, "max_hops": 8, "max_proofs": 8, "normalize_path_probability": False,
            "candidate_span_count": 32, "candidate_strategy": "root", "dense_core_count": 4, "edge_batch_size": 256,
            "compact_prefill_chunk_size": 512, "fill_auxiliary": False,
        },
        "decoded_arms": list(DECODED_ARMS),
        "coverage_only_arms": list(COVERAGE_ONLY_ARMS),
        "new_arm_definitions": {
            "uniform_partial": "literal spec: the evidence budget is split across gold spans in proportion to their "
            "token lengths (largest-remainder rounding, capped at span length); within each span, sentence units are "
            "ranked by the frozen root-relevance head (score_roots on '[title] sentence'), filled in rank order, the "
            "last unit truncated head-first to hit the share exactly. If all gold tokens fit, the arm equals oracle_proof.",
            "uniform_partial_head": "as uniform_partial but the within-span selection is the first k tokens of the span",
            "uniform_partial_f{25,50,75}_{scorer,head}": "every gold span keeps k_i = max(1, round(f * len_i)) of its "
            "own tokens by the named within-span selector; the remainder of the 512-token budget is filled by the same "
            "query-agnostic padding as every frozen arm (whole filler spans in the per-example hash order, then positional), "
            "EXCEPT that withheld gold tokens are excluded from padding so the realized per-span fraction equals the nominal one.",
            "padding_note": "The six frozen arms pad without any gold knowledge. The uniform_partial* arms are "
            "gold-informed by construction (like oracle_proof), so blocking withheld gold tokens from padding does not "
            "add information the arm does not already use.",
            "span_atomic_baselines": "bm25 / reranker_proxy / graph_similarity_proxy / dense_hash_proxy priorities from "
            "eval_proofcache_selector.py; *_span_atomic = select_by_priority (whole spans only, <= budget, no padding, "
            "the selector-protocol contract); *_exact512 = the same priority through the frozen exact_plan "
            "(last span may be truncated, positional padding to exactly 512).",
            "streamingllm_analytic_exact512": "4 sink tokens + the most recent 508 prompt positions (position-only rule; no filler padding by definition).",
        },
        "statistics": "10,000-sample paired bootstrap percentile CI (frozen bootstrap_delta) and 100,000-replicate "
        "paired sign-flip test (frozen one-sided definition plus two-sided), as in the paper's reproducibility protocol and "
        "scripts/audit_targeted_gpu_experiments.py",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_wall)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    tokenizer = None
    model = None
    if args.skip_decode:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        model, tokenizer = load_model_tokenizer(args.model, dtype="bfloat16", attn_implementation=args.attn_implementation)
        device = next(model.parameters()).device
    scorer, scorer_tokenizer = TypedRelationScorer.load_checkpoint(args.scorer_checkpoint, device=device)
    scorer.eval()
    search_config = ProofSearchConfig(root_top_k=16, beam_size=8, max_hops=8, max_proofs=8, normalize_path_probability=False)
    searcher = TypedProofSearcher(search_config)

    all_rows: list[dict[str, Any]] = []
    for position, dataset_index in enumerate(selected_indices):
        source = normalize_ircot_source(dict(dataset[dataset_index]))
        sample_id = str(source["id"])
        question = str(source["question"])
        prompt, sentences, query = make_prompt(source, tokenizer)
        encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        offsets = [(int(a), int(b)) for a, b in encoding.pop("offset_mapping")[0].tolist()]
        input_ids = encoding["input_ids"].to(device)
        seq_len = len(offsets)
        sentence_spans = [token_span(offsets, *sentence["char_span"]) for sentence in sentences]
        gold_ids = [f"s{idx}" for idx, sentence in enumerate(sentences) if sentence["gold"]]
        spans = [
            EvidenceSpan(
                f"s{idx}", start, end, f"[{sentence['title']}] {sentence['text']}",
                cues=extract_cues(f"{sentence['title']} {sentence['text']}"),
                embedding=hashed_embedding(f"{sentence['title']} {sentence['text']}", 64),
                metadata={"title": sentence["title"], "sentence_index": sentence["sent_id"]},
            )
            for idx, ((start, end), sentence) in enumerate(zip(sentence_spans, sentences))
        ]
        index = EvidenceIndexer().build(sample_id, spans, seq_len)
        query_start, _ = token_span(offsets, len(prompt) - len(query), len(prompt))
        retained_total, recent, evidence_budget = adaptive_budget_parts(seq_len, min(1.0, args.budget_tokens / seq_len), query_start)

        # ---------------- frozen selector path (identical to the frozen script)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        selector_started = time.perf_counter()
        with torch.inference_mode():
            overlay, learned_index = build_learned_overlay(
                scorer, scorer_tokenizer, question, index, candidate_span_count=32, search_config=search_config,
                candidate_strategy="root", dense_core_count=4, edge_batch_size=256,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        selector_seconds = time.perf_counter() - selector_started
        proof_ids = select_core(overlay, learned_index, evidence_budget)
        relevance_ids = tuple(sorted(overlay.root_scores, key=lambda key: (-overlay.root_scores[key], key)))
        sample_seed = stable_seed(sample_id, args.intervention_seed)
        shuffled_index = corrupt_index(learned_index, sample_seed, False)
        random_index = corrupt_index(learned_index, sample_seed + 1, True)
        shuffled_overlay = searcher.search(question, shuffled_index, root_scores=dict(overlay.root_scores), terminal_scores=dict(overlay.terminal_scores))
        random_overlay = searcher.search(question, random_index, root_scores=dict(overlay.root_scores), terminal_scores=dict(overlay.terminal_scores))
        shuffled_ids = select_core(shuffled_overlay, shuffled_index, evidence_budget)
        random_ids = select_core(random_overlay, random_index, evidence_budget)
        selected_proof_ids = set(
            AndOrProofSelector(fill_auxiliary=False)
            .select(overlay, learned_index, byte_budget=evidence_budget * 2, values_per_token=1, precision=PrecisionTier.BF16)
            .selected_proof_ids
        )
        victims: set[str] = set()
        for proof in overlay.proofs:
            if proof.proof_id not in selected_proof_ids or len(proof.span_ids) < 2:
                continue
            victims.add(proof.span_ids[len(proof.span_ids) // 2])
        if not victims:
            compositional = next((proof for proof in overlay.proofs if len(proof.span_ids) >= 2), None)
            if compositional is not None:
                victims.add(compositional.span_ids[len(compositional.span_ids) // 2])
        broken_ids = tuple(span_id for span_id in proof_ids if span_id not in victims)
        filler_ids = tuple(sorted((span.span_id for span in learned_index.spans), key=lambda value: hashlib.sha256(f"{sample_id}:{value}".encode()).hexdigest()))
        primary = {
            "proof_core": (proof_ids, set()),
            "relevance_same_bytes": (relevance_ids, set()),
            "edge_shuffled": (shuffled_ids, set()),
            "random_edges": (random_ids, set()),
            "broken_middle": (broken_ids, victims),
            "oracle_proof": (tuple(gold_ids), set()),
        }
        plans: dict[str, tuple[GuidepostPlan, dict[str, Any]]] = {}
        for method in FROZEN_METHODS:
            ids, blocked = primary[method]
            plans[method] = exact_plan(seq_len, SINK, recent, retained_total, learned_index, ids, filler_ids, blocked)

        # ---------------- span-atomic selector baselines (selector-protocol contract) and exact-512 versions
        stream = set(range(min(SINK, seq_len))) | set(range(max(0, seq_len - recent), seq_len))
        guidepost_budget = max(0, min(seq_len, retained_total) - len(stream))
        priorities = {
            "bm25": bm25_priority(question, learned_index.spans),
            "reranker_proxy": reranker_priority(question, learned_index.spans),
            "graph_similarity_proxy": graph_similarity_priority(question, learned_index.spans),
            "dense_hash_proxy": dense_priority(question, learned_index.spans),
        }
        coverage_only_routes: dict[str, tuple[list[int], dict[str, Any]]] = {}
        for name, priority in priorities.items():
            atomic_ids = select_by_priority(priority, learned_index.by_id, guidepost_budget)
            atomic_tokens = set(stream)
            for span_id in atomic_ids:
                span = learned_index.by_id[span_id]
                atomic_tokens |= {t for t in range(span.token_start, span.token_end) if SINK <= t < seq_len - recent}
            coverage_only_routes[f"{name}_span_atomic"] = (sorted(atomic_tokens), {"selected_span_ids": list(atomic_ids), "retained_tokens": len(atomic_tokens)})
            plan, meta = exact_plan(seq_len, SINK, recent, retained_total, learned_index, tuple(priority), filler_ids, set())
            coverage_only_routes[f"{name}_exact512"] = (list(plan.retained_indices), meta)
        recent_first = list(range(seq_len - recent - 1, SINK - 1, -1))
        plan, meta = token_plan(seq_len, SINK, recent, retained_total, recent_first, set(), [])
        coverage_only_routes["streamingllm_analytic_exact512"] = (list(plan.retained_indices), meta)

        # ---------------- new UNIFORM_PARTIAL arms
        gold_lengths = [learned_index.by_id[g].token_end - learned_index.by_id[g].token_start for g in gold_ids]
        gold_token_sets = {g: set(range(learned_index.by_id[g].token_start, learned_index.by_id[g].token_end)) for g in gold_ids}
        all_gold_tokens = set().union(*gold_token_sets.values()) if gold_ids else set()

        # within-span sentence units scored by the frozen root head
        unit_tokens_by_gold: dict[str, list[list[int]]] = {}
        unit_scores_by_gold: dict[str, list[float]] = {}
        for gold_id, sentence_idx in ((g, int(g[1:])) for g in gold_ids):
            sentence = sentences[sentence_idx]
            span = learned_index.by_id[gold_id]
            text = sentence["text"]
            line_start = sentence["char_span"][0]
            prefix = f"[{sentence['title']}] "
            text_start = line_start + len(prefix)
            units = sentence_units(text)
            unit_char = [(text_start + a, text_start + b) for a, b in units]
            unit_lists: list[list[int]] = [[] for _ in units]
            for token in range(span.token_start, span.token_end):
                char_start = offsets[token][0]
                unit_idx = 0
                for k, (a, b) in enumerate(unit_char):
                    if char_start >= a:
                        unit_idx = k
                unit_lists[unit_idx].append(token)
            unit_texts = [f"{prefix}{text[a:b].strip()}" for a, b in units]
            with torch.inference_mode():
                scores = score_roots(scorer, scorer_tokenizer, question, unit_texts)
            unit_tokens_by_gold[gold_id] = unit_lists
            unit_scores_by_gold[gold_id] = [float(s) for s in scores]

        def within_span(gold_id: str, k: int, selector: str) -> list[int]:
            span = learned_index.by_id[gold_id]
            if selector == "head":
                return list(range(span.token_start, min(span.token_end, span.token_start + k)))
            chosen: list[int] = []
            ranked = sorted(range(len(unit_tokens_by_gold[gold_id])), key=lambda u: (-unit_scores_by_gold[gold_id][u], u))
            for u in ranked:
                if len(chosen) >= k:
                    break
                tokens = unit_tokens_by_gold[gold_id][u]
                take = min(len(tokens), k - len(chosen))
                chosen.extend(tokens[:take])
            return chosen

        new_arm_specs: dict[str, tuple[list[int], str]] = {}
        literal_shares = proportional_shares(gold_lengths, guidepost_budget)
        new_arm_specs["uniform_partial"] = (literal_shares, "scorer")
        new_arm_specs["uniform_partial_head"] = (literal_shares, "head")
        for fraction in (0.25, 0.50, 0.75):
            shares = [max(1, int(round(fraction * length))) for length in gold_lengths]
            new_arm_specs[f"uniform_partial_f{int(fraction*100)}_scorer"] = (shares, "scorer")
            new_arm_specs[f"uniform_partial_f{int(fraction*100)}_head"] = (shares, "head")
        filler_token_order: list[int] = []
        for span_id in filler_ids:
            span = learned_index.by_id[span_id]
            filler_token_order.extend(range(span.token_start, span.token_end))
        for arm, (shares, selector) in new_arm_specs.items():
            primary_tokens: list[int] = []
            for gold_id, share in zip(gold_ids, shares):
                primary_tokens.extend(within_span(gold_id, share, selector))
            blocked = all_gold_tokens - set(primary_tokens)
            plan, meta = token_plan(seq_len, SINK, recent, retained_total, primary_tokens, blocked, [], filler_token_order)
            meta.update({"shares": shares, "gold_lengths": gold_lengths, "within_span_selector": selector,
                         "literal_all_gold_fit": bool(sum(gold_lengths) <= guidepost_budget)})
            plans[arm] = (plan, meta)

        lengths = {len(plan.retained_indices) for plan, _ in plans.values()} | {len(r) for r in (coverage_only_routes[k][0] for k in coverage_only_routes if k.endswith("exact512"))}
        if lengths != {retained_total}:
            raise RuntimeError(f"within-example byte mismatch for {sample_id}: {lengths}")

        # ---------------- rows
        route_record = {"id": sample_id, "prompt_tokens": seq_len, "retained_total": retained_total, "recent": recent,
                        "evidence_budget": evidence_budget, "gold_ids": gold_ids,
                        "gold_token_ranges": {g: [learned_index.by_id[g].token_start, learned_index.by_id[g].token_end] for g in gold_ids},
                        "routes": {}}
        for arm in DECODED_ARMS:
            plan, plan_metadata = plans[arm]
            retained_set = set(plan.retained_indices)
            cov = coverage_stats(retained_set, learned_index, gold_ids)
            row: dict[str, Any] = {
                "id": sample_id, "question_type": source["type"], "method": arm, "decoded": not args.skip_decode,
                "answer": str(source["answer"]), "prompt_tokens": seq_len,
                "retained_tokens": len(plan.retained_indices), "retained_ratio": plan.retained_ratio,
                "selector_seconds_shared": selector_seconds,
                "route_sha256": route_hash(plan.retained_indices), "broken_middle_applicable": bool(victims),
                "plan_metadata": plan_metadata, **cov,
            }
            if not args.skip_decode:
                torch.cuda.synchronize()
                gen_started = time.perf_counter()
                prediction, cache, _, warnings, generation_timing = production_compact_decode(
                    model, tokenizer, input_ids, plan.retained_indices, int(input_ids.shape[1]), args.max_new_tokens, 512
                )
                torch.cuda.synchronize()
                generation_seconds = time.perf_counter() - gen_started
                del cache
                _release_cuda()
                short = extract_short_answer(prediction)
                em, f1, contains = answer_scores(short, str(source["answer"]))
                row.update({"prediction": prediction, "short_prediction": short, "answer_em": em, "answer_f1": f1,
                            "answer_contains": contains, "generation_seconds": generation_seconds,
                            "generation_timing": generation_timing, "warnings": warnings})
            all_rows.append(row)
            append_jsonl(result_path, row)
            route_record["routes"][arm] = plan.retained_indices
        for arm, (retained, meta) in coverage_only_routes.items():
            retained_set = set(retained)
            cov = coverage_stats(retained_set, learned_index, gold_ids)
            row = {"id": sample_id, "question_type": source["type"], "method": arm, "decoded": False,
                   "answer": str(source["answer"]), "prompt_tokens": seq_len, "retained_tokens": len(retained),
                   "retained_ratio": len(retained) / seq_len, "route_sha256": route_hash(retained),
                   "plan_metadata": meta, **cov}
            all_rows.append(row)
            append_jsonl(result_path, row)
            route_record["routes"][arm] = retained
        append_jsonl(routes_path, route_record)
        print(f"[{position+1}/{len(selected_indices)}] {sample_id} gold={len(gold_ids)} seq={seq_len} "
              f"core_recall={plans['proof_core'][1]['primary_tokens_retained']} sel={selector_seconds:.2f}s", flush=True)

    # ---------------- aggregates
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[str(row["method"])].append(row)
    records = []
    for arm in ALL_ARMS:
        values = grouped[arm]
        record: dict[str, Any] = {
            "method": arm, "n": len(values), "decoded": all(v["decoded"] for v in values),
            "gold_token_recall": statistics.fmean(v["gold_token_recall"] for v in values),
            "gold_proof_complete": statistics.fmean(v["gold_proof_complete"] for v in values),
            "all_gold_spans_touched": statistics.fmean(v["all_gold_touched"] for v in values),
            "mean_gold_span_retention": statistics.fmean(v["mean_gold_span_retention"] for v in values),
            "absent_gold_span_distribution": dict(sorted(Counter(v["absent_gold_spans"] for v in values).items())),
            "absent_gold_span_mean": statistics.fmean(v["absent_gold_spans"] for v in values),
            "retained_tokens_min": min(v["retained_tokens"] for v in values),
            "retained_tokens_max": max(v["retained_tokens"] for v in values),
        }
        if record["decoded"]:
            record.update({
                "answer_em": statistics.fmean(v["answer_em"] for v in values),
                "answer_f1": statistics.fmean(v["answer_f1"] for v in values),
                "answer_contains": statistics.fmean(v["answer_contains"] for v in values),
                "generation_seconds_mean": statistics.fmean(v["generation_seconds"] for v in values),
            })
        records.append(record)

    frozen_records = {r["method"]: r for r in frozen["records"]}
    gate = {}
    for method in FROZEN_METHODS:
        ours = next(r for r in records if r["method"] == method)
        gate[method] = {
            "gold_token_recall": {"frozen": frozen_records[method]["gold_token_recall"], "recomputed": ours["gold_token_recall"]},
            "gold_proof_complete": {"frozen": frozen_records[method]["gold_proof_complete"], "recomputed": ours["gold_proof_complete"]},
        }
        if ours["decoded"]:
            gate[method]["answer_f1"] = {"frozen": frozen_records[method]["answer_f1"], "recomputed": ours["answer_f1"]}
            gate[method]["answer_em"] = {"frozen": frozen_records[method]["answer_em"], "recomputed": ours["answer_em"]}
    gate_passed = all(
        abs(gate[m]["gold_token_recall"]["frozen"] - gate[m]["gold_token_recall"]["recomputed"]) < 1e-9
        and abs(gate[m]["gold_proof_complete"]["frozen"] - gate[m]["gold_proof_complete"]["recomputed"]) < 1e-9
        for m in FROZEN_METHODS
    )
    payload: dict[str, Any] = {
        "protocol": manifest,
        "records": records,
        "route_identity_gate": {"per_method": gate, "passed_decoder_independent_aggregates": gate_passed},
    }
    if not args.skip_decode:
        by_id_method = {(str(r["id"]), str(r["method"])): r for r in all_rows}
        comparisons: dict[str, Any] = {}
        def compare(left_arm: str, right_arm: str, seed_tag: str) -> dict[str, Any]:
            left = [by_id_method[(sid, left_arm)]["answer_f1"] for sid in selected_ids]
            right = [by_id_method[(sid, right_arm)]["answer_f1"] for sid in selected_ids]
            deltas = [a - b for a, b in zip(left, right)]
            out = bootstrap_delta(left, right, args.bootstrap_samples, stable_seed(seed_tag, args.intervention_seed))
            out.update(sign_flip_pvalues(deltas, args.permutation_samples, 82000 + sum(map(ord, seed_tag))))
            out["same_route_count"] = sum(by_id_method[(sid, left_arm)]["route_sha256"] == by_id_method[(sid, right_arm)]["route_sha256"] for sid in selected_ids)
            return out
        # frozen comparisons (reproduction check)
        for opponent in ("relevance_same_bytes", "edge_shuffled", "random_edges", "broken_middle"):
            comparisons[f"proof_core_vs_{opponent}"] = compare("proof_core", opponent, opponent)
        comparisons["oracle_proof_vs_proof_core"] = compare("oracle_proof", "proof_core", "oracle_vs_core")
        # new arms vs proof core (delta = arm - proof_core), vs oracle, vs relevance
        for arm in NEW_DECODED_ARMS:
            comparisons[f"{arm}_vs_proof_core"] = compare(arm, "proof_core", f"{arm}_vs_core")
            comparisons[f"{arm}_vs_relevance_same_bytes"] = compare(arm, "relevance_same_bytes", f"{arm}_vs_rel")
            comparisons[f"oracle_proof_vs_{arm}"] = compare("oracle_proof", arm, f"oracle_vs_{arm}")
        payload["paired_bootstrap_f1"] = comparisons
        payload["frozen_paired_bootstrap_f1"] = frozen["paired_bootstrap_f1"]
        payload["integrity"] = {
            "rows_expected": len(selected_ids) * len(ALL_ARMS), "rows_observed": len(all_rows),
            "unique_pairs": len(by_id_method), "prior_id_overlap": len(overlap),
            "all_exactly_matched_tokens_decoded_arms": all(
                len({by_id_method[(sid, m)]["retained_tokens"] for m in DECODED_ARMS}) == 1 for sid in selected_ids),
            "extreme_zero_f1_rows": sum(r.get("answer_f1") == 0 for r in all_rows if r["decoded"]),
        }
    payload["wall_clock_seconds"] = time.time() - started_wall
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    manifest["finished_utc"] = payload["finished_utc"]
    manifest["wall_clock_seconds"] = payload["wall_clock_seconds"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"route_identity_gate": payload["route_identity_gate"], "records": records}, indent=2))
    if not gate_passed:
        print("ROUTE_IDENTITY_GATE_FAILED", flush=True)
        raise SystemExit(3)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
