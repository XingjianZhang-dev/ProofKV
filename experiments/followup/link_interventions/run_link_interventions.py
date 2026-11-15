#!/usr/bin/env python
"""Equal-byte link interventions on a disjoint slice of MuSiQue, HotpotQA, or 2WikiMultiHopQA.

Identical selection / intervention / exact-byte plan / compact-prefill decode path as the frozen
scripts/run_natural_topology_causality.py (whose helpers are imported unmodified): frozen a090
scorer and policy, six arms (proof_core, relevance_same_bytes, edge_shuffled, random_edges,
broken_middle, oracle_proof), exactly --budget-tokens retained tokens per arm (or the full prompt
if shorter), greedy 24-token decode.  Only the data adapter and the explicit exclusion machinery
are new.

Datasets:
  musique        IRCoT-processed jsonl (whole paragraphs; 20/ex; 2-4 gold)          -> --data-file
  2wiki          IRCoT-processed jsonl (whole paragraphs; 10/ex; 2 or 4 gold)       -> --data-file
  hotpotqa_ircot IRCoT-processed jsonl (whole paragraphs; 10/ex; exactly 2 gold)    -> --data-file
  hotpotqa_hf    hotpotqa/hotpot_qa distractor validation (SENTENCE spans; ~42/ex;
                 2-7 gold sentences inside exactly 2 gold paragraphs) — the paper's own HotpotQA
                 downstream protocol and the scorer's HotpotQA training granularity.

Slice = seed-17 shuffle positions [--sample-offset, +--max-samples).  Exclusions are explicit:
--exclude-positions ranges on the same shuffle plus --exclude-id-files; every excluded id is
recorded in the manifest and zero overlap is asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

IMPL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(IMPL / "scripts"))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402

from proofkv.eval.generation import load_model_tokenizer  # noqa: E402
from proofkv.prefill_selectors import GuidepostPlan  # noqa: E402
from proofkv.proofcache.index import EvidenceIndexer, extract_cues, hashed_embedding  # noqa: E402
from proofkv.proofcache.model import TypedRelationScorer, build_learned_overlay  # noqa: E402
from proofkv.proofcache.schema import EvidenceSpan, PrecisionTier  # noqa: E402
from proofkv.proofcache.selector import AndOrProofSelector, ProofSearchConfig, TypedProofSearcher  # noqa: E402
from proofkv.utils.io import append_jsonl  # noqa: E402

from eval_natural_route_coverage import SINK, adaptive_budget_parts, token_span  # noqa: E402
from run_hotpot_compact_qa import (  # noqa: E402
    answer_scores,
    extract_short_answer,
    make_prompt,
    normalize_ircot_source,
    production_compact_decode,
)
from run_natural_topology_causality import (  # noqa: E402
    METHODS,
    bootstrap_delta,
    corrupt_index,
    exact_plan,
    file_sha256,
    select_core,
    stable_seed,
)
from run_physical_kv_replay import _release_cuda  # noqa: E402

DESTRUCTIVE_ARMS = ("edge_shuffled", "random_edges", "broken_middle")  # precommitted one-sided
HF_HOTPOT = ("hotpotqa/hotpot_qa", "distractor", "validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("musique", "2wiki", "hotpotqa_ircot", "hotpotqa_hf"))
    parser.add_argument("--data-file", default=None, help="IRCoT-processed jsonl (not used for hotpotqa_hf)")
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--sample-offset", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--exclude-positions", default="", help="comma list of a:b ranges on the same seed shuffle")
    parser.add_argument("--exclude-id-files", default="", help="comma list of jsonl files whose question_id/id are excluded")
    parser.add_argument("--exclusion-note", default="")
    parser.add_argument("--skip-excluded", action="store_true",
                        help="walk the shuffle from --sample-offset and take the first --max-samples ids NOT on the exclusion list (skips recorded)")
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--intervention-seed", type=int, default=42823318)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=100000)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--opened-after-submission", default="false", choices=("true", "false"))
    parser.add_argument("--opened-note", default="")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def route_hash(indices: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(indices), separators=(",", ":")).encode()).hexdigest()


def sign_flip_pvalues(values: list[float], samples: int, seed: int) -> dict[str, float]:
    """Paired sign-flip randomization test (frozen definition: one-sided H1 mean>0; plus two-sided)."""
    observed = statistics.fmean(values)
    rng = random.Random(seed)
    exceed_one = exceed_two = 0
    for _ in range(samples):
        value = statistics.fmean(item if rng.random() < 0.5 else -item for item in values)
        exceed_one += value >= observed - 1e-15
        exceed_two += abs(value) >= abs(observed) - 1e-15
    return {"p_one_sided_greater": (exceed_one + 1) / (samples + 1), "p_two_sided": (exceed_two + 1) / (samples + 1)}


def parse_ranges(text: str) -> list[tuple[int, int]]:
    out = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        a, b = piece.split(":")
        out.append((int(a), int(b)))
    return out


def load_rows(args: argparse.Namespace) -> tuple[Any, str, dict[str, Any]]:
    if args.dataset == "hotpotqa_hf":
        dataset = load_dataset(HF_HOTPOT[0], HF_HOTPOT[1], split=HF_HOTPOT[2])
        return dataset, "hotpotqa/hotpot_qa distractor validation (HF cache)", {
            "hf_fingerprint": getattr(dataset, "_fingerprint", None), "span_unit": "sentence",
        }
    if not args.data_file:
        raise ValueError(f"{args.dataset} needs --data-file")
    dataset = load_dataset("json", data_files={"validation": args.data_file}, split="validation")
    return dataset, args.data_file, {"dataset_sha256": file_sha256(Path(args.data_file)), "span_unit": "paragraph"}


def source_of(dataset: Any, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = dict(dataset[index])
    source = normalize_ircot_source(raw)
    extra = {"raw_type": raw.get("type"), "raw_level": raw.get("level")}
    return source, extra


def coverage(retained: set[int], index: Any, gold_ids: Sequence[str]) -> dict[str, Any]:
    per_span = []
    for gold_id in gold_ids:
        span = index.by_id[gold_id]
        tokens = set(range(span.token_start, span.token_end))
        per_span.append(len(tokens & retained) / max(1, len(tokens)))
    gold_tokens = {t for g in gold_ids for t in range(index.by_id[g].token_start, index.by_id[g].token_end)}
    touched = sum(v > 0 for v in per_span)
    return {
        "gold_token_recall": len(retained & gold_tokens) / max(1, len(gold_tokens)),
        "gold_proof_complete": float(all(v >= 1.0 for v in per_span)),
        "gold_spans_touched": touched,
        "all_gold_touched": float(touched == len(gold_ids)),
        "per_gold_span_retention": per_span,
    }


def main() -> None:
    args = parse_args()
    started_wall = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path, routes_path = out_dir / "results.jsonl", out_dir / "routes.jsonl"
    metrics_path, manifest_path = out_dir / "metrics.json", out_dir / "run_manifest.json"
    if any(p.exists() for p in (result_path, metrics_path, manifest_path, routes_path)):
        raise FileExistsError(out_dir)

    dataset, data_desc, data_meta = load_rows(args)
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    # ---- explicit exclusion list
    excluded: dict[str, str] = {}  # id -> source
    exclude_ranges = parse_ranges(args.exclude_positions)
    for a, b in exclude_ranges:
        for pos in range(a, min(b, len(order))):
            sid = str(source_of(dataset, order[pos])[0]["id"])
            excluded.setdefault(sid, f"seed{args.seed}-positions {a}:{b}")
    id_files = [p.strip() for p in args.exclude_id_files.split(",") if p.strip()]
    id_file_sha = {}
    for path in id_files:
        id_file_sha[path] = file_sha256(Path(path))
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("question_id", row.get("id")))
            excluded.setdefault(sid, f"ids of {path}")
    skipped: list[dict[str, Any]] = []
    if args.skip_excluded:
        selected_indices = []
        pos = args.sample_offset
        while len(selected_indices) < args.max_samples and pos < len(order):
            sid = str(source_of(dataset, order[pos])[0]["id"])
            if sid in excluded:
                skipped.append({"position": pos, "id": sid, "reason": excluded[sid]})
            else:
                selected_indices.append(order[pos])
            pos += 1
        last_position_walked = pos
    else:
        selected_indices = order[args.sample_offset : args.sample_offset + args.max_samples]
        last_position_walked = args.sample_offset + args.max_samples
    if len(selected_indices) != args.max_samples:
        raise ValueError("insufficient selected examples")
    selected_ids = [str(source_of(dataset, i)[0]["id"]) for i in selected_indices]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("duplicate selected ids")
    overlap = sorted(set(selected_ids) & set(excluded))
    if overlap:
        raise RuntimeError(f"slice overlaps exclusion list: {overlap[:5]}")

    code_files = {
        "experiments/followup/link_interventions/run_link_interventions.py": Path(__file__).resolve(),
        "scripts/run_natural_topology_causality.py": IMPL / "scripts/run_natural_topology_causality.py",
        "scripts/run_hotpot_compact_qa.py": IMPL / "scripts/run_hotpot_compact_qa.py",
        "scripts/eval_natural_route_coverage.py": IMPL / "scripts/eval_natural_route_coverage.py",
        "proofkv/proofcache/model.py": IMPL / "proofkv/proofcache/model.py",
        "proofkv/proofcache/selector.py": IMPL / "proofkv/proofcache/selector.py",
        "proofkv/proofcache/index.py": IMPL / "proofkv/proofcache/index.py",
        "proofkv/physical_kv.py": IMPL / "proofkv/physical_kv.py",
        "proofkv/prefill_selectors.py": IMPL / "proofkv/prefill_selectors.py",
    }
    implementation_sha256 = {k: file_sha256(p) for k, p in code_files.items()}
    scorer_dir = Path(args.scorer_checkpoint)
    scorer_sha256 = {str(p.relative_to(scorer_dir)): file_sha256(p) for p in sorted(scorer_dir.rglob("*")) if p.is_file()}
    import transformers

    manifest: dict[str, Any] = {
        "experiment_id": "link_interventions",
        "dataset": args.dataset,
        "data_source": data_desc,
        **data_meta,
        "arguments": {k: v for k, v in vars(args).items() if k != "out_dir"},
        "command_line": " ".join(sys.argv),
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": args.model, "model_revision": args.model_revision,
        "scorer_checkpoint": str(scorer_dir), "scorer_checkpoint_sha256": scorer_sha256,
        "scorer_checkpoint_identity": "proofkv_scorer_a090 (the checkpoint of the frozen MuSiQue topology run; v6 not involved)",
        "implementation_sha256": implementation_sha256,
        "software_package_manifest_sha256": file_sha256(IMPL / "MANIFEST.sha256"),
        "config_hash": sha256_text(json.dumps({"impl": implementation_sha256, "scorer": scorer_sha256, "args": vars(args)}, sort_keys=True)),
        "methods": list(METHODS),
        "selected_indices": selected_indices, "selected_ids": selected_ids,
        "selection_rule": ("shuffle walked from --sample-offset skipping excluded ids" if args.skip_excluded else "contiguous shuffle positions"),
        "shuffle_positions_walked": [args.sample_offset, last_position_walked],
        "skipped_excluded": skipped,
        "exclusion": {
            "note": args.exclusion_note,
            "position_ranges_on_same_shuffle": exclude_ranges,
            "id_files": id_file_sha,
            "excluded_unique_ids": len(excluded),
            "excluded_ids_by_source": dict(Counter(excluded.values())),
            "excluded_ids": sorted(excluded),
            "overlap_count": len(overlap),
        },
        "opened_after_submission": args.opened_after_submission == "true",
        "opened_after_submission_note": args.opened_note,
        "frozen_policy": {
            "root_top_k": 16, "beam_size": 8, "max_hops": 8, "max_proofs": 8, "normalize_path_probability": False,
            "candidate_span_count": 32, "candidate_strategy": "root", "dense_core_count": 4, "edge_batch_size": 256,
            "compact_prefill_chunk_size": 512, "fill_auxiliary": False, "padding": "frozen exact_plan: hash-ordered whole filler spans then positional; gold-agnostic",
        },
        "statistics": "10,000-sample paired bootstrap percentile CI (frozen bootstrap_delta, frozen seeds) and 100,000-replicate paired sign-flip test; one-sided (H1: core > arm) precommitted for edge_shuffled/random_edges/broken_middle, two-sided reported for all",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_wall)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    model, tokenizer = load_model_tokenizer(args.model, dtype="bfloat16", attn_implementation=args.attn_implementation)
    device = next(model.parameters()).device
    scorer, scorer_tokenizer = TypedRelationScorer.load_checkpoint(args.scorer_checkpoint, device=device)
    scorer.eval()
    search_config = ProofSearchConfig(root_top_k=16, beam_size=8, max_hops=8, max_proofs=8, normalize_path_probability=False)
    searcher = TypedProofSearcher(search_config)

    all_rows: list[dict[str, Any]] = []
    for position, dataset_index in enumerate(selected_indices):
        source, extra = source_of(dataset, dataset_index)
        sample_id = str(source["id"])
        question = str(source["question"])
        prompt, sentences, query = make_prompt(source, tokenizer)
        encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        offsets = [(int(a), int(b)) for a, b in encoding.pop("offset_mapping")[0].tolist()]
        input_ids = encoding["input_ids"].to(device)
        seq_len = len(offsets)
        sentence_spans = [token_span(offsets, *s["char_span"]) for s in sentences]
        gold_ids = [f"s{idx}" for idx, s in enumerate(sentences) if s["gold"]]
        spans = [
            EvidenceSpan(
                f"s{idx}", start, end, f"[{s['title']}] {s['text']}",
                cues=extract_cues(f"{s['title']} {s['text']}"),
                embedding=hashed_embedding(f"{s['title']} {s['text']}", 64),
                metadata={"title": s["title"], "sentence_index": s["sent_id"]},
            )
            for idx, ((start, end), s) in enumerate(zip(sentence_spans, sentences))
        ]
        index = EvidenceIndexer().build(sample_id, spans, seq_len)
        query_start, _ = token_span(offsets, len(prompt) - len(query), len(prompt))
        retained_total, recent, evidence_budget = adaptive_budget_parts(seq_len, min(1.0, args.budget_tokens / seq_len), query_start)

        torch.cuda.synchronize()
        selector_started = time.perf_counter()
        with torch.inference_mode():
            overlay, learned_index = build_learned_overlay(
                scorer, scorer_tokenizer, question, index, candidate_span_count=32, search_config=search_config,
                candidate_strategy="root", dense_core_count=4, edge_batch_size=256,
            )
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
            compositional = next((p for p in overlay.proofs if len(p.span_ids) >= 2), None)
            if compositional is not None:
                victims.add(compositional.span_ids[len(compositional.span_ids) // 2])
        broken_ids = tuple(s for s in proof_ids if s not in victims)
        filler_ids = tuple(sorted((s.span_id for s in learned_index.spans), key=lambda v: hashlib.sha256(f"{sample_id}:{v}".encode()).hexdigest()))
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
            plans[method] = exact_plan(seq_len, SINK, recent, retained_total, learned_index, ids, filler_ids, blocked)
        lengths = {len(plan.retained_indices) for plan, _ in plans.values()}
        if lengths != {retained_total}:
            raise RuntimeError(f"within-example byte mismatch for {sample_id}: {lengths}")
        degenerate = seq_len <= args.budget_tokens  # full prompt retained: all six arms coincide by construction
        gold_token_total = sum(learned_index.by_id[g].token_end - learned_index.by_id[g].token_start for g in gold_ids)
        route_record = {"id": sample_id, "seq_len": seq_len, "retained_total": retained_total, "routes": {}}

        method_order = list(METHODS)
        random.Random(sample_seed + 2).shuffle(method_order)
        for method in method_order:
            plan, plan_metadata = plans[method]
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction, cache, _, warnings, generation_timing = production_compact_decode(
                model, tokenizer, input_ids, plan.retained_indices, int(input_ids.shape[1]), args.max_new_tokens, 512,
            )
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - started
            del cache
            _release_cuda()
            short = extract_short_answer(prediction)
            em, f1, contains = answer_scores(short, str(source["answer"]))
            retained_set = set(plan.retained_indices)
            row = {
                "id": sample_id, "dataset": args.dataset, "question_type": source["type"], **extra,
                "method": method, "answer": str(source["answer"]), "prediction": prediction, "short_prediction": short,
                "answer_em": em, "answer_f1": f1, "answer_contains": contains,
                "prompt_tokens": seq_len, "retained_tokens": len(plan.retained_indices), "retained_ratio": plan.retained_ratio,
                "degenerate_full_prompt": degenerate,
                "gold_span_count": len(gold_ids), "gold_token_total": gold_token_total,
                "gold_fits_evidence_budget": gold_token_total <= evidence_budget, "evidence_budget": evidence_budget,
                "span_count": len(spans), "mean_span_tokens": seq_len / max(1, len(spans)),
                **coverage(retained_set, learned_index, gold_ids),
                "primary_span_ids": list(plan_metadata["primary_span_ids"]),
                "selector_seconds_shared": selector_seconds, "generation_seconds": generation_seconds,
                "generation_timing": generation_timing, "route_sha256": route_hash(plan.retained_indices),
                "broken_middle_applicable": bool(victims), "plan_metadata": plan_metadata, "warnings": warnings,
            }
            all_rows.append(row)
            append_jsonl(result_path, row)
            route_record["routes"][method] = list(plan.retained_indices)
        append_jsonl(routes_path, route_record)
        print(f"[{position+1}/{len(selected_indices)}] {sample_id} gold={len(gold_ids)} seq={seq_len} degenerate={degenerate} sel={selector_seconds:.2f}s", flush=True)

    # ---------------- aggregates and statistics
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["method"]].append(row)
    records = []
    for method in METHODS:
        v = grouped[method]
        records.append({
            "method": method, "n": len(v),
            "answer_em": statistics.fmean(r["answer_em"] for r in v), "answer_f1": statistics.fmean(r["answer_f1"] for r in v),
            "answer_contains": statistics.fmean(r["answer_contains"] for r in v),
            "gold_token_recall": statistics.fmean(r["gold_token_recall"] for r in v),
            "gold_proof_complete": statistics.fmean(r["gold_proof_complete"] for r in v),
            "all_gold_spans_touched": statistics.fmean(r["all_gold_touched"] for r in v),
            "retained_tokens_min": min(r["retained_tokens"] for r in v), "retained_tokens_max": max(r["retained_tokens"] for r in v),
            "generation_seconds_mean": statistics.fmean(r["generation_seconds"] for r in v),
        })
    by_id_method = {(r["id"], r["method"]): r for r in all_rows}

    def compare(left_arm: str, right_arm: str, seed_tag: str, ids: list[str], seed_offset: int = 0) -> dict[str, Any]:
        left = [by_id_method[(sid, left_arm)]["answer_f1"] for sid in ids]
        right = [by_id_method[(sid, right_arm)]["answer_f1"] for sid in ids]
        deltas = [a - b for a, b in zip(left, right)]
        out = bootstrap_delta(left, right, args.bootstrap_samples, stable_seed(seed_tag, args.intervention_seed) + seed_offset)
        out.update(sign_flip_pvalues(deltas, args.permutation_samples, 82000 + sum(map(ord, seed_tag)) + seed_offset))
        out.update({
            "n": len(ids), "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas), "ties": sum(d == 0 for d in deltas),
            "same_route_count": sum(by_id_method[(sid, left_arm)]["route_sha256"] == by_id_method[(sid, right_arm)]["route_sha256"] for sid in ids),
        })
        return out

    def comparison_block(ids: list[str], seed_offset: int = 0) -> dict[str, Any]:
        block = {}
        for opponent in ("relevance_same_bytes", "edge_shuffled", "random_edges", "broken_middle"):
            block[f"proof_core_vs_{opponent}"] = compare("proof_core", opponent, opponent, ids, seed_offset)
        block["oracle_proof_vs_proof_core"] = compare("oracle_proof", "proof_core", "oracle_vs_core", ids, seed_offset)
        return block

    comparisons = {"all": comparison_block(selected_ids)}
    non_degenerate = [sid for sid in selected_ids if not by_id_method[(sid, "proof_core")]["degenerate_full_prompt"]]
    if len(non_degenerate) != len(selected_ids) and len(non_degenerate) >= 2:
        comparisons["non_degenerate_prompt_gt_budget"] = comparison_block(non_degenerate, 1)
    subgroups: dict[str, Any] = {}
    for key in ("gold_span_count", "raw_type", "question_type"):
        groups: dict[str, list[str]] = defaultdict(list)
        for sid in selected_ids:
            groups[str(by_id_method[(sid, "proof_core")][key])].append(sid)
        if len(groups) <= 1:
            continue
        subgroups[key] = {}
        for value, ids in sorted(groups.items()):
            if len(ids) < 8:
                subgroups[key][value] = {"n": len(ids), "skipped": "fewer than 8 examples"}
                continue
            entry = {"n": len(ids), "f1_by_arm": {m: statistics.fmean(by_id_method[(sid, m)]["answer_f1"] for sid in ids) for m in METHODS}}
            entry.update({f"proof_core_vs_{opp}": compare("proof_core", opp, opp, ids, 7 + int(hashlib.sha256(f"{key}:{value}".encode()).hexdigest()[:6], 16) % 1000) for opp in ("relevance_same_bytes",) + DESTRUCTIVE_ARMS})
            subgroups[key][value] = entry
    edge_gate = comparisons["all"]["proof_core_vs_edge_shuffled"]
    integrity = {
        "rows_expected": len(selected_ids) * len(METHODS), "rows_observed": len(all_rows), "unique_pairs": len(by_id_method),
        "prior_id_overlap": len(overlap),
        "all_exactly_matched_tokens": all(len({by_id_method[(sid, m)]["retained_tokens"] for m in METHODS}) == 1 for sid in selected_ids),
        "degenerate_full_prompt_examples": len(selected_ids) - len(non_degenerate),
        "broken_middle_applicable_rate": statistics.fmean(float(by_id_method[(sid, "broken_middle")]["broken_middle_applicable"]) for sid in selected_ids),
        "gold_fits_evidence_budget_rate": statistics.fmean(float(by_id_method[(sid, "proof_core")]["gold_fits_evidence_budget"]) for sid in selected_ids),
        "gold_span_count_distribution": dict(sorted(Counter(by_id_method[(sid, "proof_core")]["gold_span_count"] for sid in selected_ids).items())),
        "span_count_mean": statistics.fmean(by_id_method[(sid, "proof_core")]["span_count"] for sid in selected_ids),
        "prompt_tokens_mean": statistics.fmean(by_id_method[(sid, "proof_core")]["prompt_tokens"] for sid in selected_ids),
        "gold_token_total_mean": statistics.fmean(by_id_method[(sid, "proof_core")]["gold_token_total"] for sid in selected_ids),
        "extreme_zero_f1_rows": sum(r["answer_f1"] == 0 for r in all_rows),
    }
    integrity["passed"] = (
        integrity["rows_expected"] == integrity["rows_observed"] == integrity["unique_pairs"]
        and integrity["prior_id_overlap"] == 0 and integrity["all_exactly_matched_tokens"]
    )
    payload = {
        "protocol": manifest, "records": records, "paired_bootstrap_f1": comparisons, "subgroups": subgroups, "integrity": integrity,
        "primary_causal_gate": {"delta_at_least_0_03": edge_gate["delta"] >= 0.03, "ci95_low_above_zero": edge_gate["ci95_low"] > 0,
                                "passed": edge_gate["delta"] >= 0.03 and edge_gate["ci95_low"] > 0},
        "wall_clock_seconds": time.time() - started_wall, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    manifest["finished_utc"] = payload["finished_utc"]
    manifest["wall_clock_seconds"] = payload["wall_clock_seconds"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"records": records, "paired_bootstrap_f1": comparisons["all"], "integrity": integrity}, indent=2))
    if not integrity["passed"]:
        raise SystemExit(2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
