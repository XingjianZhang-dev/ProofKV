#!/usr/bin/env python
"""Decomposition of the gap between ProofKV and the oracle on the held-out MuSiQue selector test.

Selector-only re-run under the locked MuSiQue selector protocol of
scripts/eval_proofcache_selector.py (same prompt rendering, tokenizer offsets, index,
frozen scorer, frozen policy, same budget accounting), with per-example instrumentation:
  - root / terminal scores for all spans, the 32-span candidate set
  - every candidate pair proposed for edge scoring, with the full 9-way softmax
  - the learned typed relation graph after the 0.35 floor
  - the beam-search trace (per-hop beams) and the final proofs
  - the selected routes for learned_single_path / learned_topk_paths / learned_and_or /
    learned_and_or_aux and the oracle, at every budget
  - the gold proof set and the gold directed pairs from the official MuSiQue
    question_decomposition (#k references -> paragraph_support_idx)
and a mutually exclusive failure attribution per (example, budget, method).

Span granularity: whole MuSiQue paragraph (20 per example).  Nothing in the frozen
scorer, policy or compiler is modified; the traced searcher is asserted to reproduce
the frozen searcher's proofs exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import re
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
from transformers import AutoTokenizer  # noqa: E402

from proofkv.proofcache.datasets import from_musique  # noqa: E402
from proofkv.proofcache.index import EvidenceIndexer, extract_cues, hashed_embedding  # noqa: E402
from proofkv.proofcache.model import (  # noqa: E402
    RELATION_LABELS,
    TypedRelationScorer,
    score_roots,
    score_terminals,
    tokenize_edge_batch,
)
from proofkv.proofcache.schema import EvidenceIndex, EvidenceSpan, PrecisionTier, Relation, RelationType  # noqa: E402
from proofkv.proofcache.selector import AndOrProofSelector, ProofSearchConfig, TypedProofSearcher  # noqa: E402
from proofkv.utils.io import append_jsonl  # noqa: E402
from eval_proofcache_selector import (  # noqa: E402
    SINK,
    bm25_priority,
    dense_priority,
    graph_similarity_priority,
    parse_budget,
    proof_priority,
    reranker_priority,
    select_by_priority,
    selector_metrics,
    token_span,
)
from run_natural_topology_causality import file_sha256  # noqa: E402

EDGE_FLOOR = 0.35
BASELINE_METHODS = ("bm25", "reranker_proxy", "graph_similarity_proxy", "dense_hash_proxy")
METHODS = ("learned_single_path", "learned_topk_paths", "learned_and_or", "learned_and_or_aux", "oracle_proof") + BASELINE_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--official-dev", required=True, help="musique_ans_v1.0_dev.jsonl for question_decomposition")
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--model", required=True, help="decoder path (tokenizer only)")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--budgets", default="256,512")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--opened-after-submission", action="store_true")
    parser.add_argument("--protocol-note", default="")
    parser.add_argument("--frozen-reference", default="")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


# ---------------------------------------------------------------- instrumented overlay
@torch.inference_mode()
def score_edges_full(model, tokenizer, question, pairs, batch_size=256, max_length=512):
    """score_edges with the full softmax returned (argmax/prob identical to the frozen function)."""
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        encoded = tokenize_edge_batch(tokenizer, [question] * len(batch_pairs), [s for s, _ in batch_pairs],
                                      [t for _, t in batch_pairs], max_length, edge_encoding=getattr(model, "edge_encoding", "single"))
        encoded = {k: v.to(device) for k, v in encoded.items()}
        probabilities = torch.softmax(model.relation_logits(encoded), dim=-1).cpu()
        values, labels = probabilities.max(dim=-1)
        for row, label, prob in zip(probabilities.tolist(), labels.tolist(), values.tolist()):
            out.append((None if label == 0 else RelationType(RELATION_LABELS[label]), float(prob), [float(x) for x in row]))
    return out


def build_overlay_instrumented(model, tokenizer, question, index, search_config, candidate_span_count=32,
                               dense_core_count=4, edge_batch_size=256):
    """Copy of build_learned_overlay(candidate_strategy='root') that also returns the trace."""
    spans = list(index.spans)
    with torch.inference_mode():
        root_probabilities = score_roots(model, tokenizer, question, [s.text for s in spans])
        terminal_probabilities = score_terminals(model, tokenizer, question, [s.text for s in spans])
    root_scores = {s.span_id: p for s, p in zip(spans, root_probabilities)}
    terminal_scores = {s.span_id: p for s, p in zip(spans, terminal_probabilities)}
    candidate_ids = sorted(root_scores, key=lambda sid: (-root_scores[sid], sid))[:candidate_span_count]
    candidate_set = set(candidate_ids)
    by_id = index.by_id
    pair_ids: list[tuple[str, str]] = []
    pair_origin: dict[tuple[str, str], str] = {}
    for relation in index.relations:
        if relation.source in candidate_set and relation.target in candidate_set:
            pair_ids.append((relation.source, relation.target))
            pair_origin.setdefault((relation.source, relation.target), "index")
    dense_core = candidate_ids[: min(dense_core_count, len(candidate_ids))]
    for s in dense_core:
        for t in dense_core:
            if s != t:
                pair_ids.append((s, t))
                pair_origin.setdefault((s, t), "dense_core")
    pair_ids = list(dict.fromkeys(pair_ids))
    predictions = score_edges_full(model, tokenizer, question, [(by_id[s].text, by_id[t].text) for s, t in pair_ids], batch_size=edge_batch_size)
    relations = [Relation(s, t, rtype, prob) for (s, t), (rtype, prob, _) in zip(pair_ids, predictions)
                 if rtype is not None and prob >= EDGE_FLOOR]
    learned_index = EvidenceIndex(index.document_id, index.spans, tuple(relations), index.token_count,
                                  index_version=f"{index.index_version}+learned-relations")
    overlay = TypedProofSearcher(search_config).search(question, learned_index, root_scores=root_scores, terminal_scores=terminal_scores)
    trace = {
        "root_scores": root_scores, "terminal_scores": terminal_scores, "candidate_ids": candidate_ids,
        "dense_core_ids": dense_core,
        "scored_pairs": [{"source": s, "target": t, "origin": pair_origin[(s, t)], "type": (rtype.value if rtype else "none"),
                          "probability": prob, "p_none": row[0], "softmax": row}
                         for (s, t), (rtype, prob, row) in zip(pair_ids, predictions)],
        "learned_relations": [{"source": r.source, "target": r.target, "type": r.relation_type.value, "probability": r.probability} for r in relations],
    }
    return overlay, learned_index, trace


def traced_search(config: ProofSearchConfig, query: str, index: EvidenceIndex, root_scores, terminal_scores):
    """Copy of TypedProofSearcher.search that records the per-hop beams. Output is asserted equal."""
    roots = sorted(root_scores, key=lambda sid: (-root_scores[sid], sid))[: config.root_top_k]
    adjacency = index.adjacency
    beam = [(max(1e-6, root_scores[r]), (r,), (), root_scores[r]) for r in roots]
    hops: list[list[dict[str, Any]]] = []
    query_has_negation = bool(re.search(r"\b(?:not|never|false|incorrect|except)\b", query, re.I))
    candidates_count = 0
    for hop in range(config.max_hops):
        expanded = []
        for probability, span_ids, relations, root_score in beam:
            for relation in adjacency.get(span_ids[-1], ()):
                if relation.probability < config.min_relation_probability or relation.target in span_ids:
                    continue
                edge_probability = relation.probability
                if relation.relation_type == RelationType.CONTRADICTION and not query_has_negation:
                    edge_probability *= config.contradiction_penalty
                expanded.append((probability * edge_probability * math.exp(-config.length_penalty),
                                 span_ids + (relation.target,), relations + (relation,), root_score))
        if not expanded:
            hops.append([])
            break
        expanded.sort(key=lambda row: (-row[0], row[1]))
        beam = expanded[: config.beam_size]
        hops.append([{"probability": p, "span_ids": list(sids), "terminal": terminal_scores.get(sids[-1], 0.0)} for p, sids, _, _ in beam])
        candidates_count += sum(terminal_scores.get(sids[-1], 0.0) >= config.terminal_probability_floor for _, sids, _, _ in beam)
    return {"roots": roots, "hops": hops, "query_has_negation": query_has_negation, "terminal_passing_beam_rows": candidates_count}


# ---------------------------------------------------------------- gold structure
def gold_directed_pairs(official_row: dict[str, Any]) -> tuple[list[tuple[str, str]], list[str], str]:
    """Directed dependency pairs from the official decomposition: hop j references #i => p_i -> p_j."""
    decomposition = official_row["question_decomposition"]
    support = [f"p{int(step['paragraph_support_idx'])}" for step in decomposition]
    pairs: list[tuple[str, str]] = []
    for j, step in enumerate(decomposition):
        for ref in re.findall(r"#(\d+)", str(step["question"])):
            i = int(ref) - 1
            if 0 <= i < len(support) and support[i] != support[j]:
                pairs.append((support[i], support[j]))
    # structure label: chain if every hop j>0 references exactly hop j-1
    is_chain = all(re.findall(r"#(\d+)", str(step["question"])) == [str(j)] for j, step in enumerate(decomposition) if j > 0) and \
        not re.findall(r"#(\d+)", str(decomposition[0]["question"]))
    return list(dict.fromkeys(pairs)), support, ("chain" if is_chain else "branching")


def hamiltonian_path_exists(nodes: Sequence[str], edges: set[tuple[str, str]], directed: bool,
                            start_ok=None, end_ok=None) -> bool:
    nodes = list(nodes)
    if len(nodes) <= 1:
        return True
    for perm in itertools.permutations(nodes):
        if start_ok is not None and not start_ok(perm[0]):
            continue
        if end_ok is not None and not end_ok(perm[-1]):
            continue
        ok = True
        for a, b in zip(perm, perm[1:]):
            if (a, b) in edges or (not directed and (b, a) in edges):
                continue
            ok = False
            break
        if ok:
            return True
    return False


def attribute(gold: set[str], trace: dict[str, Any], search_trace: dict[str, Any], overlay, selected: set[str],
              oracle_complete: bool, config: ProofSearchConfig) -> dict[str, Any]:
    """Mutually exclusive failure cause for one (example, budget, method) route."""
    result: dict[str, Any] = {"complete": gold <= selected}
    if result["complete"]:
        result["cause"] = "success"
        return result
    if not oracle_complete:
        result["cause"] = "iv_budget_infeasible"
    candidate_set = set(trace["candidate_ids"])
    result["gold_in_candidates"] = gold <= candidate_set
    scored = {(p["source"], p["target"]) for p in trace["scored_pairs"]}
    learned = {(r["source"], r["target"]): r for r in trace["learned_relations"]}
    learned_edges = set(learned)
    penalized = {(s, t) for (s, t), r in learned.items() if r["type"] == "contradiction" and not search_trace["query_has_negation"]}
    roots = set(search_trace["roots"])
    terminal_ok = lambda sid: trace["terminal_scores"].get(sid, 0.0) >= config.terminal_probability_floor
    result["proposed_chain_exists"] = hamiltonian_path_exists(gold, scored, directed=False)
    result["undirected_learned_chain_exists"] = hamiltonian_path_exists(gold, learned_edges, directed=False)
    result["directed_learned_chain_exists"] = hamiltonian_path_exists(gold, learned_edges, directed=True)
    result["directed_unpenalized_chain_exists"] = hamiltonian_path_exists(gold, learned_edges - penalized, directed=True)
    result["directed_chain_from_root_to_terminal_exists"] = hamiltonian_path_exists(
        gold, learned_edges - penalized, directed=True, start_ok=lambda s: s in roots, end_ok=terminal_ok)
    result["gold_root_in_top_k"] = any(g in roots for g in gold)
    result["gold_terminal_passes_floor"] = any(terminal_ok(g) for g in gold)
    complete_found = [p for p in overlay.proofs if gold <= set(p.span_ids)]
    exact_found = [p for p in complete_found if set(p.span_ids) == gold]
    result["complete_proof_found"] = bool(complete_found)
    result["exact_gold_proof_found"] = bool(exact_found)
    result["beam_reached_gold_set"] = any(gold <= set(row["span_ids"]) for hop in search_trace["hops"] for row in hop)
    if "cause" in result:
        return result
    if not result["gold_in_candidates"]:
        result["cause"] = "i_candidate_span_miss"
    elif not result["proposed_chain_exists"]:
        result["cause"] = "ii_candidate_relation_miss"
    elif not result["undirected_learned_chain_exists"]:
        result["cause"] = "iii_a_edge_mis_scored"
    elif not result["directed_learned_chain_exists"]:
        result["cause"] = "iii_b_wrong_direction"
    elif not result["directed_unpenalized_chain_exists"]:
        result["cause"] = "iii_c_wrong_relation_type"
    elif not result["complete_proof_found"]:
        if not result["directed_chain_from_root_to_terminal_exists"]:
            result["cause"] = "iii_d_search_root_or_terminal_gate"
        else:
            result["cause"] = "iii_d_search_beam_pruned"
    else:
        result["cause"] = "iii_e_found_but_not_selected"
    return result


# ---------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    started_wall = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path, trace_path, metrics_path, manifest_path = (out_dir / "results.jsonl", out_dir / "traces.jsonl",
                                                            out_dir / "metrics.json", out_dir / "run_manifest.json")
    if any(p.exists() for p in (result_path, trace_path, metrics_path, manifest_path)):
        raise FileExistsError(out_dir)
    budget_specs = [b.strip() for b in args.budgets.split(",") if b.strip()]

    dataset = load_dataset("json", data_files={args.split: args.data_file}, split=args.split)
    rows = [dataset[i] for i in range(len(dataset))]
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    examples = [from_musique(rows[i], args.split) for i in order[: args.max_samples]]
    official = {}
    for line in Path(args.official_dev).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            official[r["id"]] = r
    missing = [e.sample_id for e in examples if e.sample_id not in official]
    if missing:
        raise RuntimeError(f"{len(missing)} examples missing from the official decomposition file")

    scorer_dir = Path(args.scorer_checkpoint)
    code_files = {
        "experiments/followup/heldout_selector/oracle_gap.py": Path(__file__).resolve(),
        "scripts/eval_proofcache_selector.py": IMPL / "scripts/eval_proofcache_selector.py",
        "proofkv/proofcache/model.py": IMPL / "proofkv/proofcache/model.py",
        "proofkv/proofcache/selector.py": IMPL / "proofkv/proofcache/selector.py",
        "proofkv/proofcache/index.py": IMPL / "proofkv/proofcache/index.py",
        "proofkv/proofcache/datasets.py": IMPL / "proofkv/proofcache/datasets.py",
        "proofkv/proofcache/schema.py": IMPL / "proofkv/proofcache/schema.py",
    }
    import transformers
    manifest = {
        "experiment_id": "heldout_selector/oracle_gap",
        "purpose": "Oracle-gap error decomposition of the locked MuSiQue selector protocol with per-example candidate/relation/beam instrumentation",
        "span_granularity": "whole MuSiQue paragraph (20 per example); gold proof = official supporting paragraphs; directed gold pairs from official question_decomposition (#k references)",
        "arguments": {k: v for k, v in vars(args).items() if k != "out_dir"},
        "command_line": " ".join(sys.argv), "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scorer_checkpoint": str(scorer_dir),
        "scorer_checkpoint_sha256": {str(p.relative_to(scorer_dir)): file_sha256(p) for p in sorted(scorer_dir.rglob("*")) if p.is_file()},
        "implementation_sha256": {k: file_sha256(p) for k, p in code_files.items()},
        "dataset_sha256": file_sha256(Path(args.data_file)), "official_dev_sha256": file_sha256(Path(args.official_dev)),
        "frozen_policy": {"root_top_k": 16, "beam_size": 8, "max_hops": 8, "max_proofs": 8, "normalize_path_probability": False,
                          "candidate_span_count": 32, "candidate_strategy": "root", "dense_core_count": 4, "edge_batch_size": 256,
                          "edge_probability_floor": EDGE_FLOOR},
        "selected_ids": [e.sample_id for e in examples],
        "opened_after_submission": bool(args.opened_after_submission),
        "protocol_note": args.protocol_note, "frozen_reference": args.frozen_reference,
        "attribution_precedence": ["iv_budget_infeasible (oracle route incomplete; independent of the checkpoint)",
                                   "i_candidate_span_miss", "ii_candidate_relation_miss (no chain over gold spans among scored pairs, either direction)",
                                   "iii_a_edge_mis_scored (no chain over gold in the learned graph even ignoring direction)",
                                   "iii_b_wrong_direction (undirected chain exists, no directed chain)",
                                   "iii_c_wrong_relation_type (directed chain exists only through penalized contradiction edges)",
                                   "iii_d_search_root_or_terminal_gate / iii_d_search_beam_pruned (directed chain exists, no found proof covers gold)",
                                   "iii_e_found_but_not_selected (a found proof covers gold; selected route does not)"],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_wall)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer, scorer_tokenizer = TypedRelationScorer.load_checkpoint(args.scorer_checkpoint, device=device)
    scorer.eval()
    config = ProofSearchConfig(root_top_k=16, beam_size=8, max_hops=8, max_proofs=8, normalize_path_probability=False)
    indexer = EvidenceIndexer()
    kv_bytes_per_token = args.num_layers * 2 * args.num_kv_heads * args.head_dim * 2
    values_per_token = args.num_layers * 2 * args.num_kv_heads * args.head_dim
    all_rows: list[dict[str, Any]] = []

    for position, example in enumerate(examples):
        context, char_spans = example.render_context()
        query = f"\nQuestion: {example.question}\nAnswer:"
        prompt = context + query
        encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [(int(a), int(b)) for a, b in encoding["offset_mapping"]]
        token_spans = [token_span(offsets, *cs) for cs in char_spans]
        evidence_spans = [
            EvidenceSpan(span_id=s.span_id, token_start=a, token_end=b, text=f"[{s.title}] {s.text}", kind=s.kind,
                         cues=extract_cues(f"{s.title} {s.text}"), embedding=hashed_embedding(f"{s.title} {s.text}", 64),
                         metadata={**s.metadata, "title": s.title})
            for s, (a, b) in zip(example.spans, token_spans)]
        index = indexer.build(example.sample_id, evidence_spans, len(offsets))
        by_id = index.by_id
        query_start, _ = token_span(offsets, len(context), len(prompt))
        query_tokens = set(range(query_start, len(offsets)))
        sink_tokens = set(range(min(SINK, len(offsets))))
        gold = set(example.gold_span_ids)
        pairs, hop_support, structure = gold_directed_pairs(official[example.sample_id])
        if set(hop_support) != gold:
            raise RuntimeError(f"{example.sample_id}: decomposition support {hop_support} != gold {sorted(gold)}")

        t0 = time.perf_counter()
        overlay, learned_index, trace = build_overlay_instrumented(scorer, scorer_tokenizer, example.question, index, config)
        selector_ms = (time.perf_counter() - t0) * 1000
        search_trace = traced_search(config, example.question, learned_index, trace["root_scores"], trace["terminal_scores"])
        reference = TypedProofSearcher(config).search(example.question, learned_index, root_scores=trace["root_scores"], terminal_scores=trace["terminal_scores"])
        if [p.span_ids for p in reference.proofs] != [p.span_ids for p in overlay.proofs]:
            raise RuntimeError("frozen searcher is not deterministic")
        gold_pairs_status = []
        learned = {(r["source"], r["target"]): r for r in trace["learned_relations"]}
        scored = {(p["source"], p["target"]): p for p in trace["scored_pairs"]}
        for s, t in pairs:
            gold_pairs_status.append({"source": s, "target": t,
                                      "proposed_forward": (s, t) in scored, "proposed_reverse": (t, s) in scored,
                                      "learned_forward": (s, t) in learned, "learned_reverse": (t, s) in learned,
                                      "forward_type": learned.get((s, t), {}).get("type"), "forward_p_none": scored.get((s, t), {}).get("p_none")})
        trace_row = {"id": example.sample_id, "question_type": example.question_type, "structure": structure,
                     "gold": sorted(gold), "gold_directed_pairs": pairs, "hop_support": hop_support,
                     "gold_token_count": sum(by_id[g].token_count for g in gold), "prompt_tokens": len(offsets),
                     **trace, "search": search_trace,
                     "proofs": [{"proof_id": p.proof_id, "span_ids": list(p.span_ids), "probability": p.probability,
                                 "relations": [{"source": r.source, "target": r.target, "type": r.relation_type.value, "probability": r.probability} for r in p.relations]}
                                for p in overlay.proofs],
                     "gold_pairs_status": gold_pairs_status, "selector_ms": selector_ms}
        append_jsonl(trace_path, trace_row)

        for budget_input in budget_specs:
            budget_spec, total_budget = parse_budget(budget_input, len(offsets))
            mandatory = query_tokens | sink_tokens
            evidence_budget = max(0, total_budget - len(mandatory))
            selections: dict[str, tuple[tuple[str, ...], tuple[str, ...] | None]] = {}
            compositional = [p for p in overlay.proofs if len(p.span_ids) >= 2]
            priority = list((compositional or list(overlay.proofs))[0].span_ids) if overlay.proofs else []
            selections["learned_single_path"] = (select_by_priority(priority, by_id, evidence_budget), None)
            selections["learned_topk_paths"] = (select_by_priority(proof_priority(overlay), by_id, evidence_budget), None)
            for method in ("learned_and_or", "learned_and_or_aux"):
                sel = AndOrProofSelector(fill_auxiliary=method.endswith("_aux")).select(
                    overlay, learned_index, byte_budget=evidence_budget * kv_bytes_per_token,
                    values_per_token=values_per_token, precision=PrecisionTier.BF16)
                selections[method] = (sel.selected_span_ids + sel.auxiliary_span_ids, sel.selected_span_ids)
            gold_proofs = sorted(example.gold_proofs, key=lambda proof: (sum(by_id[x].token_count for x in proof), sorted(proof)))
            oracle_priority = sorted(gold_proofs[0], key=lambda sid: by_id[sid].token_start) if gold_proofs else []
            selections["oracle_proof"] = (select_by_priority(oracle_priority, by_id, evidence_budget), None)
            oracle_complete = gold <= set(selections["oracle_proof"][0])
            # span-atomic lexical/proxy baselines exactly as in eval_proofcache_selector.py (no candidate graph)
            selections["bm25"] = (select_by_priority(bm25_priority(example.question, index.spans), by_id, evidence_budget), None)
            selections["reranker_proxy"] = (select_by_priority(reranker_priority(example.question, index.spans), by_id, evidence_budget), None)
            selections["graph_similarity_proxy"] = (select_by_priority(graph_similarity_priority(example.question, index.spans), by_id, evidence_budget), None)
            selections["dense_hash_proxy"] = (select_by_priority(dense_priority(example.question, index.spans), by_id, evidence_budget), None)
            for method in METHODS:
                selected_ids, core_ids = selections[method]
                retained = mandatory | {t for sid in selected_ids for t in range(by_id[sid].token_start, by_id[sid].token_end)}
                metrics = selector_metrics(example, selected_ids, by_id, retained, len(offsets), kv_bytes_per_token)
                metrics["core_pcr"] = metrics["pcr"] if core_ids is None else float(any(proof <= set(core_ids) for proof in example.gold_proofs))
                if method in BASELINE_METHODS:
                    attribution = {"complete": gold <= set(selected_ids), "cause": "success" if gold <= set(selected_ids) else "n/a_baseline_no_candidate_graph"}
                else:
                    attribution = attribute(gold, trace, search_trace, overlay, set(selected_ids), oracle_complete, config)
                row = {"id": example.sample_id, "question_type": example.question_type, "structure": structure, "method": method,
                       "budget_spec": budget_spec, "total_budget": total_budget, "evidence_budget": evidence_budget,
                       "selected_span_ids": list(selected_ids), "gold": sorted(gold), "gold_token_count": trace_row["gold_token_count"],
                       "oracle_complete": oracle_complete, **metrics, "attribution": attribution,
                       "selector_ms": selector_ms, "index_build_ms": 0.0}
                all_rows.append(row)
                append_jsonl(result_path, row)
        print(f"[{position+1}/{len(examples)}] {example.sample_id} {example.question_type} {structure} gold_tokens={trace_row['gold_token_count']} "
              f"cands_have_gold={gold <= set(trace['candidate_ids'])} proofs={len(overlay.proofs)} sel={selector_ms:.0f}ms", flush=True)

    # ---------------- aggregates
    records = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["method"], row["budget_spec"])].append(row)
    for (method, budget), values in sorted(grouped.items()):
        rec = {"method": method, "budget_spec": budget, "n": len(values)}
        for metric in ("pcr", "core_pcr", "complete_proofs", "route_precision", "gold_token_recall", "retained_ratio", "actual_kv_bytes"):
            rec[metric] = statistics.fmean(float(v[metric]) for v in values)
        causes = Counter(v["attribution"]["cause"] for v in values)
        rec["failure_causes_count"] = dict(sorted(causes.items()))
        rec["failure_causes_pct_of_n"] = {k: 100 * c / len(values) for k, c in sorted(causes.items())}
        non_budget = [v for v in values if v["attribution"]["cause"] not in ("success", "iv_budget_infeasible")]
        rec["non_budget_gap_pct"] = 100 * len(non_budget) / len(values)
        rec["non_budget_gap_breakdown_pct_of_n"] = {k: 100 * c / len(values) for k, c in sorted(Counter(v["attribution"]["cause"] for v in non_budget).items())}
        rec["budget_infeasible_pct"] = 100 * causes.get("iv_budget_infeasible", 0) / len(values)
        rec["infeasible_and_candidate_miss_overlap_pct"] = 100 * sum(1 for v in values if v["attribution"]["cause"] == "iv_budget_infeasible" and not v["attribution"].get("gold_in_candidates", True)) / len(values)
        by_type: dict[str, dict[str, Any]] = {}
        for qtype in sorted({v["question_type"] for v in values}):
            sub = [v for v in values if v["question_type"] == qtype]
            by_type[qtype] = {"n": len(sub), "pcr": statistics.fmean(v["pcr"] for v in sub),
                              "causes_pct": {k: 100 * c / len(sub) for k, c in sorted(Counter(v["attribution"]["cause"] for v in sub).items())}}
        rec["by_question_type"] = by_type
        records.append(rec)
    payload = {"protocol": manifest, "records": records,
               "integrity": {"rows_expected": len(examples) * len(budget_specs) * len(METHODS), "rows_observed": len(all_rows)},
               "wall_clock_seconds": time.time() - started_wall, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    manifest.update({"finished_utc": payload["finished_utc"], "wall_clock_seconds": payload["wall_clock_seconds"]})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for rec in records:
        print(rec["method"], rec["budget_spec"], "PCR", round(rec["pcr"], 4), "causes", rec["failure_causes_pct_of_n"])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
