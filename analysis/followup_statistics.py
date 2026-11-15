#!/usr/bin/env python3
"""Recompute every follow-up statistic quoted in the paper from the released per-example records.

Usage:
    python analysis/followup_statistics.py results/followup results/followup/summary.json

The resampling routines and seeds are the ones used when the experiments were first analysed, so the
intervals printed here are the intervals reported in the paper. Everything is derived from
``results.jsonl`` files; no number is typed in by hand.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 42823318


# --------------------------------------------------------------------------- resampling
def stable_seed(tag: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{tag}".encode()).digest()[:8], "big")


def bootstrap_py(left, right, samples, seed):
    """Paired percentile bootstrap used by the equal-byte intervention runner."""
    deltas = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return statistics.fmean(deltas), means[max(0, int(0.025 * samples) - 1)], means[min(samples - 1, int(0.975 * samples))]


def sign_flip_np(deltas, samples, seed):
    d = np.asarray(deltas, float)
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(samples, len(d))) * d).mean(axis=1)
    return float((np.sum(np.abs(null) >= abs(d.mean()) - 1e-15) + 1) / (samples + 1))


def paired_np(left, right, seed):
    """Paired bootstrap + sign-flip with one generator (budget-regime and distractor analyses)."""
    d = np.asarray(left, float) - np.asarray(right, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    boots = np.sort(d[rng.integers(0, n, size=(10000, n))].mean(axis=1))
    lo, hi = boots[max(0, int(0.025 * 10000) - 1)], boots[min(9999, int(0.975 * 10000))]
    null = (d * rng.choice([-1.0, 1.0], size=(100000, n))).mean(axis=1)
    p2 = (np.sum(np.abs(null) >= abs(d.mean()) - 1e-15) + 1) / 100001
    return {"delta": 100 * float(d.mean()), "lo": 100 * float(lo), "hi": 100 * float(hi), "p": float(p2), "n": int(n)}


def paired_np_e5(left, right, seed):
    d = np.asarray(left, float) - np.asarray(right, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    b = np.sort(d[rng.integers(0, n, size=(10000, n))].mean(axis=1))
    null = (d * rng.choice([-1.0, 1.0], size=(100000, n))).mean(axis=1)
    p2 = (np.sum(np.abs(null) >= abs(d.mean()) - 1e-15) + 1) / 100001
    return {"delta": 100 * float(d.mean()), "lo": 100 * float(b[248]), "hi": 100 * float(b[9750]), "p": float(p2), "n": int(n)}


def paired_np_split(deltas, boot_seed=1, flip_seed=2):
    d = np.asarray(deltas, float)
    rng = np.random.default_rng(boot_seed)
    means = np.sort(d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1))
    return {
        "delta": 100 * float(d.mean()),
        "lo": 100 * float(means[max(0, int(0.025 * 10000) - 1)]),
        "hi": 100 * float(means[min(9999, int(0.975 * 10000))]),
        "p": sign_flip_np(d, 100000, flip_seed),
        "n": int(len(d)),
    }


def load(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def mean100(values):
    return 100 * statistics.fmean(values)


# --------------------------------------------------------------------------- experiments
def premise_test(root: Path) -> dict:
    """Deployed compressors vs the chain core on the MuSiQue intervention slice (512 tokens)."""
    out = {"compressors": {}, "chain": {}}
    rows = load(root / "premise_test" / "compressors" / "results.jsonl")
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    for method, v in by.items():
        out["compressors"][method] = {
            "n": len(v),
            "f1": mean100(r["answer_f1"] for r in v),
            "em": mean100(r["answer_em"] for r in v),
            "all_gold_spans_touched": mean100(r["coverage_head_union"]["all_gold_touched"] for r in v),
            "gold_token_recall": mean100(r["coverage_head_union"]["gold_token_recall"] for r in v),
            "chain_complete": mean100(r["coverage_head_union"]["gold_proof_complete"] for r in v),
            "union_fraction_of_context": mean100(r["union_fraction_of_context"] for r in v),
        }
    rows = [r for r in load(root / "premise_test" / "chain_and_oracles" / "results.jsonl") if "answer_f1" in r]
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][r["id"]] = r
    ids = sorted(by["proof_core"])
    for method in ("proof_core", "oracle_proof", "uniform_partial_f50_scorer"):
        v = [by[method][i] for i in ids]
        out["chain"][method] = {
            "n": len(v),
            "f1": mean100(r["answer_f1"] for r in v),
            "chain_complete": mean100(r["gold_proof_complete"] for r in v),
            "gold_token_recall": mean100(r["gold_token_recall"] for r in v),
        }
    half = [by["uniform_partial_f50_scorer"][i]["answer_f1"] for i in ids]
    chain = [by["proof_core"][i]["answer_f1"] for i in ids]
    delta, lo, hi = bootstrap_py(half, chain, 10000, stable_seed("uniform_partial_f50_scorer_vs_core", SEED))
    out["chain"]["half_span_oracle_minus_chain"] = {
        "delta": 100 * delta, "lo": 100 * lo, "hi": 100 * hi, "n": len(ids),
        "p": sign_flip_np([a - b for a, b in zip(half, chain)], 100000, 82000 + sum(map(ord, "uniform_partial_f50_scorer_vs_core"))),
    }
    return out


INTERVENTION_RUNS = {
    "musique_slice": ("musique_pos128_256", "MuSiQue (intervention slice)", "paragraph"),
    "musique_more": ("musique_pos256_500", "MuSiQue (further 244)", "paragraph"),
    "hotpot_paragraph": ("hotpotqa_paragraph", "HotpotQA", "paragraph"),
    "hotpot_sentence": ("hotpotqa_sentence", "HotpotQA", "sentence"),
    "twowiki": ("2wikimultihopqa", "2WikiMultiHopQA", "paragraph"),
}
ARMS = ("relevance_same_bytes", "edge_shuffled", "random_edges", "broken_middle")


def interventions(root: Path) -> dict:
    out = {}
    loaded = {}
    for key, (folder, label, unit) in INTERVENTION_RUNS.items():
        rows = load(root / "link_interventions" / folder / "results.jsonl")
        by = {(r["id"], r["method"]): r for r in rows}
        ids = list(dict.fromkeys(r["id"] for r in rows))
        loaded[key] = (by, ids)
        out[key] = intervention_block(by, ids, 0)
        out[key].update(label=label, span_unit=unit)
        applicable = [i for i in ids if by[(i, "broken_middle")]["broken_middle_applicable"]]
        out[key]["chains_with_two_or_more_spans"] = len(applicable)
    by = {**loaded["musique_slice"][0], **loaded["musique_more"][0]}
    ids = loaded["musique_slice"][1] + loaded["musique_more"][1]
    out["musique_pooled"] = intervention_block(by, ids, 0)
    out["musique_pooled"].update(label="MuSiQue (pooled)", span_unit="paragraph")
    out["attention_backends"] = {}
    for name in ("sdpa_flash", "sdpa_math", "eager"):
        rows = load(root / "link_interventions" / "attention_backends" / f"rows_{name}.jsonl")
        f1 = {(r["id"], r["method"]): r["answer_f1"] for r in rows}
        bids = list(dict.fromkeys(r["id"] for r in rows))
        out["attention_backends"][name] = {
            arm: 100 * statistics.fmean(f1[(i, "proof_core")] - f1[(i, arm)] for i in bids)
            for arm in ("edge_shuffled", "random_edges", "broken_middle")
        }
    return out


def intervention_block(by, ids, seed_offset):
    core = [by[(i, "proof_core")]["answer_f1"] for i in ids]
    block = {
        "n": len(ids),
        "f1": {m: mean100(by[(i, m)]["answer_f1"] for i in ids) for m in ("proof_core",) + ARMS + ("oracle_proof",)},
        "chain_complete": {m: mean100(by[(i, m)]["gold_proof_complete"] for i in ids) for m in ("proof_core",) + ARMS + ("oracle_proof",)},
        "loss": {},
    }
    for arm in ARMS:
        other = [by[(i, arm)]["answer_f1"] for i in ids]
        delta, lo, hi = bootstrap_py(core, other, 10000, stable_seed(arm, SEED) + seed_offset)
        deltas = [a - b for a, b in zip(core, other)]
        block["loss"][arm] = {
            "delta": 100 * delta, "lo": 100 * lo, "hi": 100 * hi,
            "p": sign_flip_np(deltas, 100000, 82000 + sum(map(ord, arm)) + seed_offset), "n": len(ids),
        }
    return block


def heldout_selector(root: Path) -> dict:
    """Held-out MuSiQue selector test (n=500): chain-complete recall and the oracle-gap decomposition."""
    out = {"pcr": {}, "by_hop_512": {}, "paired_512": {}, "paired_256": {}, "gap_512": {}}
    rows = load(root / "heldout_selector" / "oracle_gap" / "results.jsonl")
    rows += load(root / "heldout_selector" / "iterative_retrieval" / "results.jsonl")
    table = defaultdict(dict)
    for r in rows:
        table[(r["method"], r["budget_spec"])][r["id"]] = r
    wanted = {
        "graph_similarity_proxy": "similarity_graph", "reranker_proxy": "hybrid_reranker", "bm25": "bm25",
        "root_topk": "same_scorer_independent", "learned_single_path": "single_chain", "learned_topk_paths": "chains",
        "learned_and_or": "and_or", "iter_k1_h4_hop": "iterative_hop_order", "iter_k1_h4_score": "iterative_score_order",
        "oracle_proof": "oracle",
    }
    for method, name in wanted.items():
        out["pcr"][name] = {b: mean100(r["pcr"] for r in table[(method, b)].values()) for b in ("256", "512")}
        hops = defaultdict(list)
        for r in table[(method, "512")].values():
            hops[r["question_type"]].append(r["pcr"])
        out["by_hop_512"][name] = {k: {"pcr": mean100(v), "n": len(v)} for k, v in sorted(hops.items())}
    ids = sorted(table[("learned_topk_paths", "512")])
    for budget, slot in (("512", "paired_512"), ("256", "paired_256")):
        chains = [table[("learned_topk_paths", budget)][i]["pcr"] for i in ids]
        for method, name in (("reranker_proxy", "hybrid_reranker"), ("root_topk", "same_scorer_independent"), ("iter_k1_h4_hop", "iterative_hop_order")):
            other = [table[(method, budget)][i]["pcr"] for i in ids]
            out[slot][f"chains_minus_{name}"] = paired_np_split([a - b for a, b in zip(chains, other)])
    causes = Counter()
    for r in table[("learned_topk_paths", "512")].values():
        a = r.get("attribution") or {}
        causes["complete" if a.get("complete") else a["cause"]] += 1
    out["gap_512"] = {k: 100 * v / len(ids) for k, v in sorted(causes.items())}
    out["n"] = len(ids)
    return out


BUDGETS = (64, 96, 128, 192, 256, 384, 512)
GEN = {
    "chains": "learned_topk_paths",
    "iterative_hop_order": "learned_iterative_retrieval_k1_h4_hoporder_tokenfill",
    "iterative_score_order": "learned_iterative_retrieval_k1_h4_tokenfill",
    "same_scorer_independent": "learned_root_topk_tokenfill",
}


def budget_regime(root: Path) -> dict:
    feats = {r["id"]: r for r in load(root / "budget_regime" / "features_musique.jsonl")}
    out = {"hop_counts": dict(Counter(f["hop_count"] for f in feats.values())),
           "gold_tokens_mean": statistics.fmean(f["gold_tokens"] for f in feats.values()), "budgets": {}}
    for budget in BUDGETS:
        rows = load(root / "budget_regime" / "musique" / f"fixed{budget}" / "results.jsonl")
        by = {(r["id"], r["method"]): r for r in rows}
        ids = [i for i in feats if all((i, m) in by for m in GEN.values())]
        cell = {"n": len(ids), "f1": {k: mean100(by[(i, m)]["answer_f1"] for i in ids) for k, m in GEN.items()},
                "gold_fits_pct": mean100(feats[i][f"gold_fits_{budget}"] for i in ids), "gain": {}, "splits": {}}
        chains = [by[(i, GEN["chains"])]["answer_f1"] for i in ids]
        for arm in ("iterative_hop_order", "iterative_score_order", "same_scorer_independent"):
            cell["gain"][arm] = paired_np(chains, [by[(i, GEN[arm])]["answer_f1"] for i in ids], SEED + budget)
        cell["gain_score_order_resample_b"] = paired_np_split(
            [by[(i, GEN["chains"])]["answer_f1"] - by[(i, GEN["iterative_score_order"])]["answer_f1"] for i in ids])
        groups = defaultdict(list)
        for i in ids:
            groups["chain_fits" if feats[i][f"gold_fits_{budget}"] else "chain_exceeds_budget"].append(i)
            groups[f"hop{feats[i]['hop_count']}"].append(i)
        for name, gi in sorted(groups.items()):
            if len(gi) >= 8:
                cell["splits"][name] = paired_np([by[(i, GEN["chains"])]["answer_f1"] for i in gi],
                                                 [by[(i, GEN["iterative_hop_order"])]["answer_f1"] for i in gi], SEED + budget + 7)
        out["budgets"][str(budget)] = cell
    return out


def distractors(root: Path) -> dict:
    out = {}
    for pool, folder in (("tfidf_hard", "tfidf_hard"), ("same_entity", "same_entity")):
        out[pool] = {}
        for target in (8000, 16000, 32000):
            rows = load(root / "related_distractors" / folder / f"hotpotqa_target{target}" / "results.jsonl")
            by = {(r["id"], r["method"]): r for r in rows}
            ids = sorted({r["id"] for r in rows})
            f = lambda m, k="answer_f1": mean100(by[(i, m)][k] for i in ids)
            cell = {"n": len(ids), "full_f1": f("full"), "entity_f1": f("entity_graph_adaptive"), "chains_f1": f("learned_topk_paths"),
                    "chains_kv_pct": f("learned_topk_paths", "retained_ratio"), "oracle_f1": f("oracle_adaptive")}
            cell["chains_minus_full"] = paired_np_e5([by[(i, "learned_topk_paths")]["answer_f1"] for i in ids],
                                                     [by[(i, "full")]["answer_f1"] for i in ids], 4242 + target)
            out[pool][str(target)] = cell
    return out


def main() -> None:
    root, target = Path(sys.argv[1]), Path(sys.argv[2])
    summary = {
        "premise_test": premise_test(root),
        "link_interventions": interventions(root),
        "heldout_selector": heldout_selector(root),
        "budget_regime": budget_regime(root),
        "related_distractors": distractors(root),
    }
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    li = summary["link_interventions"]
    print("link deletion (F1 points):")
    for key in ("musique_slice", "musique_pooled", "hotpot_paragraph", "hotpot_sentence", "twowiki"):
        s = li[key]["loss"]["broken_middle"]
        print(f"  {key:18s} n={s['n']:3d} {s['delta']:+5.1f} [{s['lo']:+5.1f}, {s['hi']:+5.1f}]")
    hs = summary["heldout_selector"]
    print("held-out PCR@512:", {k: round(v["512"], 1) for k, v in hs["pcr"].items()})
    print("wrote", target)


if __name__ == "__main__":
    main()
