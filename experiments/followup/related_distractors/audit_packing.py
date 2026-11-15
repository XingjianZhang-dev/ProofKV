#!/usr/bin/env python
"""Packing audit for the related-distractor contexts (the checks of the packing audit table in the paper): utilization (>= 0.90 of target and never above),
answer leak (no ADDED paragraph matches the packer's normalized-answer pattern), duplicate page (unique titles), question
identity (id / question / answer / all original paragraphs unchanged vs the HF row), tokenizer (re-tokenized packed prompt
== recorded actual_prompt_tokens <= target).  Plus relatedness diagnostics: mean TF-IDF cosine(question, added page) for
each pool and for the frozen unrelated packing reconstructed with the frozen packer and the same seeds."""
from __future__ import annotations
import json, random, re, statistics, sys
from pathlib import Path
IMPL = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(IMPL / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset
from transformers import AutoTokenizer
import run_hotpot_compact_qa as rq
from run_related_matrix import TfidfIndex
OUT = Path(__import__("os").environ.get("PROOFKV_OUT", "runs")) / "related_distractors"; MIN_UTIL = 0.9
tok = AutoTokenizer.from_pretrained(__import__("os").environ.get("PROOFKV_DECODER", "Qwen/Qwen2.5-7B-Instruct"), use_fast=True)
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"); order = list(range(len(ds))); random.Random(17).shuffle(order)
hf = {ds[i]["id"]: dict(ds[i]) for i in order[:128]}
pool = rq.build_natural_distractor_pool(ds); tfidf = TfidfIndex(pool)
pool_by_title = {}
for e in pool: pool_by_title.setdefault(e["title"], e)
out = {"minimum_utilization": MIN_UTIL, "packing_seed": 911, "seed": 17, "n": 128, "pools": {}, "issues": []}
for kind in ("HARD_TFIDF", "SAME_ENTITY"):
    for T in (8000, 16000, 32000):
        p = OUT / kind / f"hotpotqa_target{T}" / "packed_contexts.jsonl"
        if not p.exists():
            continue
        recs = []; issues = 0; sims = []
        for l in open(p):
            r = json.loads(l); src = hf[r["id"]]; iss = []
            pk = r["packing"]; util = pk["actual_prompt_tokens"] / T
            if util < MIN_UTIL or pk["actual_prompt_tokens"] > T: iss.append(f"utilization {util:.3f}")
            titles = r["context"]["title"]
            if len(titles) != len(set(titles)): iss.append("duplicate page title")
            orig = set(src["context"]["title"])
            if r["question"] != src["question"] or r["answer"] != src["answer"]: iss.append("question identity")
            orig_pages = {t: [s for s in ss] for t, ss in zip(src["context"]["title"], src["context"]["sentences"])}
            for t, ss in zip(titles, r["context"]["sentences"]):
                if t in orig_pages and orig_pages[t] != ss: iss.append(f"original paragraph altered: {t}")
            if not orig <= set(titles): iss.append("original paragraph missing")
            answer = rq.normalize(r["answer"]); pat = re.compile(rf"(?:^|\s){re.escape(answer)}(?:\s|$)") if len(answer) >= 2 else None
            added = [(t, ss) for t, ss in zip(titles, r["context"]["sentences"]) if t not in orig]
            if pat is not None and any(pat.search(rq.normalize(" ".join(ss))) for _, ss in added): iss.append("answer leak in added paragraph")
            packed_src = {**src, "context": r["context"]}
            prompt, _, _ = rq.make_prompt(packed_src, tok); n_tok = len(tok(prompt, add_special_tokens=False)["input_ids"])
            if n_tok != pk["actual_prompt_tokens"] or n_tok > T: iss.append(f"tokenizer {n_tok} vs {pk['actual_prompt_tokens']}")
            qs = tfidf.scores(r["question"]); sims.append(statistics.fmean(qs.get(pool.index(pool_by_title[t]), 0.0) if t in pool_by_title else 0.0 for t, _ in added) if added else 0.0)
            recs.append({"id": r["id"], "utilization": util, "added_pages": pk["added_pages"], "related_pages_admitted": pk.get("related_pages_admitted"),
                         "fallback_unrelated_pages": pk.get("fallback_unrelated_pages"), "answer_leak_candidates_skipped": pk["answer_leak_candidates_skipped"], "issues": iss})
            issues += bool(iss)
        # frozen unrelated packing, reconstructed for the same questions with the frozen packer (same packing seed)
        u_sims = []
        for r in recs[:32]:
            src = rq.normalize_ircot_source(hf[r["id"]]); packed, meta = rq.pack_source_to_prompt_target(src, tok, pool, T, 911, False)
            orig = set(hf[r["id"]]["context"]["title"]); added = [t for t in packed["context"]["title"] if t not in orig]; qs = tfidf.scores(src["question"])
            u_sims.append(statistics.fmean(qs.get(pool.index(pool_by_title[t]), 0.0) for t in added) if added else 0.0)
        summ = {"n": len(recs), "utilization_min": min(x["utilization"] for x in recs), "utilization_median": statistics.median(x["utilization"] for x in recs),
                "utilization_max": max(x["utilization"] for x in recs), "added_pages_median": statistics.median(x["added_pages"] for x in recs),
                "samples_with_issues": issues, "mean_tfidf_cosine_added_pages": statistics.fmean(sims),
                "mean_tfidf_cosine_added_pages_FROZEN_UNRELATED_reconstructed_first32": statistics.fmean(u_sims),
                "related_pages_admitted_mean": statistics.fmean(x["related_pages_admitted"] or 0 for x in recs), "fallback_unrelated_pages_mean": statistics.fmean(x["fallback_unrelated_pages"] or 0 for x in recs),
                "answer_leak_candidates_skipped_mean": statistics.fmean(x["answer_leak_candidates_skipped"] for x in recs), "records": recs}
        out["pools"].setdefault(kind, {})[str(T)] = summ; out["issues"] += [f"{kind}/{T}/{x['id']}: {x['issues']}" for x in recs if x["issues"]]
        print(kind, T, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items() if k != "records"}, flush=True)
(OUT / "packing_audit.json").write_text(json.dumps(out, indent=2) + "\n")
print("AUDIT issues:", len(out["issues"]))
