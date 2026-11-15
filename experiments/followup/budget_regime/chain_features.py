#!/usr/bin/env python
"""Per-example chain features for the budget-regime analysis (tokenizer only): gold span count, total gold-span tokens, required hop count, and for every
budget whether a complete gold proof fits the evidence budget (adaptive_budget_parts: budget − 4 sink − query tail)."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
IMPL = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(IMPL / "scripts"))
from datasets import load_dataset
from transformers import AutoTokenizer
from run_hotpot_compact_qa import make_prompt, normalize_ircot_source
from eval_natural_route_coverage import adaptive_budget_parts, token_span
OUT = Path(__import__("os").environ.get("PROOFKV_OUT", "runs")) / "budget_regime"; BUDGETS = (64, 96, 128, 192, 256, 384, 512)
tok = AutoTokenizer.from_pretrained(__import__("os").environ.get("PROOFKV_DECODER", "Qwen/Qwen2.5-7B-Instruct"), use_fast=True)
for name, ds in (("musique", load_dataset("json", data_files={"v": __import__("os").path.join(__import__("os").environ.get("PROOFKV_IRCOT", "data/ircot/processed_data"), "musique/test_subsampled.jsonl")}, split="v")),
                 ("hotpot", load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))):
    order = list(range(len(ds))); random.Random(17).shuffle(order)
    mans = sorted((OUT / f"matrix_{'musique_v6' if name == 'musique' else 'hotpot_a090'}_n128").glob("fixed*/run_manifest.json"))
    for mp in mans:
        assert json.load(open(mp))["selected_indices"] == order[:128], mp
    with open(OUT / f"features_{name}.jsonl", "w") as f:
        for idx in order[:128]:
            raw = dict(ds[idx]); src = normalize_ircot_source(raw)
            prompt, sentences, query = make_prompt(src, tok)
            offs = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]; L = len(offs)
            gold = [token_span(offs, *s["char_span"]) for s in sentences if s["gold"]]
            gold_tokens = sum(b - a for a, b in gold)
            qs, _ = token_span(offs, len(prompt) - len(query), len(prompt))
            hop = int(src["type"][0]) if name == "musique" else 2
            row = {"id": str(src["id"]), "hop_count": hop, "question_type": raw.get("type", src["type"]), "gold_span_count": len(gold), "gold_tokens": gold_tokens,
                   "prompt_tokens": L, "span_count": len(sentences)}
            for B in BUDGETS:
                rt, rec, ev = adaptive_budget_parts(L, min(1.0, B / L), qs)
                row[f"evidence_budget_{B}"] = ev; row[f"gold_fits_{B}"] = gold_tokens <= ev
            f.write(json.dumps(row) + "\n")
    print(name, "features written")
