#!/usr/bin/env python
"""Iterative retrieval on the held-out MuSiQue selector test (protocol of scripts/eval_proofcache_selector.py): PCR at 256/512
for iterative retrieval (k in {1,2,4} x hops in {1,2,3,4}, score-order and hop-order unions) and the
same-scorer root top-k control, next to the frozen typed-paths / AND-OR / oracle records."""
from __future__ import annotations
import argparse, json, platform, random, statistics, sys, time
from collections import defaultdict
from pathlib import Path
IMPL = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(IMPL / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from proofkv.proofcache.datasets import from_musique
from proofkv.proofcache.index import EvidenceIndexer, extract_cues, hashed_embedding
from proofkv.proofcache.model import TypedRelationScorer, score_roots
from proofkv.proofcache.schema import EvidenceSpan
from proofkv.utils.io import append_jsonl
from eval_proofcache_selector import SINK, parse_budget, select_by_priority, selector_metrics, token_span
from run_natural_topology_causality import file_sha256
from iterative_retrieval import iterative_retrieval_priority

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--scorer-checkpoint", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--max-samples", type=int, default=1000); ap.add_argument("--budgets", default="256,512"); ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--ks", default="1,2,4"); ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--frozen-reference", default=""); ap.add_argument("--protocol-note", default=""); ap.add_argument("--opened-after-submission", action="store_true")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    t0 = time.time(); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for f in ("results.jsonl", "traces.jsonl", "metrics.json", "run_manifest.json"):
        if (out / f).exists(): raise FileExistsError(out / f)
    dataset = load_dataset("json", data_files={args.split: args.data_file}, split=args.split)
    rows = [dataset[i] for i in range(len(dataset))]; order = list(range(len(rows))); random.Random(args.seed).shuffle(order)
    examples = [from_musique(rows[i], args.split) for i in order[: args.max_samples]]
    ks = [int(x) for x in args.ks.split(",")]; budgets = [b.strip() for b in args.budgets.split(",")]
    scorer_dir = Path(args.scorer_checkpoint)
    import transformers
    manifest = {"experiment_id": "heldout_selector/iterative_retrieval", "arm": "iterative retrieval + root_topk control",
                "purpose": "structure without a learned graph — multi-hop retrieval with the package encoder's root head, no typed relations, no completeness objective",
                "definition": Path(__file__).resolve().parent.joinpath("iterative_retrieval.py").read_text().split('"""')[1],
                "arguments": {k: v for k, v in vars(args).items() if k != "out_dir"}, "command_line": " ".join(sys.argv),
                "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                "scorer_checkpoint": str(scorer_dir), "scorer_checkpoint_sha256": {str(p.relative_to(scorer_dir)): file_sha256(p) for p in sorted(scorer_dir.rglob("*")) if p.is_file()},
                "dataset_sha256": file_sha256(Path(args.data_file)), "selected_ids": [e.sample_id for e in examples],
                "opened_after_submission": bool(args.opened_after_submission), "protocol_note": args.protocol_note, "frozen_reference": args.frozen_reference,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    scorer, stok = TypedRelationScorer.load_checkpoint(args.scorer_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"); scorer.eval()
    indexer = EvidenceIndexer(); kv_bytes = 28 * 2 * 4 * 128 * 2
    all_rows = []
    for pos, ex in enumerate(examples):
        context, char_spans = ex.render_context(); query = f"\nQuestion: {ex.question}\nAnswer:"; prompt = context + query
        enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True); offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        spans = [EvidenceSpan(span_id=s.span_id, token_start=a, token_end=b, text=f"[{s.title}] {s.text}", kind=s.kind, cues=extract_cues(f"{s.title} {s.text}"),
                              embedding=hashed_embedding(f"{s.title} {s.text}", 64), metadata={**s.metadata, "title": s.title})
                 for s, (a, b) in zip(ex.spans, [token_span(offsets, *cs) for cs in char_spans])]
        index = indexer.build(ex.sample_id, spans, len(offsets)); by_id = index.by_id
        qs, _ = token_span(offsets, len(context), len(prompt)); mandatory = set(range(qs, len(offsets))) | set(range(min(SINK, len(offsets))))
        t1 = time.perf_counter()
        with torch.inference_mode():
            root = score_roots(scorer, stok, ex.question, [s.text for s in index.spans])
        root_order = [i for _, i in sorted(zip(root, [s.span_id for s in index.spans]), key=lambda t: (-t[0], t[1]))]
        priorities = {"root_topk": root_order}; traces = {}
        for k in ks:
            score_order, tr = iterative_retrieval_priority(scorer, stok, ex.question, index.spans, k=k, max_hops=args.max_hops)
            traces[f"k{k}"] = tr
            for h in range(1, args.max_hops + 1):   # hop-truncated unions come from the same trace
                sub = {i for i, hp in tr["retrieved_hops"].items() if hp <= h}
                priorities[f"iter_k{k}_h{h}_score"] = [i for i in tr["score_order"] if i in sub]
                priorities[f"iter_k{k}_h{h}_hop"] = [i for i in tr["hop_order"] if i in sub]
        ms = (time.perf_counter() - t1) * 1000
        append_jsonl(out / "traces.jsonl", {"id": ex.sample_id, "question_type": ex.question_type, "gold": sorted(ex.gold_span_ids), "root_order": root_order, "iterative": traces})
        for b in budgets:
            spec, total = parse_budget(b, len(offsets)); ebudget = max(0, total - len(mandatory))
            for name, pr in priorities.items():
                sel = select_by_priority(pr, by_id, ebudget)
                retained = mandatory | {t for sid in sel for t in range(by_id[sid].token_start, by_id[sid].token_end)}
                m = selector_metrics(ex, sel, by_id, retained, len(offsets), kv_bytes); m["core_pcr"] = m["pcr"]
                row = {"id": ex.sample_id, "question_type": ex.question_type, "method": name, "budget_spec": spec, "selected_span_ids": list(sel), "gold": sorted(ex.gold_span_ids), **m, "selector_ms": ms, "index_build_ms": 0.0}
                all_rows.append(row); append_jsonl(out / "results.jsonl", row)
        if (pos + 1) % 50 == 0: print(f"[{pos+1}/{len(examples)}] {time.time()-t0:.0f}s", flush=True)
    grouped = defaultdict(list)
    for r in all_rows: grouped[(r["method"], r["budget_spec"])].append(r)
    records = []
    for (m, b), v in sorted(grouped.items()):
        rec = {"method": m, "budget_spec": b, "n": len(v)}
        for key in ("pcr", "complete_proofs", "route_precision", "gold_token_recall", "retained_ratio", "actual_kv_bytes"): rec[key] = statistics.fmean(float(r[key]) for r in v)
        rec["pcr_by_type"] = {t: statistics.fmean(r["pcr"] for r in v if r["question_type"] == t) for t in sorted({r["question_type"] for r in v})}
        records.append(rec)
    payload = {"protocol": manifest, "records": records, "integrity": {"rows_expected": len(examples) * len(budgets) * len(priorities), "rows_observed": len(all_rows)},
               "wall_clock_seconds": time.time() - t0, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    manifest.update({"finished_utc": payload["finished_utc"], "wall_clock_seconds": payload["wall_clock_seconds"]}); (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for r in records: print(r["method"], r["budget_spec"], "PCR", round(100 * r["pcr"], 1))
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
