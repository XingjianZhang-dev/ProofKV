#!/usr/bin/env python
"""Per-gold-span coverage of token-level KV-eviction baselines.

Runs SnapKV, PyramidKV, H2O-style (ObservedAttention) and StreamingLLM through the
vendored KVPress pipeline under the frozen runner contract of
scripts/run_kvpress_natural_qa.py (context compressed, question uncompressed, total
prefill budget = 512 tokens), on the 128-example natural MuSiQue topology slice, and
records the kept token indices of EVERY (layer, KV-head) so coverage can be reported
as head-union, per-head mean, and all-head intersection.

Caveat stated for the record: these presses keep a different token set per layer and
per KV head, so "the retained token set" is not a single set. The head-union is the
most generous reading (a token counts as retained if any of the 28 x 4 = 112 (layer,
head) caches keeps it); the intersection is the strictest.

Span granularity: whole MuSiQue paragraph (20 per example, 2-4 gold), not sentence.
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
from typing import Any

IMPL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(IMPL / "scripts"))
sys.path.insert(0, str(IMPL / "third_party/kvpress"))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from proofkv.eval.generation import load_model_tokenizer  # noqa: E402
from proofkv.utils.io import append_jsonl  # noqa: E402
from eval_natural_route_coverage import build_hotpot_prompt, token_span  # noqa: E402
from run_hotpot_compact_qa import answer_scores, extract_short_answer, normalize_ircot_source  # noqa: E402
from run_kvpress_natural_qa import fixed_budget_targets, make_press, pipeline_token_counts  # noqa: E402
from run_natural_topology_causality import file_sha256  # noqa: E402

FROZEN_METRICS = (
    str(Path(__file__).resolve().parents[3] / "results/paper/reference/musique_interventions_original_metrics.json")
)
RECORDER: dict[int, torch.Tensor] = {}


def install_recorders() -> None:
    """Replace ScorerPress/PyramidKVPress.compress with index-recording copies of the upstream code."""
    from kvpress.presses.pyramidkv_press import PyramidKVPress
    from kvpress.presses.scorer_press import ScorerPress

    def scorer_compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        RECORDER[int(module.layer_idx)] = indices[0].detach().to("cpu", copy=True)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous()

    def pyramid_compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        k_len = keys.shape[2]
        n_kept = self.get_layer_budget(module, k_len)
        indices = scores.topk(n_kept, dim=-1).indices
        RECORDER[int(module.layer_idx)] = indices[0].detach().to("cpu", copy=True)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous()

    ScorerPress.compress = scorer_compress
    PyramidKVPress.compress = pyramid_compress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--methods", default="snapkv,pyramidkv,streamingllm,observed_attention_h2o_style")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--sample-offset", type=int, default=128)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_wall = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path, metrics_path, manifest_path = out_dir / "results.jsonl", out_dir / "metrics.json", out_dir / "run_manifest.json"
    if any(p.exists() for p in (result_path, metrics_path, manifest_path)):
        raise FileExistsError(out_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    dataset = load_dataset("json", data_files={"validation": args.data_file}, split="validation")
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    selected_indices = order[args.sample_offset : args.sample_offset + args.max_samples]
    selected_ids = [str(normalize_ircot_source(dict(dataset[i]))["id"]) for i in selected_indices]
    frozen = json.loads(Path(FROZEN_METRICS).read_text())
    if selected_indices != frozen["protocol"]["selected_indices"] or selected_ids != frozen["protocol"]["selected_ids"]:
        raise RuntimeError("slice does not match the frozen manifest")

    import kvpress
    import transformers

    install_recorders()
    from kvpress import KVPressTextGenerationPipeline

    code_files = {
        "experiments/followup/premise_test/compressors.py": Path(__file__).resolve(),
        "scripts/run_kvpress_natural_qa.py": IMPL / "scripts/run_kvpress_natural_qa.py",
        "third_party/kvpress/kvpress/pipeline.py": IMPL / "third_party/kvpress/kvpress/pipeline.py",
        "third_party/kvpress/kvpress/presses/base_press.py": IMPL / "third_party/kvpress/kvpress/presses/base_press.py",
        "third_party/kvpress/kvpress/presses/scorer_press.py": IMPL / "third_party/kvpress/kvpress/presses/scorer_press.py",
        "third_party/kvpress/kvpress/presses/snapkv_press.py": IMPL / "third_party/kvpress/kvpress/presses/snapkv_press.py",
        "third_party/kvpress/kvpress/presses/pyramidkv_press.py": IMPL / "third_party/kvpress/kvpress/presses/pyramidkv_press.py",
        "third_party/kvpress/kvpress/presses/observed_attention_press.py": IMPL / "third_party/kvpress/kvpress/presses/observed_attention_press.py",
        "third_party/kvpress/kvpress/presses/streaming_llm_press.py": IMPL / "third_party/kvpress/kvpress/presses/streaming_llm_press.py",
    }
    manifest = {
        "experiment_id": "premise_test/compressors",
        "purpose": "per-(layer, KV-head) kept-index capture for token-level eviction baselines on the topology slice",
        "span_granularity": "whole MuSiQue paragraph (20 per example, 2-4 gold per question); NOT sentence-level",
        "contract": "scripts/run_kvpress_natural_qa.py native KVPress contract: compressed context + uncompressed question prefill; "
        "budget_tokens=512 total prefill; press compression_ratio = 1 - (512 - question_tokens)/context_tokens; "
        "SnapKV/PyramidKV window_size=64 kernel_size=5; StreamingLLM n_sink=4; H2O-style = ObservedAttentionPress (eager attention)",
        "per_head_caveat": "SnapKV/PyramidKV/H2O keep a different token set per layer and per KV head (28 layers x 4 KV heads = 112 sets). "
        "Coverage is reported as head-union (any set keeps the token), per-head mean, and all-head intersection.",
        "arguments": {k: v for k, v in vars(args).items() if k != "out_dir"},
        "command_line": " ".join(sys.argv),
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "kvpress_version": getattr(kvpress, "__version__", "vendored"),
        "kvpress_commit": "6d965557a5b9f0201a2301b23c454473dd681d0d",
        "kvpress_compatibility_patch": "kvpress_transformers_5_cache_position.patch",
        "cuda_device": torch.cuda.get_device_name(0),
        "model": args.model, "model_revision": args.model_revision,
        "implementation_sha256": {k: file_sha256(p) for k, p in code_files.items()},
        "dataset_sha256": file_sha256(Path(args.data_file)),
        "selected_indices": selected_indices, "selected_ids": selected_ids,
        "opened_after_submission": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_wall)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    all_rows: list[dict[str, Any]] = []
    attn_groups = [("sdpa", [m for m in methods if m != "observed_attention_h2o_style"]),
                   ("eager", [m for m in methods if m == "observed_attention_h2o_style"])]
    for attn_implementation, group in attn_groups:
        if not group:
            continue
        model, tokenizer = load_model_tokenizer(args.model, dtype="bfloat16", attn_implementation=attn_implementation)
        pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer, device=model.device)
        num_layers = model.config.num_hidden_layers
        num_kv_heads = model.config.num_key_value_heads
        for method in group:
            for position, dataset_index in enumerate(selected_indices):
                source = normalize_ircot_source(dict(dataset[dataset_index]))
                sample_id = str(source["id"])
                base_prompt, sentences, old_query = build_hotpot_prompt(source)
                evidence_context = base_prompt[: -len(old_query)]
                instruction = "Answer using only a short answer phrase from the evidence. Do not explain.\n"
                context = instruction + evidence_context
                question = f"\nQuestion: {source['question']}\nReturn only the shortest answer phrase from the evidence."
                context_tokens, question_tokens = pipeline_token_counts(tokenizer, context, question)
                retained_total_target, retained_context_target, effective_budget = fixed_budget_targets(
                    context_tokens, question_tokens, 0.08, args.budget_tokens)
                # gold paragraph token ranges inside the chat-formatted context exactly as the pipeline tokenizes it
                separator = "#" * (len(context) + 10)
                rendered = tokenizer.apply_chat_template([{"role": "user", "content": context + separator}],
                                                         add_generation_prompt=True, tokenize=False, enable_thinking=False)
                formatted, _ = rendered.split(separator, 1)
                shift = formatted.find(context)
                if shift < 0:
                    raise RuntimeError("chat template did not preserve the context text")
                shift += len(instruction)
                enc = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
                offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
                if len(enc["input_ids"]) != context_tokens:
                    raise RuntimeError(f"context tokenization mismatch {len(enc['input_ids'])} != {context_tokens}")
                gold = [(f"s{i}", token_span(offsets, s["char_span"][0] + shift, s["char_span"][1] + shift))
                        for i, s in enumerate(sentences) if s["gold"]]
                gold_token_sets = {g: set(range(a, b)) for g, (a, b) in gold}
                press = make_press(method, effective_budget, exact_kept_tokens=retained_context_target)
                RECORDER.clear()
                torch.cuda.empty_cache()
                started = time.perf_counter()
                cache = DynamicCache()
                prediction = pipe(context, question=question, press=press, cache=cache, max_new_tokens=args.max_new_tokens)["answer"]
                elapsed = time.perf_counter() - started
                actual_context_tokens = int(getattr(cache, "_proofcache_compressed_context_length", context_tokens))
                short = extract_short_answer(prediction)
                em, f1, contains = answer_scores(short, str(source["answer"]))
                if len(RECORDER) != num_layers:
                    raise RuntimeError(f"{method}: recorded {len(RECORDER)} layers, expected {num_layers}")
                head_sets: list[set[int]] = []
                for layer_idx in range(num_layers):
                    kept = RECORDER[layer_idx]
                    if kept.shape[0] != num_kv_heads:
                        raise RuntimeError("unexpected head count")
                    for head in range(num_kv_heads):
                        head_sets.append(set(kept[head].tolist()))
                union = set().union(*head_sets)
                intersection = set.intersection(*head_sets)
                def cov(retained: set[int]) -> dict[str, Any]:
                    per = [len(gold_token_sets[g] & retained) / max(1, len(gold_token_sets[g])) for g, _ in gold]
                    touched = sum(v > 0 for v in per)
                    all_gold = set().union(*gold_token_sets.values())
                    return {"per_gold_span_retention": per, "mean_gold_span_retention": statistics.fmean(per),
                            "min_gold_span_retention": min(per), "gold_spans_touched": touched,
                            "all_gold_touched": float(touched == len(per)), "absent_gold_spans": len(per) - touched,
                            "gold_token_recall": len(all_gold & retained) / max(1, len(all_gold)),
                            "gold_proof_complete": float(all(v >= 1.0 for v in per))}
                per_head = [cov(s) for s in head_sets]
                row = {
                    "id": sample_id, "question_type": source["type"], "method": method,
                    "attn_implementation": attn_implementation,
                    "context_tokens": context_tokens, "question_tokens": question_tokens,
                    "total_prefill_tokens": context_tokens + question_tokens,
                    "retained_tokens_target": retained_total_target, "retained_context_tokens_target": retained_context_target,
                    "actual_retained_context_tokens": actual_context_tokens,
                    "actual_retained_tokens": actual_context_tokens + question_tokens,
                    "press_compression_ratio": 1.0 - effective_budget,
                    "kept_per_head_min": min(len(s) for s in head_sets), "kept_per_head_max": max(len(s) for s in head_sets),
                    "head_sets": len(head_sets), "union_size": len(union), "intersection_size": len(intersection),
                    "union_fraction_of_context": len(union) / context_tokens,
                    "gold_span_count": len(gold),
                    "gold_token_ranges": {g: [a, b] for g, (a, b) in gold},
                    "coverage_head_union": cov(union),
                    "coverage_head_intersection": cov(intersection),
                    "coverage_per_head_mean": {
                        "mean_gold_span_retention": statistics.fmean(c["mean_gold_span_retention"] for c in per_head),
                        "all_gold_touched": statistics.fmean(c["all_gold_touched"] for c in per_head),
                        "gold_token_recall": statistics.fmean(c["gold_token_recall"] for c in per_head),
                        "gold_proof_complete": statistics.fmean(c["gold_proof_complete"] for c in per_head),
                        "absent_gold_spans": statistics.fmean(c["absent_gold_spans"] for c in per_head),
                    },
                    "answer": str(source["answer"]), "prediction": prediction, "short_prediction": short,
                    "answer_em": em, "answer_f1": f1, "answer_contains": contains, "seconds": elapsed,
                }
                all_rows.append(row)
                append_jsonl(result_path, row)
                print(f"[{method} {position+1}/{len(selected_indices)}] {sample_id} ctx={context_tokens} kept={actual_context_tokens} "
                      f"union={len(union)} touched_union={row['coverage_head_union']['gold_spans_touched']}/{len(gold)} f1={f1:.2f}", flush=True)
        del pipe, model
        torch.cuda.empty_cache()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["method"]].append(row)
    records = []
    for method in methods:
        values = grouped[method]
        rec: dict[str, Any] = {"method": method, "n": len(values),
                               "answer_em": statistics.fmean(v["answer_em"] for v in values),
                               "answer_f1": statistics.fmean(v["answer_f1"] for v in values),
                               "answer_contains": statistics.fmean(v["answer_contains"] for v in values),
                               "actual_retained_tokens_mean": statistics.fmean(v["actual_retained_tokens"] for v in values),
                               "actual_retained_tokens_min": min(v["actual_retained_tokens"] for v in values),
                               "actual_retained_tokens_max": max(v["actual_retained_tokens"] for v in values),
                               "union_fraction_of_context_mean": statistics.fmean(v["union_fraction_of_context"] for v in values),
                               "union_size_mean": statistics.fmean(v["union_size"] for v in values),
                               "intersection_size_mean": statistics.fmean(v["intersection_size"] for v in values)}
        for view in ("coverage_head_union", "coverage_head_intersection"):
            rec[view] = {
                "all_gold_spans_touched": statistics.fmean(v[view]["all_gold_touched"] for v in values),
                "mean_gold_span_retention": statistics.fmean(v[view]["mean_gold_span_retention"] for v in values),
                "gold_token_recall": statistics.fmean(v[view]["gold_token_recall"] for v in values),
                "gold_proof_complete": statistics.fmean(v[view]["gold_proof_complete"] for v in values),
                "absent_gold_span_distribution": dict(sorted(Counter(v[view]["absent_gold_spans"] for v in values).items())),
            }
        rec["coverage_per_head_mean"] = {k: statistics.fmean(v["coverage_per_head_mean"][k] for v in values)
                                         for k in values[0]["coverage_per_head_mean"]}
        records.append(rec)
    payload = {"protocol": manifest, "records": records,
               "integrity": {"rows_expected": len(methods) * len(selected_ids), "rows_observed": len(all_rows),
                             "unique_pairs": len({(r["id"], r["method"]) for r in all_rows})},
               "wall_clock_seconds": time.time() - started_wall,
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    manifest.update({"finished_utc": payload["finished_utc"], "wall_clock_seconds": payload["wall_clock_seconds"]})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
