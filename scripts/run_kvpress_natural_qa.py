#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from proofkv.eval.generation import load_model_tokenizer
from proofkv.utils.io import append_jsonl
from eval_natural_route_coverage import build_hotpot_prompt
from run_hotpot_compact_qa import (
    answer_scores,
    extract_short_answer,
    load_sources,
    normalize_ircot_source,
)


METHODS = {
    "full",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "adakv_snapkv",
    "chunkkv_snapkv",
    "kvzip",
    "observed_attention_h2o_style",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official NVIDIA KVPress natural QA baseline panel.")
    parser.add_argument("--dataset", choices=("hotpotqa", "musique", "2wiki"), default="hotpotqa")
    parser.add_argument("--data-file")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--unlock-locked-test", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Offset into the deterministically shuffled sample order.",
    )
    parser.add_argument("--budget", type=float, default=0.08)
    parser.add_argument(
        "--budget-tokens",
        type=int,
        default=None,
        help="Optional fixed retained-context-token target; overrides --budget per sample.",
    )
    parser.add_argument(
        "--methods",
        default="full,streamingllm,snapkv,pyramidkv,adakv_snapkv,chunkkv_snapkv,kvzip",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def make_press(
    method: str,
    budget: float,
    exact_kept_tokens: int | None = None,
):
    from kvpress import (
        AdaKVPress,
        ChunkKVPress,
        KVzipPress,
        ObservedAttentionPress,
        PyramidKVPress,
        SnapKVPress,
        StreamingLLMPress,
    )

    compression_ratio = 1.0 - budget
    if method == "full":
        return None
    if method == "streamingllm":
        return StreamingLLMPress(compression_ratio=compression_ratio, n_sink=4)
    if method == "snapkv":
        return SnapKVPress(compression_ratio=compression_ratio, window_size=64, kernel_size=5)
    if method == "pyramidkv":
        return PyramidKVPress(compression_ratio=compression_ratio, window_size=64, kernel_size=5)
    if method == "adakv_snapkv":
        return AdaKVPress(
            SnapKVPress(compression_ratio=compression_ratio, window_size=64, kernel_size=5),
            alpha_safeguard=0.2,
        )
    if method == "chunkkv_snapkv":
        return ChunkKVPress(
            SnapKVPress(compression_ratio=compression_ratio, window_size=64, kernel_size=5),
            chunk_length=20,
            exact_kept_tokens=exact_kept_tokens,
        )
    if method == "kvzip":
        return KVzipPress(compression_ratio=compression_ratio, layerwise=False, n_sink=4)
    if method == "observed_attention_h2o_style":
        return ObservedAttentionPress(compression_ratio=compression_ratio)
    raise ValueError(method)


def pipeline_token_counts(tokenizer: Any, context: str, question: str) -> tuple[int, int]:
    """Mirror KVPress preprocessing for a fair total-prefill KV budget."""
    if tokenizer.chat_template is None:
        formatted = str(getattr(tokenizer, "bos_token", "") or "") + context
        question_suffix = "\n"
    else:
        separator = "#" * (len(context) + 10)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        formatted, question_suffix = rendered.split(separator, 1)
    context_tokens = len(tokenizer.encode(formatted, add_special_tokens=False))
    question_tokens = len(tokenizer.encode(question + question_suffix, add_special_tokens=False))
    return context_tokens, question_tokens


def fixed_budget_targets(
    context_tokens: int,
    question_tokens: int,
    budget: float,
    budget_tokens: int | None,
) -> tuple[int, int, float]:
    """Convert a total-prefill budget into the context-only KVPress budget."""
    if context_tokens <= 0 or question_tokens < 0:
        raise ValueError("token counts must be non-negative and context must be non-empty")
    total_prefill_tokens = context_tokens + question_tokens
    retained_total_target = min(
        total_prefill_tokens,
        budget_tokens if budget_tokens is not None else round(total_prefill_tokens * budget),
    )
    retained_context_target = max(5, retained_total_target - question_tokens)
    retained_context_target = min(context_tokens, retained_context_target)
    effective_budget = retained_context_target / context_tokens
    return retained_total_target, retained_context_target, effective_budget


def read_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    keys = [(str(row["id"]), str(row["method"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate sample/method rows in {path}")
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    records = []
    for method, values in sorted(grouped.items()):
        elapsed = sorted(value["seconds"] for value in values)
        peak = sorted(value["peak_cuda_memory_gb"] for value in values)
        p95_index = min(len(values) - 1, max(0, int(0.95 * len(values) + 0.999999) - 1))
        records.append(
            {
                "method": method,
                "n": len(values),
                "answer_em": sum(value["answer_em"] for value in values) / len(values),
                "answer_f1": sum(value["answer_f1"] for value in values) / len(values),
                "answer_contains": sum(value["answer_contains"] for value in values) / len(values),
                "actual_retained_ratio": sum(value["actual_retained_ratio"] for value in values) / len(values),
                "actual_kv_bytes": sum(value["actual_kv_bytes"] for value in values) / len(values),
                "seconds_mean": sum(elapsed) / len(elapsed),
                "seconds_p95": elapsed[p95_index],
                "peak_cuda_memory_gb_p95": peak[p95_index],
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if not 0.0 < args.budget <= 1.0:
        raise ValueError("budget must lie in (0, 1]")
    if args.budget_tokens is not None and args.budget_tokens <= 4:
        raise ValueError("budget-tokens must exceed the four-token sink")
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = set(methods) - METHODS
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    metrics_path = out_dir / "metrics.json"
    run_manifest_path = out_dir / "run_manifest.json"
    if metrics_path.exists():
        raise FileExistsError(f"completed output already exists: {metrics_path}")

    # Importing kvpress registers and patches the supported attention classes.
    from kvpress import KVPressTextGenerationPipeline
    from transformers import DynamicCache

    dataset = load_sources(args)
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    if args.sample_offset < 0:
        raise ValueError("--sample-offset must be non-negative")
    selected_indices = order[args.sample_offset : args.sample_offset + args.max_samples]
    if len(selected_indices) != args.max_samples:
        raise ValueError(
            f"requested {args.max_samples} samples at offset {args.sample_offset}, "
            f"but only {len(selected_indices)} are available"
        )
    selected_ids = [
        str(normalize_ircot_source(dict(dataset[dataset_idx]))["id"])
        for dataset_idx in selected_indices
    ]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected dataset contains duplicate ids")
    run_manifest = {
        "arguments": {key: value for key, value in vars(args).items() if key != "out_dir"},
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "selected_indices": selected_indices,
        "selected_ids": selected_ids,
    }
    if run_manifest_path.exists():
        existing_manifest = json.loads(run_manifest_path.read_text())
        if existing_manifest != run_manifest:
            raise ValueError("run manifest does not match this resumed KVPress evaluation")
    else:
        if results_path.exists():
            raise ValueError("cannot resume results without a frozen run manifest")
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    attn_implementation = "eager" if "observed_attention_h2o_style" in methods else "sdpa"
    model, tokenizer = load_model_tokenizer(
        args.model, dtype="bfloat16", attn_implementation=attn_implementation
    )
    pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer, device=model.device)
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    kv_bytes_per_token = (
        model.config.num_hidden_layers * model.config.num_key_value_heads * head_dim * 2 * 2
    )
    rows = read_existing_results(results_path)
    completed = {(str(row["id"]), str(row["method"])) for row in rows}

    for method in methods:
        for dataset_idx in tqdm(selected_indices, desc=method):
            source = normalize_ircot_source(dict(dataset[dataset_idx]))
            source_id = str(source["id"])
            if (source_id, method) in completed:
                continue
            base_prompt, _, old_query = build_hotpot_prompt(source)
            evidence_context = base_prompt[: -len(old_query)]
            context = (
                "Answer using only a short answer phrase from the evidence. Do not explain.\n"
                + evidence_context
            )
            question = (
                f"\nQuestion: {source['question']}\n"
                "Return only the shortest answer phrase from the evidence."
            )
            context_tokens, question_tokens = pipeline_token_counts(tokenizer, context, question)
            total_prefill_tokens = context_tokens + question_tokens
            retained_total_target, retained_context_target, effective_budget = fixed_budget_targets(
                context_tokens,
                question_tokens,
                args.budget,
                args.budget_tokens,
            )
            press = make_press(
                method,
                effective_budget,
                exact_kept_tokens=retained_context_target,
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            cache = DynamicCache()
            prediction = pipe(
                context,
                question=question,
                press=press,
                cache=cache,
                max_new_tokens=args.max_new_tokens,
            )["answer"]
            elapsed = time.perf_counter() - started
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            actual_context_tokens = int(
                getattr(cache, "_proofcache_compressed_context_length", context_tokens)
            )
            actual_context_kv_bytes = int(
                getattr(
                    cache,
                    "_proofcache_compressed_context_bytes",
                    actual_context_tokens * kv_bytes_per_token,
                )
            )
            actual_total_kv_bytes = actual_context_kv_bytes + question_tokens * kv_bytes_per_token
            actual_total_prefill_tokens = actual_total_kv_bytes / kv_bytes_per_token
            short_prediction = extract_short_answer(prediction)
            em, f1, contains = answer_scores(short_prediction, source["answer"])
            row = {
                "id": source["id"],
                "question_type": source["type"],
                "method": method,
                "target_retained_ratio": 1.0 if method == "full" else (
                    retained_context_target + question_tokens
                ) / total_prefill_tokens,
                "context_tokens": context_tokens,
                "question_tokens": question_tokens,
                "total_prefill_tokens": total_prefill_tokens,
                "retained_context_tokens_target": context_tokens if method == "full" else retained_context_target,
                "retained_tokens_target": total_prefill_tokens if method == "full" else (
                    retained_context_target + question_tokens
                ),
                "actual_retained_context_tokens": actual_context_tokens,
                "actual_retained_context_kv_bytes": actual_context_kv_bytes,
                "actual_total_prefill_tokens": actual_total_prefill_tokens,
                "actual_retained_ratio": actual_total_prefill_tokens / total_prefill_tokens,
                "actual_kv_bytes": actual_total_kv_bytes,
                "answer": source["answer"],
                "prediction": prediction,
                "short_prediction": short_prediction,
                "answer_em": em,
                "answer_f1": f1,
                "answer_contains": contains,
                "seconds": elapsed,
                "peak_cuda_memory_gb": peak_gb,
            }
            rows.append(row)
            append_jsonl(results_path, row)
            completed.add((source_id, method))

    expected_rows = len(methods) * len(selected_indices)
    if len(rows) != expected_rows:
        raise RuntimeError(f"incomplete result matrix: {len(rows)} != {expected_rows}")

    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "data_file": args.data_file,
        "max_samples": args.max_samples,
        "sample_offset": args.sample_offset,
        "budget": args.budget,
        "budget_tokens": args.budget_tokens,
        "budget_scope": "compressed_context_plus_uncompressed_question_prefill",
        "attn_implementation": attn_implementation,
        "kv_bytes_per_token_bf16": kv_bytes_per_token,
        "max_new_tokens": args.max_new_tokens,
        "methods": methods,
        "kvpress_commit": "6d965557a5b9f0201a2301b23c454473dd681d0d",
        "kvpress_compatibility_patch": "kvpress_transformers_5_cache_position.patch",
        "kvpress_instrumentation_patches": [
            "kvpress_record_actual_cache_length.patch",
            "kvpress_record_actual_cache_bytes.patch",
        ],
        "records": aggregate(rows),
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "records": payload["records"]}))


if __name__ == "__main__":
    main()
