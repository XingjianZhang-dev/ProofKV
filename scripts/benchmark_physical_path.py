#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import run_physical_kv_replay as replay
from proofkv.eval.generation import load_model_tokenizer
from proofkv.physical_kv import (
    compact_cache_stats,
    gather_dynamic_cache_inplace,
    gather_legacy_past_key_values,
    legacy_cache_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent-process physical KV path benchmark.")
    parser.add_argument(
        "--method",
        choices=(
            "full",
            "gather",
            "gather_inplace",
            "compact_prefill",
            "chunked_compact_prefill",
        ),
        required=True,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--context-length", type=int, default=8000)
    parser.add_argument("--budget-ratio", type=float, default=0.08)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--full-prefill-chunk-size",
        type=int,
        default=0,
        help="Chunk size for Full/gather prompt processing; 0 uses one-shot prefill.",
    )
    parser.add_argument(
        "--compact-prefill-chunk-size",
        type=int,
        default=512,
        help="Chunk size for chunked_compact_prefill.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def cache_bytes(cache: Any) -> int:
    return sum(key.numel() * key.element_size() + value.numel() * value.element_size() for key, value in legacy_cache_layers(cache))


def dynamic_cache(legacy_cache: Any):
    from transformers.cache_utils import DynamicCache

    return DynamicCache(
        ddp_cache_data=tuple((key.clone(), value.clone()) for key, value in legacy_cache)
    )


@torch.inference_mode()
def full_prefill(
    model: Any,
    input_ids: torch.Tensor,
    warnings: list[str],
    chunk_size: int,
) -> Any:
    """Run a one-shot or production-style chunked Full prefill."""
    if chunk_size <= 0 or chunk_size >= input_ids.shape[1]:
        return replay._full_prefill(model, input_ids, warnings)
    past = None
    out = None
    seq_len = int(input_ids.shape[1])
    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        attention_mask = torch.ones(
            (input_ids.shape[0], end), dtype=torch.long, device=input_ids.device
        )
        position_ids = replay.make_position_ids(start, end - start, input_ids.device)
        cache_position = replay.make_cache_position(start, end - start, input_ids.device)
        out = replay._forward(
            model,
            input_ids[:, start:end],
            attention_mask,
            position_ids,
            past,
            cache_position,
            warnings,
        )
        past = out.past_key_values
    if out is None:  # pragma: no cover
        raise ValueError("empty Full prefill")
    return out


@torch.inference_mode()
def chunked_compact_prefill(
    model: Any,
    input_ids: torch.Tensor,
    retained_indices: Sequence[int],
    warnings: list[str],
    chunk_size: int,
) -> tuple[Any, Any]:
    """Prefill the sparse proof route in bounded compact-cache chunks.

    Original RoPE positions remain sparse, while cache positions are compact
    and gap-free. Causally this is equivalent to one-shot compact prefill, but
    it bounds temporary activations for larger batches and longer routes.
    """
    if chunk_size <= 0:
        raise ValueError("compact-prefill-chunk-size must be positive")
    prepared = replay.prepare_compact_prefill_inputs(input_ids, retained_indices)
    if prepared.input_ids.shape[1] <= 0:
        raise ValueError("chunked compact prefill needs at least one retained prompt token")
    past = None
    out = None
    compact_len = int(prepared.input_ids.shape[1])
    for start in range(0, compact_len, chunk_size):
        end = min(compact_len, start + chunk_size)
        attention_mask = torch.ones(
            (input_ids.shape[0], end), dtype=torch.long, device=input_ids.device
        )
        original_position_ids = prepared.position_ids[:, start:end]
        compact_cache_position = prepared.cache_position[start:end]
        attn_implementation = str(
            getattr(model.config, "_attn_implementation", "")
        )
        rotary_owner = getattr(model, "model", None)
        rotary_emb = getattr(rotary_owner, "rotary_emb", None)
        if attn_implementation.startswith("flash_attention") and rotary_emb is not None:
            # Hugging Face FlashAttention interprets any non-unit jump in
            # position_ids as a packed-sequence boundary. Sparse original RoPE
            # positions are therefore valid for a one-shot route but crash on
            # later cached chunks, which no longer contain a zero position.
            # Feed compact positions to the mask/FlashAttention dispatcher
            # while overriding only the rotary embedding with the original
            # sparse positions. Causal/cache order stays compact and gap-free.
            flash_position_ids = compact_cache_position.unsqueeze(0).expand(
                input_ids.shape[0], -1
            )
            original_rotary_forward = rotary_emb.forward

            def sparse_rotary_forward(hidden_states, _position_ids):
                return original_rotary_forward(hidden_states, original_position_ids)

            rotary_emb.forward = sparse_rotary_forward
            try:
                out = replay._forward(
                    model,
                    prepared.input_ids[:, start:end],
                    attention_mask,
                    flash_position_ids,
                    past,
                    compact_cache_position,
                    warnings,
                )
            finally:
                rotary_emb.forward = original_rotary_forward
        else:
            out = replay._forward(
                model,
                prepared.input_ids[:, start:end],
                attention_mask,
                original_position_ids,
                past,
                compact_cache_position,
                warnings,
            )
        past = out.past_key_values
    if out is None:  # pragma: no cover
        raise ValueError("empty chunked compact prefill")
    return out, prepared


@torch.inference_mode()
def fixed_decode_from_prefill(
    model: Any,
    past: Any,
    logits: torch.Tensor,
    original_position: int,
    cache_position: int,
    output_tokens: int,
) -> None:
    warnings: list[str] = []
    next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
    for step in range(1, output_tokens):
        out = replay._step_with_cache(
            model,
            next_id,
            past,
            original_position + step - 1,
            warnings,
            cache_position + step - 1,
        )
        past = out.past_key_values
        next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)


@torch.inference_mode()
def fixed_decode_from_bootstrap(
    model: Any,
    past: Any,
    bootstrap_ids: torch.Tensor,
    original_position: int,
    cache_position: int,
    output_tokens: int,
) -> None:
    warnings: list[str] = []
    next_input = bootstrap_ids
    for step in range(output_tokens):
        out = replay._step_with_cache(
            model,
            next_input,
            past,
            original_position + step,
            warnings,
            cache_position + step,
        )
        past = out.past_key_values
        next_input = out.logits[:, -1].argmax(dim=-1, keepdim=True)


@torch.inference_mode()
def bootstrap_compact_cache(
    model: Any,
    past: Any,
    bootstrap_ids: torch.Tensor,
    original_position: int,
    cache_position: int,
) -> Any:
    """Recontextualize the final prompt token before TTFT is recorded.

    A gathered cache excludes the final prompt token so that it can attend to
    the compacted prefix. This step is prompt processing, not autoregressive
    decode, and therefore belongs in the prefill/TTFT interval.
    """
    warnings: list[str] = []
    return replay._step_with_cache(
        model,
        bootstrap_ids,
        past,
        original_position,
        warnings,
        cache_position,
    )


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def main() -> None:
    args = parse_args()
    if args.output_tokens <= 0 or args.batch_size <= 0 or args.repeats <= 0 or args.warmups < 0:
        raise ValueError("output-tokens/batch-size/repeats must be positive and warmups non-negative")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(metrics_path)

    model, tokenizer = load_model_tokenizer(
        args.model,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )
    device = replay._model_device(model)
    retained_total, recent, guidepost_budget = replay.local_ruler.budget_parts(
        args.context_length, args.budget_ratio
    )
    sample_args = argparse.Namespace(
        tasks="exact_retrieval",
        source="local_ruler_lite",
        n_per_task=1,
        context_length=args.context_length,
        seed=args.seed,
        model=args.model,
    )
    samples, sample_meta = replay._build_samples(sample_args, tokenizer, out_dir)
    sample = samples[0]
    encoded = tokenizer(sample["prompt"], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device).repeat(args.batch_size, 1)
    seq_len = int(input_ids.shape[1])
    plan = replay._plan_for_sample(sample_args, sample, tokenizer, recent, guidepost_budget)
    retained_indices = sorted(int(index) for index in plan.retained_indices)
    if retained_indices[-1] != seq_len - 1:
        raise ValueError("benchmark plan must retain the final prompt token")
    bootstrap_ids = input_ids[:, -1:]
    model_allocated_gb = torch.cuda.memory_allocated() / 1024**3

    measurements: list[dict[str, float]] = []
    for iteration in range(args.warmups + args.repeats):
        replay._release_cuda()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        prefill_started = started
        resident_kv_bytes = 0

        if args.method == "full":
            warnings: list[str] = []
            prefill_out = full_prefill(
                model, input_ids, warnings, args.full_prefill_chunk_size
            )
            torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - prefill_started
            resident_kv_bytes = cache_bytes(prefill_out.past_key_values)
            decode_started = time.perf_counter()
            fixed_decode_from_prefill(
                model,
                prefill_out.past_key_values,
                prefill_out.logits,
                seq_len,
                seq_len,
                args.output_tokens,
            )
            del prefill_out
        elif args.method in ("gather", "gather_inplace"):
            warnings = []
            prefill_out = full_prefill(
                model, input_ids, warnings, args.full_prefill_chunk_size
            )
            retained_without_bootstrap = [index for index in retained_indices if index != seq_len - 1]
            compact_legacy = None
            if args.method == "gather_inplace":
                compact = gather_dynamic_cache_inplace(
                    prefill_out.past_key_values, retained_without_bootstrap
                )
            else:
                compact_legacy = gather_legacy_past_key_values(
                    prefill_out.past_key_values,
                    retained_without_bootstrap,
                    return_dynamic_cache=False,
                )
                compact = dynamic_cache(compact_legacy)
            bootstrap_out = bootstrap_compact_cache(
                model,
                compact,
                bootstrap_ids,
                seq_len - 1,
                len(retained_without_bootstrap),
            )
            torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - prefill_started
            resident_kv_bytes = cache_bytes(bootstrap_out.past_key_values)
            decode_started = time.perf_counter()
            fixed_decode_from_prefill(
                model,
                bootstrap_out.past_key_values,
                bootstrap_out.logits,
                seq_len,
                len(retained_without_bootstrap) + 1,
                args.output_tokens,
            )
            del prefill_out, compact, compact_legacy, bootstrap_out
        else:
            warnings = []
            if args.method == "chunked_compact_prefill":
                prefill_out, prepared = chunked_compact_prefill(
                    model,
                    input_ids,
                    retained_indices,
                    warnings,
                    args.compact_prefill_chunk_size,
                )
            else:
                prefill_out, prepared = replay._compact_prefill(
                    model, input_ids, retained_indices, warnings
                )
            torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - prefill_started
            resident_kv_bytes = cache_bytes(prefill_out.past_key_values)
            decode_started = time.perf_counter()
            fixed_decode_from_prefill(
                model,
                prefill_out.past_key_values,
                prefill_out.logits,
                seq_len,
                int(prepared.input_ids.shape[1]),
                args.output_tokens,
            )
            del prefill_out, prepared

        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        total_seconds = time.perf_counter() - started
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        decode_tokens_timed = max(0, args.output_tokens - 1)
        if iteration >= args.warmups:
            measurements.append(
                {
                    "prefill_seconds": prefill_seconds,
                    "decode_seconds": decode_seconds,
                    "total_seconds": total_seconds,
                    "tokens_per_second": args.batch_size * decode_tokens_timed / decode_seconds,
                    "time_per_decode_token_seconds": decode_seconds / max(1, decode_tokens_timed),
                    "requests_per_second": args.batch_size / total_seconds,
                    "peak_cuda_memory_gb": peak_gb,
                    "temporary_peak_gb_above_model": max(0.0, peak_gb - model_allocated_gb),
                    "resident_kv_gb": resident_kv_bytes / 1024**3,
                }
            )

    def summary(key: str) -> dict[str, float]:
        values = [row[key] for row in measurements]
        return {
            "mean": float(sum(values) / len(values)),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
        }

    payload = {
        "method": args.method,
        "model": args.model,
        "context_length_requested": args.context_length,
        "prompt_tokens_actual": seq_len,
        "budget_ratio_requested": args.budget_ratio,
        "retained_tokens_actual": len(retained_indices),
        "retained_ratio_actual": len(retained_indices) / seq_len,
        "output_tokens": args.output_tokens,
        "decode_tokens_timed_per_request": max(0, args.output_tokens - 1),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "full_prefill_chunk_size": args.full_prefill_chunk_size,
        "compact_prefill_chunk_size": (
            args.compact_prefill_chunk_size
            if args.method == "chunked_compact_prefill"
            else 0
        ),
        "sparse_flash_rope_bridge": bool(
            args.method == "chunked_compact_prefill"
            and str(getattr(model.config, "_attn_implementation", "")).startswith(
                "flash_attention"
            )
        ),
        "model_allocated_gb": model_allocated_gb,
        "sample_meta": sample_meta,
        "summaries": {
            key: summary(key)
            for key in (
                "prefill_seconds",
                "decode_seconds",
                "total_seconds",
                "tokens_per_second",
                "time_per_decode_token_seconds",
                "requests_per_second",
                "peak_cuda_memory_gb",
                "temporary_peak_gb_above_model",
                "resident_kv_gb",
            )
        },
        "measurements": measurements,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "summaries": payload["summaries"]}))


if __name__ == "__main__":
    main()
