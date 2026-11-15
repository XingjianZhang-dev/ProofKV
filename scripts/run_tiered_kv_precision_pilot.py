#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import load_dataset
from transformers.cache_utils import DynamicCache

from benchmark_physical_path import chunked_compact_prefill
from proofkv.eval.generation import load_model_tokenizer
from proofkv.physical_kv import legacy_cache_layers
from proofkv.proofcache.model import TypedRelationScorer
from proofkv.utils.io import append_jsonl
from run_hotpot_compact_qa import (
    answer_scores,
    build_natural_distractor_pool,
    extract_short_answer,
    make_plan,
    make_prompt,
    normalize_ircot_source,
    pack_source_to_prompt_target,
)
from run_physical_kv_replay import (
    _decode_text,
    _greedy_decode_from_prefill_logits,
    _release_cuda,
    _step_with_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-storage proof-core BF16 / auxiliary INT8-INT4 pilot.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--target-prompt-tokens", type=int, default=8000)
    parser.add_argument("--budget", type=float, default=0.08)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--precisions", default="int8,int4")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--packing-seed", type=int, default=911)
    parser.add_argument("--compact-prefill-chunk-size", type=int, default=512)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def cache_bytes(cache: Any) -> int:
    return sum(tensor_bytes(key) + tensor_bytes(value) for key, value in legacy_cache_layers(cache))


@dataclass
class QuantizedSlice:
    packed: torch.Tensor
    scale: torch.Tensor
    precision: str
    original_dim: int

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, precision: str) -> "QuantizedSlice":
        qmax = 127 if precision == "int8" else 7
        scale = tensor.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        quantized = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
        if precision == "int8":
            packed = quantized
        elif precision == "int4":
            if tensor.shape[-1] % 2:
                raise ValueError("INT4 pilot requires an even head dimension")
            shifted = (quantized.to(torch.int16) + 8).to(torch.uint8)
            packed = shifted[..., 0::2] | (shifted[..., 1::2] << 4)
        else:
            raise ValueError(precision)
        return cls(packed=packed, scale=scale.to(tensor.dtype), precision=precision, original_dim=tensor.shape[-1])

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        if self.precision == "int8":
            quantized = self.packed.to(torch.int8)
        else:
            low = (self.packed & 0x0F).to(torch.int16) - 8
            high = ((self.packed >> 4) & 0x0F).to(torch.int16) - 8
            quantized = torch.stack((low, high), dim=-1).reshape(*self.packed.shape[:-1], self.original_dim)
        return quantized.to(dtype) * self.scale

    def storage_bytes(self) -> int:
        return tensor_bytes(self.packed) + tensor_bytes(self.scale)


@dataclass
class TieredLayer:
    key_bf16: torch.Tensor
    value_bf16: torch.Tensor
    key_quantized: QuantizedSlice
    value_quantized: QuantizedSlice

    def storage_bytes(self) -> int:
        return (
            tensor_bytes(self.key_bf16)
            + tensor_bytes(self.value_bf16)
            + self.key_quantized.storage_bytes()
            + self.value_quantized.storage_bytes()
        )


class TieredStorage:
    def __init__(
        self,
        cache: Any,
        auxiliary_positions: Sequence[int],
        precision: str,
    ) -> None:
        layers = legacy_cache_layers(cache)
        if not layers:
            raise ValueError("empty cache")
        seq_len = int(layers[0][0].shape[-2])
        device = layers[0][0].device
        aux = sorted(set(map(int, auxiliary_positions)))
        if any(position < 0 or position >= seq_len for position in aux):
            raise ValueError("auxiliary position outside compact cache")
        aux_set = set(aux)
        bf16 = [position for position in range(seq_len) if position not in aux_set]
        self.bf16_positions = torch.tensor(bf16, dtype=torch.long, device=device)
        self.auxiliary_positions = torch.tensor(aux, dtype=torch.long, device=device)
        self.seq_len = seq_len
        self.layers: list[TieredLayer] = []
        for key, value in layers:
            key_bf16 = key.index_select(-2, self.bf16_positions).contiguous()
            value_bf16 = value.index_select(-2, self.bf16_positions).contiguous()
            key_aux = key.index_select(-2, self.auxiliary_positions).contiguous()
            value_aux = value.index_select(-2, self.auxiliary_positions).contiguous()
            self.layers.append(
                TieredLayer(
                    key_bf16,
                    value_bf16,
                    QuantizedSlice.from_tensor(key_aux, precision),
                    QuantizedSlice.from_tensor(value_aux, precision),
                )
            )

    def storage_bytes(self) -> int:
        # Positions are shared across layers and are included exactly once.
        return (
            tensor_bytes(self.bf16_positions)
            + tensor_bytes(self.auxiliary_positions)
            + sum(layer.storage_bytes() for layer in self.layers)
        )

    def materialize(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        materialized = []
        for layer in self.layers:
            shape = (*layer.key_bf16.shape[:-2], self.seq_len, layer.key_bf16.shape[-1])
            key = torch.empty(shape, dtype=layer.key_bf16.dtype, device=layer.key_bf16.device)
            value = torch.empty(shape, dtype=layer.value_bf16.dtype, device=layer.value_bf16.device)
            key.index_copy_(-2, self.bf16_positions, layer.key_bf16)
            value.index_copy_(-2, self.bf16_positions, layer.value_bf16)
            if self.auxiliary_positions.numel():
                key.index_copy_(
                    -2,
                    self.auxiliary_positions,
                    layer.key_quantized.dequantize(layer.key_bf16.dtype),
                )
                value.index_copy_(
                    -2,
                    self.auxiliary_positions,
                    layer.value_quantized.dequantize(layer.value_bf16.dtype),
                )
            materialized.append((key, value))
        return tuple(materialized)

    def append_bf16(self, cache: Any) -> None:
        layers = legacy_cache_layers(cache)
        if len(layers) != len(self.layers):
            raise ValueError("layer-count mismatch while appending generated KV")
        position = torch.tensor([self.seq_len], dtype=torch.long, device=self.bf16_positions.device)
        self.bf16_positions = torch.cat((self.bf16_positions, position))
        for storage_layer, (key, value) in zip(self.layers, layers):
            storage_layer.key_bf16 = torch.cat((storage_layer.key_bf16, key[..., -1:, :].contiguous()), dim=-2)
            storage_layer.value_bf16 = torch.cat(
                (storage_layer.value_bf16, value[..., -1:, :].contiguous()), dim=-2
            )
        self.seq_len += 1


@torch.inference_mode()
def tiered_decode(
    model: Any,
    tokenizer: Any,
    storage: TieredStorage,
    logits: torch.Tensor,
    prompt_len: int,
    compact_len: int,
    max_new_tokens: int,
) -> tuple[str, float, float]:
    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated: list[int] = []
    materialize_seconds = 0.0
    decode_seconds = 0.0
    for step in range(max_new_tokens):
        token = int(next_id[0, 0].item())
        generated.append(token)
        if tokenizer.eos_token_id is not None and token == int(tokenizer.eos_token_id):
            break
        if step + 1 >= max_new_tokens:
            break
        torch.cuda.synchronize()
        materialize_started = time.perf_counter()
        legacy = storage.materialize()
        cache = DynamicCache(ddp_cache_data=legacy)
        torch.cuda.synchronize()
        materialize_seconds += time.perf_counter() - materialize_started
        decode_started = time.perf_counter()
        out = _step_with_cache(
            model,
            next_id,
            cache,
            prompt_len + step,
            [],
            compact_len + step,
        )
        torch.cuda.synchronize()
        decode_seconds += time.perf_counter() - decode_started
        storage.append_bf16(out.past_key_values)
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        del legacy, cache, out
    return _decode_text(tokenizer, generated), materialize_seconds, decode_seconds


def main() -> None:
    args = parse_args()
    precisions = [value.strip() for value in args.precisions.split(",") if value.strip()]
    if not precisions or set(precisions) - {"int8", "int4"}:
        raise ValueError("precisions must be int8 and/or int4")
    if args.max_samples <= 0 or args.sample_offset < 0:
        raise ValueError("max-samples must be positive and sample-offset nonnegative")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    metrics_path = out_dir / "metrics.json"
    if results_path.exists() or metrics_path.exists():
        raise FileExistsError(out_dir)

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    selected = order[args.sample_offset : args.sample_offset + args.max_samples]
    if len(selected) != args.max_samples:
        raise ValueError("requested precision sample slice exceeds dataset")
    model, tokenizer = load_model_tokenizer(
        args.model, dtype="bfloat16", attn_implementation=args.attn_implementation
    )
    scorer, scorer_tokenizer = TypedRelationScorer.load_checkpoint(
        args.scorer_checkpoint, device=next(model.parameters()).device
    )
    learned_bundle = ("cross", scorer, scorer_tokenizer)
    pool = build_natural_distractor_pool(dataset)
    overlay_cache: dict[str, tuple[Any, Any]] = {}
    device = next(model.parameters()).device
    records = []

    for dataset_idx in selected:
        source = normalize_ircot_source(dict(dataset[dataset_idx]))
        source, _ = pack_source_to_prompt_target(
            source,
            tokenizer,
            pool,
            args.target_prompt_tokens,
            args.packing_seed,
            False,
        )
        prompt, sentences, query = make_prompt(source, tokenizer)
        encoding = tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt"
        )
        offsets = [(int(start), int(end)) for start, end in encoding.pop("offset_mapping")[0].tolist()]
        input_ids = encoding["input_ids"].to(device)
        partition: dict[str, Any] = {}
        plan, _, _ = make_plan(
            "learned_and_or_aux",
            source,
            prompt,
            query,
            sentences,
            offsets,
            args.budget,
            learned_bundle,
            learned_dense_core_count=4,
            learned_edge_batch_size=256,
            learned_overlay_cache=overlay_cache,
            selection_metadata_out=partition,
            include_selection_indices=True,
        )

        warnings: list[str] = []
        baseline_prefill, prepared = chunked_compact_prefill(
            model,
            input_ids,
            plan.retained_indices,
            warnings,
            args.compact_prefill_chunk_size,
        )
        baseline_bytes = cache_bytes(baseline_prefill.past_key_values)
        baseline_cache = DynamicCache(
            ddp_cache_data=tuple(
                (key.clone(), value.clone())
                for key, value in legacy_cache_layers(baseline_prefill.past_key_values)
            )
        )
        baseline_text, baseline_final_cache, _ = _greedy_decode_from_prefill_logits(
            model,
            tokenizer,
            baseline_cache,
            baseline_prefill.logits,
            int(input_ids.shape[1]),
            int(prepared.input_ids.shape[1]),
            args.max_new_tokens,
        )
        baseline_short = extract_short_answer(baseline_text)
        _, baseline_f1, _ = answer_scores(baseline_short, str(source["answer"]))
        retained_original = [int(value) for value in prepared.retained_indices.tolist()]
        compact_len = int(prepared.input_ids.shape[1])
        auxiliary_original = set(map(int, partition["auxiliary_indices"]))
        auxiliary_positions = [
            position for position, original in enumerate(retained_original) if original in auxiliary_original
        ]
        del baseline_final_cache, baseline_cache, baseline_prefill, prepared
        _release_cuda()

        for precision in precisions:
            warnings = []
            prefill, prepared_precision = chunked_compact_prefill(
                model,
                input_ids,
                plan.retained_indices,
                warnings,
                args.compact_prefill_chunk_size,
            )
            storage = TieredStorage(prefill.past_key_values, auxiliary_positions, precision)
            storage_bytes_initial = storage.storage_bytes()
            logits = prefill.logits[:, -1:, :].clone()
            del prefill
            _release_cuda()
            torch.cuda.reset_peak_memory_stats()
            allocated_before_decode = torch.cuda.memory_allocated()
            tiered_text, materialize_seconds, decode_seconds = tiered_decode(
                model,
                tokenizer,
                storage,
                logits,
                int(input_ids.shape[1]),
                compact_len,
                args.max_new_tokens,
            )
            peak_allocated = torch.cuda.max_memory_allocated()
            tiered_short = extract_short_answer(tiered_text)
            _, tiered_f1, _ = answer_scores(tiered_short, str(source["answer"]))
            row = {
                "id": str(source["id"]),
                "precision": precision,
                "prompt_tokens": int(input_ids.shape[1]),
                "retained_tokens": len(retained_original),
                "auxiliary_tokens": len(auxiliary_positions),
                "proof_core_tokens": int(partition["proof_core_tokens"]),
                "baseline_cache_bytes": baseline_bytes,
                "tiered_storage_bytes_initial": storage_bytes_initial,
                "storage_reduction_factor": baseline_bytes / storage_bytes_initial,
                "allocated_before_decode_bytes": allocated_before_decode,
                "peak_allocated_during_decode_bytes": peak_allocated,
                "peak_extra_during_decode_bytes": peak_allocated - allocated_before_decode,
                "materialize_seconds": materialize_seconds,
                "decode_seconds": decode_seconds,
                "answer": source["answer"],
                "baseline_prediction": baseline_text,
                "tiered_prediction": tiered_text,
                "exact_generation_match": float(tiered_text == baseline_text),
                "baseline_answer_f1": baseline_f1,
                "tiered_answer_f1": tiered_f1,
            }
            records.append(row)
            append_jsonl(results_path, row)
            del storage, logits, prepared_precision
            _release_cuda()

        del input_ids
        _release_cuda()

    summaries = []
    for precision in precisions:
        values = [row for row in records if row["precision"] == precision]
        baseline_f1 = sum(row["baseline_answer_f1"] for row in values) / len(values)
        tiered_f1 = sum(row["tiered_answer_f1"] for row in values) / len(values)
        exact_rate = sum(row["exact_generation_match"] for row in values) / len(values)
        storage_reduction = sum(row["storage_reduction_factor"] for row in values) / len(values)
        gates = {
            "mean_storage_reduction_at_least_1_2": storage_reduction >= 1.2,
            "answer_f1_drop_at_most_0_02": tiered_f1 >= baseline_f1 - 0.02,
            "exact_generation_rate_at_least_0_875": exact_rate >= 0.875,
        }
        summaries.append(
            {
                "precision": precision,
                "n": len(values),
                "mean_storage_reduction_factor": storage_reduction,
                "exact_generation_match_rate": exact_rate,
                "baseline_answer_f1": baseline_f1,
                "tiered_answer_f1": tiered_f1,
                "answer_f1_delta": tiered_f1 - baseline_f1,
                "mean_materialize_seconds": sum(row["materialize_seconds"] for row in values)
                / len(values),
                "mean_decode_seconds": sum(row["decode_seconds"] for row in values) / len(values),
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    payload = {
        "model": args.model,
        "scorer_checkpoint": args.scorer_checkpoint,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "max_samples": args.max_samples,
        "sample_offset": args.sample_offset,
        "selected_indices": selected,
        "target_prompt_tokens": args.target_prompt_tokens,
        "budget": args.budget,
        "max_new_tokens": args.max_new_tokens,
        "precisions": precisions,
        "seed": args.seed,
        "packing_seed": args.packing_seed,
        "compact_prefill_chunk_size": args.compact_prefill_chunk_size,
        "summaries": summaries,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"summaries": summaries}, indent=2))
    if not any(summary["passed"] for summary in summaries):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
