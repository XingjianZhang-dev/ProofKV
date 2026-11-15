#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
import string
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pynvml
import torch
import vllm
from transformers import AutoConfig
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProofCache on the real vLLM paged-KV backend.")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--expected-vllm-version", default="0.25.0")
    parser.add_argument(
        "--accuracy-methods",
        default="full,learned_topk_paths,learned_and_or,learned_and_or_aux",
    )
    parser.add_argument("--throughput-methods", default="full,learned_and_or_aux")
    parser.add_argument("--route-method", default="learned_and_or_aux")
    parser.add_argument("--accuracy-samples", type=int, default=32)
    parser.add_argument("--throughput-samples", type=int, default=16)
    parser.add_argument("--accuracy-output-tokens", type=int, default=24)
    parser.add_argument("--throughput-output-tokens", default="256,1024")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    words = [word for word in text.split() if word not in {"a", "an", "the"}]
    return " ".join(words)


def extract_short_answer(prediction: str) -> str:
    for line in prediction.splitlines():
        line = line.strip()
        if line:
            lowered = line.lower()
            if lowered.startswith("answer:"):
                line = line.split(":", 1)[1].strip()
            return line
    return prediction.strip()


def answer_scores(prediction: str, answer: str) -> tuple[float, float, float]:
    pred, gold = normalize(prediction), normalize(answer)
    em = float(pred == gold)
    pred_tokens, gold_tokens = pred.split(), gold.split()
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not pred_tokens or not gold_tokens:
        f1 = float(pred_tokens == gold_tokens)
    elif overlap == 0:
        f1 = 0.0
    else:
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    return em, f1, float(bool(gold) and gold in pred)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("empty percentile")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
    return float(ordered[index])


class NvmlSampler:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples: list[int] = []
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> int:
        return int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self.samples.append(self._read())

    def __enter__(self) -> "NvmlSampler":
        self.samples.append(self._read())
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.samples.append(self._read())
        self.stop_event.set()
        self.thread.join()
        pynvml.nvmlShutdown()

    @property
    def peak_bytes(self) -> int:
        return max(self.samples)

    @property
    def minimum_bytes(self) -> int:
        return min(self.samples)


def load_inputs(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty paged-input file")
    keys = {(str(row["id"]), str(row["method"])) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate sample/method inputs")
    return rows


def select_rows(
    rows: list[dict[str, Any]], method: str, sample_ids: list[str], limit: int
) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): row for row in rows if row["method"] == method}
    selected = [by_id[sample_id] for sample_id in sample_ids[:limit] if sample_id in by_id]
    if len(selected) != min(limit, len(sample_ids)):
        raise ValueError(f"missing prepared inputs for {method}")
    return selected


def request_record(
    scenario: str,
    method: str,
    repeat: int,
    source: dict[str, Any],
    output: Any,
    block_size: int,
    bytes_per_token: int,
) -> dict[str, Any]:
    completion = output.outputs[0]
    generated = len(completion.token_ids)
    prompt_tokens = len(source["prompt_token_ids"])
    final_tokens = prompt_tokens + generated
    slots = math.ceil(final_tokens / block_size) * block_size
    metrics = output.metrics
    if metrics is None:
        raise RuntimeError("vLLM did not return per-request metrics")
    short = extract_short_answer(completion.text)
    em, f1, contains = answer_scores(short, str(source["answer"]))
    return {
        "scenario": scenario,
        "method": method,
        "repeat": repeat,
        "id": str(source["id"]),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "text": completion.text,
        "finish_reason": completion.finish_reason,
        "answer": source["answer"],
        "short_prediction": short,
        "answer_em": em,
        "answer_f1": f1,
        "answer_contains": contains,
        "first_token_latency_seconds": float(metrics.first_token_latency),
        "queue_to_first_token_seconds": float(metrics.first_token_ts - metrics.queued_ts),
        "scheduled_to_first_token_seconds": float(metrics.first_token_ts - metrics.scheduled_ts),
        "decode_span_seconds": float(metrics.last_token_ts - metrics.first_token_ts),
        "scheduled_to_last_token_seconds": float(metrics.last_token_ts - metrics.scheduled_ts),
        "is_corrupted": bool(metrics.is_corrupted),
        "num_cached_tokens": int(output.num_cached_tokens or 0),
        "paged_block_size": block_size,
        "active_paged_slots_final": slots,
        "active_paged_kv_bytes_final": slots * bytes_per_token,
        "paged_internal_fragmentation_tokens_final": slots - final_tokens,
    }


def run_batch(
    llm: LLM,
    scenario: str,
    method: str,
    repeat: int,
    sources: list[dict[str, Any]],
    max_tokens: int,
    ignore_eos: bool,
    seed: int,
    block_size: int,
    bytes_per_token: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = [{"prompt_token_ids": row["prompt_token_ids"]} for row in sources]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
        seed=seed,
    )
    with NvmlSampler() as memory:
        started = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        wall_seconds = time.perf_counter() - started
    if len(outputs) != len(sources):
        raise RuntimeError("vLLM output count differs from request count")
    records = [
        request_record(
            scenario,
            method,
            repeat,
            source,
            output,
            block_size,
            bytes_per_token,
        )
        for source, output in zip(sources, outputs)
    ]
    output_tokens = sum(row["generated_tokens"] for row in records)
    prompt_tokens = sum(row["prompt_tokens"] for row in records)
    if ignore_eos and output_tokens != len(sources) * max_tokens:
        raise RuntimeError("fixed-output paged benchmark did not emit the requested token count")
    batch = {
        "scenario": scenario,
        "method": method,
        "repeat": repeat,
        "requests": len(sources),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "wall_seconds": wall_seconds,
        "prompt_tokens_per_second": prompt_tokens / wall_seconds,
        "output_tokens_per_second": output_tokens / wall_seconds,
        "requests_per_second": len(sources) / wall_seconds,
        "nvml_min_used_bytes": memory.minimum_bytes,
        "nvml_peak_used_bytes": memory.peak_bytes,
    }
    return records, batch


def summarize_requests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_request_records": len(rows),
        "answer_em": statistics.fmean(row["answer_em"] for row in rows),
        "answer_f1": statistics.fmean(row["answer_f1"] for row in rows),
        "answer_contains": statistics.fmean(row["answer_contains"] for row in rows),
        "mean_prompt_tokens": statistics.fmean(row["prompt_tokens"] for row in rows),
        "mean_generated_tokens": statistics.fmean(row["generated_tokens"] for row in rows),
        "ttft_p50_seconds": percentile(
            [row["scheduled_to_first_token_seconds"] for row in rows], 0.5
        ),
        "ttft_p95_seconds": percentile(
            [row["scheduled_to_first_token_seconds"] for row in rows], 0.95
        ),
        "e2e_p50_seconds": percentile(
            [row["scheduled_to_last_token_seconds"] for row in rows], 0.5
        ),
        "e2e_p95_seconds": percentile(
            [row["scheduled_to_last_token_seconds"] for row in rows], 0.95
        ),
        "mean_active_paged_kv_bytes_final": statistics.fmean(
            row["active_paged_kv_bytes_final"] for row in rows
        ),
        "mean_fragmentation_tokens_final": statistics.fmean(
            row["paged_internal_fragmentation_tokens_final"] for row in rows
        ),
        "corrupted_requests": sum(row["is_corrupted"] for row in rows),
        "cached_prompt_tokens": sum(row["num_cached_tokens"] for row in rows),
    }


def summarize_batches(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_batches": len(rows),
        "wall_p50_seconds": percentile([row["wall_seconds"] for row in rows], 0.5),
        "wall_p95_seconds": percentile([row["wall_seconds"] for row in rows], 0.95),
        "output_tokens_per_second_p50": percentile(
            [row["output_tokens_per_second"] for row in rows], 0.5
        ),
        "output_tokens_per_second_p95": percentile(
            [row["output_tokens_per_second"] for row in rows], 0.95
        ),
        "requests_per_second_p50": percentile(
            [row["requests_per_second"] for row in rows], 0.5
        ),
        "nvml_peak_used_bytes_max": max(row["nvml_peak_used_bytes"] for row in rows),
    }


def main() -> None:
    args = parse_args()
    if vllm.__version__ != args.expected_vllm_version:
        raise RuntimeError(f"vLLM version drift: {vllm.__version__} != {args.expected_vllm_version}")
    accuracy_methods = [value.strip() for value in args.accuracy_methods.split(",") if value.strip()]
    throughput_methods = [
        value.strip() for value in args.throughput_methods.split(",") if value.strip()
    ]
    output_lengths = [int(value) for value in args.throughput_output_tokens.split(",")]
    if (
        not accuracy_methods
        or not throughput_methods
        or any(value <= 0 for value in output_lengths)
        or args.accuracy_samples <= 0
        or args.throughput_samples <= 0
        or args.repeats <= 0
    ):
        raise ValueError("invalid paged-suite protocol")
    if args.route_method not in accuracy_methods or args.route_method not in throughput_methods:
        raise ValueError("route method must be included in accuracy and throughput methods")
    if args.block_size <= 0 or args.max_model_len <= max(output_lengths):
        raise ValueError("invalid paged block/model length")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    request_path = out_dir / "request_results.jsonl"
    batch_path = out_dir / "batch_results.jsonl"
    if metrics_path.exists() or request_path.exists() or batch_path.exists():
        raise FileExistsError(out_dir)
    inputs_path = Path(args.inputs)
    inputs = load_inputs(inputs_path)
    sample_ids = []
    for row in inputs:
        sample_id = str(row["id"])
        if sample_id not in sample_ids:
            sample_ids.append(sample_id)

    config = AutoConfig.from_pretrained(args.model)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    kv_element_bytes = 1 if args.kv_cache_dtype.startswith("fp8") else 2
    bytes_per_token = (
        int(config.num_hidden_layers) * kv_heads * head_dim * 2 * kv_element_bytes
    )

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.block_size,
        enable_prefix_caching=False,
        enforce_eager=True,
        # vLLM's offline LLM wrapper defaults this to True, which suppresses
        # RequestOutput.metrics even though the fields exist in the installed
        # API.  The paged-cache suite needs those per-request timestamps for
        # TTFT/decode attribution and corruption checks.
        disable_log_stats=False,
        seed=args.seed,
    )

    # A real two-request engine smoke runs after initialization and before the
    # frozen suite. It is not reused as a measured result.
    smoke_method = args.route_method
    smoke_sources = select_rows(inputs, smoke_method, sample_ids, 2)
    smoke_records, smoke_batch = run_batch(
        llm,
        "smoke",
        smoke_method,
        -1,
        smoke_sources,
        2,
        True,
        args.seed,
        args.block_size,
        bytes_per_token,
    )
    if any(row["generated_tokens"] != 2 or row["is_corrupted"] for row in smoke_records):
        raise RuntimeError("vLLM paged smoke failed")

    request_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    accuracy_selected = {
        method: select_rows(inputs, method, sample_ids, args.accuracy_samples)
        for method in accuracy_methods
    }
    for method in accuracy_methods:
        records, batch = run_batch(
            llm,
            "accuracy",
            method,
            0,
            accuracy_selected[method],
            args.accuracy_output_tokens,
            False,
            args.seed,
            args.block_size,
            bytes_per_token,
        )
        request_rows.extend(records)
        batch_rows.append(batch)

    throughput_selected = {
        method: select_rows(inputs, method, sample_ids, args.throughput_samples)
        for method in throughput_methods
    }
    for output_tokens in output_lengths:
        scenario = f"throughput_out{output_tokens}"
        for method in throughput_methods:
            for _ in range(args.warmups):
                run_batch(
                    llm,
                    f"warmup_out{output_tokens}",
                    method,
                    -1,
                    throughput_selected[method][:2],
                    output_tokens,
                    True,
                    args.seed,
                    args.block_size,
                    bytes_per_token,
                )
        for repeat in range(args.repeats):
            order = throughput_methods if repeat % 2 == 0 else list(reversed(throughput_methods))
            for method in order:
                records, batch = run_batch(
                    llm,
                    scenario,
                    method,
                    repeat,
                    throughput_selected[method],
                    output_tokens,
                    True,
                    args.seed,
                    args.block_size,
                    bytes_per_token,
                )
                request_rows.extend(records)
                batch_rows.append(batch)

    with request_path.open("w") as handle:
        for row in request_rows:
            handle.write(json.dumps(row) + "\n")
    with batch_path.open("w") as handle:
        for row in batch_rows:
            handle.write(json.dumps(row) + "\n")

    grouped_requests: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_batches: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        grouped_requests[(row["scenario"], row["method"])].append(row)
    for row in batch_rows:
        grouped_batches[(row["scenario"], row["method"])].append(row)
    records = []
    for key in sorted(grouped_requests):
        scenario, method = key
        records.append(
            {
                "scenario": scenario,
                "method": method,
                **summarize_requests(grouped_requests[key]),
                "batch": summarize_batches(grouped_batches[key]),
            }
        )
    by_key = {(row["scenario"], row["method"]): row for row in records}
    gates: dict[str, Any] = {}
    accuracy_full = by_key[("accuracy", "full")]
    accuracy_route = by_key[("accuracy", args.route_method)]
    relative_accuracy = (
        accuracy_route["answer_f1"] / accuracy_full["answer_f1"]
        if accuracy_full["answer_f1"] > 0
        else None
    )
    absolute_drop = accuracy_full["answer_f1"] - accuracy_route["answer_f1"]
    gates["accuracy"] = {
        "route_f1": accuracy_route["answer_f1"],
        "full_f1": accuracy_full["answer_f1"],
        "relative_f1_to_full": relative_accuracy,
        "absolute_f1_drop": absolute_drop,
        "relative_f1_at_least_0_90": (
            relative_accuracy is not None and relative_accuracy >= 0.90
        ),
        "absolute_f1_drop_at_most_0_02": absolute_drop <= 0.02,
        "no_corrupted_requests": (
            accuracy_full["corrupted_requests"] == 0
            and accuracy_route["corrupted_requests"] == 0
        ),
    }
    gates["accuracy"]["passed"] = (
        gates["accuracy"]["relative_f1_at_least_0_90"]
        and gates["accuracy"]["absolute_f1_drop_at_most_0_02"]
        and gates["accuracy"]["no_corrupted_requests"]
    )
    for output_tokens in output_lengths:
        scenario = f"throughput_out{output_tokens}"
        full = by_key[(scenario, "full")]
        route = by_key[(scenario, args.route_method)]
        throughput_speedup = (
            route["batch"]["output_tokens_per_second_p50"]
            / full["batch"]["output_tokens_per_second_p50"]
        )
        active_byte_reduction = (
            full["mean_active_paged_kv_bytes_final"]
            / route["mean_active_paged_kv_bytes_final"]
        )
        gates[scenario] = {
            "route_over_full_output_throughput": throughput_speedup,
            "full_over_route_active_paged_bytes": active_byte_reduction,
            "throughput_at_least_1_5x": throughput_speedup >= 1.5,
            "active_paged_bytes_at_least_4x_lower": active_byte_reduction >= 4.0,
            "no_corrupted_requests": (
                full["corrupted_requests"] == 0 and route["corrupted_requests"] == 0
            ),
        }
        gates[scenario]["passed"] = all(
            value for name, value in gates[scenario].items() if name.endswith("lower") or name.startswith("throughput_at") or name.startswith("no_corrupted")
        )
    payload = {
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "model": args.model,
        "inputs": str(inputs_path),
        "protocol": vars(args),
        "paged_kv_layout": {
            "block_size": args.block_size,
            "layers": int(config.num_hidden_layers),
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "kv_element_bytes": kv_element_bytes,
            "kv_bytes_per_token": bytes_per_token,
            "note": (
                "NVML reports the fixed reserved vLLM pool; active paged bytes count occupied "
                "scheduler block slots, including block-fragmentation padding."
            ),
        },
        "smoke": {"request_records": smoke_records, "batch": smoke_batch},
        "records": records,
        "acceptance_gates": gates,
        "passed": all(value["passed"] for value in gates.values()),
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"records": records, "acceptance_gates": gates}, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
