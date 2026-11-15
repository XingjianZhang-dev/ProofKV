#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ACCURACY_METHODS = ("full", "learned_topk_paths", "learned_and_or")
THROUGHPUT_METHODS = ("full", "learned_and_or")
OUTPUT_LENGTHS = (256, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent vLLM paged-suite audit.")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    if values.ndim != 1 or len(values) == 0 or draws <= 0:
        raise ValueError("invalid bootstrap input")
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    chunk = 4096
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        sampled[start:stop] = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [float(value) for value in np.quantile(sampled, [0.025, 0.975])],
        "bootstrap_draws": draws,
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    inputs = read_jsonl(args.inputs)
    metrics = json.loads(Path(args.metrics).read_text())
    requests = read_jsonl(args.requests)
    batches = read_jsonl(args.batches)
    protocol = metrics.get("protocol", {})
    layout = metrics.get("paged_kv_layout", {})
    block_size = int(layout.get("block_size", -1))
    bytes_per_token = int(layout.get("kv_bytes_per_token", -1))

    issues: list[str] = []
    input_keys = [(str(row.get("id")), str(row.get("method"))) for row in inputs]
    input_ids = list(dict.fromkeys(key[0] for key in input_keys))
    input_by_key = {(str(row["id"]), str(row["method"])): row for row in inputs}
    if len(inputs) != 96 or len(set(input_keys)) != 96:
        issues.append("inputs_expected_96_unique_id_method_rows")
    if len(input_ids) != 32:
        issues.append(f"inputs_expected_32_ids:{len(input_ids)}")
    if Counter(method for _, method in input_keys) != Counter({method: 32 for method in ACCURACY_METHODS}):
        issues.append("input_method_counts_mismatch")
    for row in inputs:
        if int(row.get("paged_prompt_tokens", -1)) != len(row.get("prompt_token_ids", [])):
            issues.append(f"input_prompt_length_mismatch:{row.get('id')}:{row.get('method')}")

    expected_protocol = {
        "expected_vllm_version": "0.25.0",
        "route_method": "learned_and_or",
        "accuracy_samples": 32,
        "throughput_samples": 16,
        "accuracy_output_tokens": 24,
        "warmups": 3,
        "repeats": 10,
        "block_size": 16,
        "max_model_len": 32768,
        "enable_prefix_caching": None,
    }
    for key, expected in expected_protocol.items():
        if key == "enable_prefix_caching":
            continue
        if protocol.get(key) != expected:
            issues.append(f"protocol_mismatch:{key}:{protocol.get(key)}!={expected}")
    if protocol.get("accuracy_methods") != ",".join(ACCURACY_METHODS):
        issues.append("protocol_accuracy_methods_mismatch")
    if protocol.get("throughput_methods") != ",".join(THROUGHPUT_METHODS):
        issues.append("protocol_throughput_methods_mismatch")
    if protocol.get("throughput_output_tokens") != "256,1024":
        issues.append("protocol_output_lengths_mismatch")
    if metrics.get("vllm_version") != "0.25.0":
        issues.append("metrics_vllm_version_mismatch")
    if block_size != 16 or bytes_per_token <= 0:
        issues.append("invalid_paged_layout")

    if len(requests) != 736:
        issues.append(f"request_count:{len(requests)}!=736")
    if len(batches) != 43:
        issues.append(f"batch_count:{len(batches)}!=43")

    accuracy_requests = [row for row in requests if row.get("scenario") == "accuracy"]
    if len(accuracy_requests) != 96:
        issues.append(f"accuracy_request_count:{len(accuracy_requests)}!=96")
    accuracy_keys = [
        (str(row.get("id")), str(row.get("method")), int(row.get("repeat", -1)))
        for row in accuracy_requests
    ]
    if len(set(accuracy_keys)) != len(accuracy_keys) or any(key[2] != 0 for key in accuracy_keys):
        issues.append("accuracy_request_keys_not_unique_repeat_zero")

    throughput_requests = [row for row in requests if str(row.get("scenario", "")).startswith("throughput_out")]
    expected_throughput_keys = {
        (f"throughput_out{output}", method, repeat, sample_id)
        for output in OUTPUT_LENGTHS
        for method in THROUGHPUT_METHODS
        for repeat in range(10)
        for sample_id in input_ids[:16]
    }
    observed_throughput_keys = {
        (str(row.get("scenario")), str(row.get("method")), int(row.get("repeat", -1)), str(row.get("id")))
        for row in throughput_requests
    }
    if observed_throughput_keys != expected_throughput_keys or len(throughput_requests) != len(expected_throughput_keys):
        issues.append("throughput_request_key_matrix_mismatch")

    expected_batch_keys = {
        ("accuracy", method, 0) for method in ACCURACY_METHODS
    } | {
        (f"throughput_out{output}", method, repeat)
        for output in OUTPUT_LENGTHS
        for method in THROUGHPUT_METHODS
        for repeat in range(10)
    }
    observed_batch_keys = [
        (str(row.get("scenario")), str(row.get("method")), int(row.get("repeat", -1)))
        for row in batches
    ]
    if set(observed_batch_keys) != expected_batch_keys or len(observed_batch_keys) != len(set(observed_batch_keys)):
        issues.append("batch_key_matrix_mismatch")

    finite_request_fields = (
        "first_token_latency_seconds",
        "queue_to_first_token_seconds",
        "scheduled_to_first_token_seconds",
        "decode_span_seconds",
        "scheduled_to_last_token_seconds",
    )
    for row in requests:
        label = f"{row.get('scenario')}:{row.get('method')}:{row.get('repeat')}:{row.get('id')}"
        source = input_by_key.get((str(row.get("id")), str(row.get("method"))))
        if source is None:
            issues.append(f"request_missing_input:{label}")
            continue
        if int(row.get("prompt_tokens", -1)) != int(source.get("paged_prompt_tokens", -2)):
            issues.append(f"request_prompt_length_mismatch:{label}")
        values = [float(row.get(key, float("nan"))) for key in finite_request_fields]
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            issues.append(f"request_invalid_timing:{label}")
        if values[4] + 1e-9 < values[2] or values[4] + 1e-9 < values[3]:
            issues.append(f"request_inconsistent_timing:{label}")
        generated = int(row.get("generated_tokens", -1))
        scenario = str(row.get("scenario"))
        if scenario.startswith("throughput_out") and generated != int(scenario.removeprefix("throughput_out")):
            issues.append(f"request_fixed_output_mismatch:{label}")
        final_tokens = int(row.get("prompt_tokens", -1)) + generated
        expected_slots = math.ceil(final_tokens / block_size) * block_size
        if int(row.get("active_paged_slots_final", -1)) != expected_slots:
            issues.append(f"request_slot_formula_mismatch:{label}")
        if int(row.get("active_paged_kv_bytes_final", -1)) != expected_slots * bytes_per_token:
            issues.append(f"request_byte_formula_mismatch:{label}")
        if int(row.get("paged_internal_fragmentation_tokens_final", -1)) != expected_slots - final_tokens:
            issues.append(f"request_fragmentation_formula_mismatch:{label}")
        if bool(row.get("is_corrupted")):
            issues.append(f"request_corrupted:{label}")
        if int(row.get("num_cached_tokens", -1)) != 0:
            issues.append(f"unexpected_cached_prompt_tokens:{label}")
        for score in ("answer_em", "answer_f1", "answer_contains"):
            value = float(row.get(score, float("nan")))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                issues.append(f"request_invalid_score:{score}:{label}")

    for row in batches:
        label = f"{row.get('scenario')}:{row.get('method')}:{row.get('repeat')}"
        for key in ("wall_seconds", "prompt_tokens_per_second", "output_tokens_per_second", "requests_per_second"):
            value = float(row.get(key, float("nan")))
            if not math.isfinite(value) or value <= 0.0:
                issues.append(f"batch_invalid_metric:{key}:{label}")
        if int(row.get("nvml_peak_used_bytes", -1)) < int(row.get("nvml_min_used_bytes", 0)):
            issues.append(f"batch_nvml_range_invalid:{label}")

    accuracy_by_key = {
        (str(row["id"]), str(row["method"])): row for row in accuracy_requests
    }
    full_accuracy = np.asarray(
        [float(accuracy_by_key[(sample_id, "full")]["answer_f1"]) for sample_id in input_ids],
        dtype=np.float64,
    )
    route_accuracy = np.asarray(
        [float(accuracy_by_key[(sample_id, "learned_and_or")]["answer_f1"]) for sample_id in input_ids],
        dtype=np.float64,
    )
    topk_accuracy = np.asarray(
        [float(accuracy_by_key[(sample_id, "learned_topk_paths")]["answer_f1"]) for sample_id in input_ids],
        dtype=np.float64,
    )
    accuracy_statistics = {
        "route_minus_full_f1": bootstrap_mean(route_accuracy - full_accuracy, args.bootstrap_draws, args.seed),
        "route_minus_topk_f1": bootstrap_mean(route_accuracy - topk_accuracy, args.bootstrap_draws, args.seed + 1),
        "route_topk_exact_generation_match_rate": float(np.mean([
            accuracy_by_key[(sample_id, "learned_and_or")]["text"]
            == accuracy_by_key[(sample_id, "learned_topk_paths")]["text"]
            for sample_id in input_ids
        ])),
    }

    batch_by_key = {
        (str(row["scenario"]), str(row["method"]), int(row["repeat"])): row
        for row in batches
    }
    throughput_statistics: dict[str, Any] = {}
    throughput_request_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in throughput_requests:
        throughput_request_groups[(str(row["scenario"]), str(row["method"]))].append(row)
    for offset, output in enumerate(OUTPUT_LENGTHS, start=1):
        scenario = f"throughput_out{output}"
        ratios = np.asarray(
            [
                float(batch_by_key[(scenario, "full", repeat)]["wall_seconds"])
                / float(batch_by_key[(scenario, "learned_and_or", repeat)]["wall_seconds"])
                for repeat in range(10)
            ],
            dtype=np.float64,
        )
        full_bytes = np.mean([
            float(row["active_paged_kv_bytes_final"])
            for row in throughput_request_groups[(scenario, "full")]
        ])
        route_bytes = np.mean([
            float(row["active_paged_kv_bytes_final"])
            for row in throughput_request_groups[(scenario, "learned_and_or")]
        ])
        throughput_statistics[scenario] = {
            "paired_full_wall_over_route_wall": bootstrap_mean(
                ratios, args.bootstrap_draws, args.seed + 10 * offset
            ),
            "full_over_route_mean_active_paged_bytes": float(full_bytes / route_bytes),
        }

    payload = {
        "protocol": {
            "expected_request_rows": 736,
            "expected_batch_rows": 43,
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
        },
        "issues": issues,
        "integrity_passed": not issues,
        "runner_scientific_gate_passed": metrics.get("passed"),
        "accuracy_statistics": accuracy_statistics,
        "throughput_statistics": throughput_statistics,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
