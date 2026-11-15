#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from proofkv.eval.generation import load_model_tokenizer
from proofkv.utils.io import append_jsonl
from freeze_longbench_structured_selection import official_fields, row_id
from run_hotpot_compact_qa_upgraded_v10 import (
    production_compact_decode,
)
from run_longbench_structured_upgraded_v10 import (
    MAX_NEW_TOKENS,
    aggregate,
    best_qa_f1,
    format_parts,
    load_existing_rows,
    sha256_file,
    tokenize_parts,
)
from run_physical_kv_replay import _release_cuda


METHOD = "llmlingua2_official"
COMPRESSOR_REVISION = "ebaba9b0e874dadd3003ffcff828e4397e568089"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--compressor-model", required=True)
    parser.add_argument(
        "--compressor-revision",
        default=COMPRESSOR_REVISION,
    )
    parser.add_argument("--capacities", default="128,256,512")
    parser.add_argument("--sample-start-per-task", type=int, default=1)
    parser.add_argument("--max-samples-per-task", type=int, default=46)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def directory_file_sha256(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    }


def fit_compressed_context(
    tokenizer: Any,
    compressed_context: str,
    question: str,
    answer_prefix: str,
    capacity: int,
) -> tuple[str, torch.Tensor, int, int, int]:
    """Retokenize compressor text and deterministically enforce total KV cap."""
    raw_ids = list(
        tokenizer(
            compressed_context,
            add_special_tokens=False,
        )["input_ids"]
    )
    candidate = compressed_context
    while True:
        formatted_context, question_tail = format_parts(
            tokenizer,
            candidate,
            question,
            answer_prefix,
        )
        input_ids, offsets, context_tokens, question_tokens = tokenize_parts(
            tokenizer,
            formatted_context,
            question_tail,
        )
        if len(offsets) <= capacity:
            return (
                candidate,
                input_ids,
                context_tokens,
                question_tokens,
                len(offsets),
            )
        overflow = len(offsets) - capacity
        if len(raw_ids) <= overflow:
            raise ValueError(
                f"capacity {capacity} cannot preserve the uncompressed "
                "question and chat wrapper"
            )
        raw_ids = raw_ids[: -max(1, overflow)]
        candidate = tokenizer.decode(
            raw_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def compress_to_decoder_budget(
    compressor: Any,
    tokenizer: Any,
    context: str,
    question: str,
    answer_prefix: str,
    capacity: int,
    initial_target: int,
    max_attempts: int = 8,
) -> tuple[dict[str, Any], str, torch.Tensor, int, int, int, list[dict[str, int]]]:
    """Normalize an official compressor output to the decoder's KV budget."""
    target = max(1, int(initial_target))
    best: tuple[
        dict[str, Any],
        str,
        torch.Tensor,
        int,
        int,
        int,
    ] | None = None
    attempts: list[dict[str, int]] = []
    for _ in range(max_attempts):
        compressed = compressor.compress_prompt(
            [context],
            target_token=target,
            use_context_level_filter=False,
            use_token_level_filter=True,
        )
        (
            fitted_context,
            input_ids,
            retained_context_tokens,
            question_tokens,
            actual_tokens,
        ) = fit_compressed_context(
            tokenizer,
            str(compressed["compressed_prompt"]),
            question,
            answer_prefix,
            capacity,
        )
        attempts.append(
            {
                "compressor_target_tokens": target,
                "compressor_reported_tokens": int(
                    compressed["compressed_tokens"]
                ),
                "decoder_actual_tokens": actual_tokens,
            }
        )
        candidate = (
            compressed,
            fitted_context,
            input_ids,
            retained_context_tokens,
            question_tokens,
            actual_tokens,
        )
        if best is None or actual_tokens > best[-1]:
            best = candidate
        if actual_tokens == capacity:
            return (*candidate, attempts)
        target *= 2
    assert best is not None
    return (*best, attempts)


def main() -> None:
    args = parse_args()
    if importlib.metadata.version("llmlingua") != "0.2.2":
        raise ValueError("formal runner requires llmlingua==0.2.2")
    if args.compressor_revision != COMPRESSOR_REVISION:
        raise ValueError("unexpected LLMLingua-2 model revision")
    capacities = [
        int(value)
        for value in args.capacities.split(",")
        if value.strip()
    ]
    if capacities != [128, 256, 512]:
        raise ValueError("formal capacities must be 128,256,512")

    selection_path = Path(args.selection_manifest)
    selection = json.loads(selection_path.read_text())
    tasks = list(selection["tasks"])
    start = int(args.sample_start_per_task)
    count = int(args.max_samples_per_task)
    frozen_count = int(selection["max_samples_per_task"])
    if start != 1 or count != 46 or start + count != frozen_count:
        raise ValueError("formal sample slice must be indices 1--46")

    compressor_path = Path(args.compressor_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    metrics_path = out_dir / "metrics.json"
    manifest_path = out_dir / "run_manifest.json"
    if metrics_path.exists():
        raise FileExistsError(metrics_path)
    manifest = {
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"out_dir"}
        },
        "selection_manifest_sha256": sha256_file(selection_path),
        "implementation_sha256": {
            str(Path(__file__).resolve()): sha256_file(
                Path(__file__).resolve()
            ),
            "physical_decode": sha256_file(
                Path(__file__).with_name(
                    "run_hotpot_compact_qa_upgraded_v10.py"
                )
            ),
            "longbench_contract": sha256_file(
                Path(__file__).with_name(
                    "run_longbench_structured_upgraded_v10.py"
                )
            ),
        },
        "llmlingua_package_version": importlib.metadata.version(
            "llmlingua"
        ),
        "compressor_model_sha256": directory_file_sha256(compressor_path),
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("run manifest differs")
    else:
        if results_path.exists():
            raise ValueError("cannot resume results without manifest")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    model, tokenizer = load_model_tokenizer(
        args.model,
        dtype="bfloat16",
        attn_implementation=args.attn_implementation,
    )
    from llmlingua import PromptCompressor

    compressor = PromptCompressor(
        model_name=str(compressor_path),
        use_llmlingua2=True,
        device_map="cuda",
    )
    head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    kv_bytes_per_token = (
        model.config.num_hidden_layers
        * model.config.num_key_value_heads
        * head_dim
        * 2
        * 2
    )
    rows = load_existing_rows(results_path)
    completed = {
        (str(row["id"]), int(row["capacity"]))
        for row in rows
    }

    for task in tasks:
        task_selection = selection["selections"][task]
        data_path = Path(task_selection["data_file"])
        if sha256_file(data_path) != task_selection["data_sha256"]:
            raise ValueError(f"data hash mismatch for {task}")
        source_rows = [
            json.loads(line)
            for line in data_path.read_text().splitlines()
            if line.strip()
        ]
        selected_rows = task_selection["selected"][start : start + count]
        for selected in tqdm(
            selected_rows,
            desc=f"{args.model_label}:{task}:{METHOD}",
        ):
            raw = source_rows[int(selected["index"])]
            source_id = row_id(task, int(selected["index"]), raw)
            if source_id != selected["id"]:
                raise ValueError(f"row identity mismatch: {source_id}")
            context, question, answer_prefix = official_fields(task, raw)
            original_formatted, question_tail = format_parts(
                tokenizer,
                context,
                question,
                answer_prefix,
            )
            (
                _,
                original_offsets,
                _,
                original_question_tokens,
            ) = tokenize_parts(
                tokenizer,
                original_formatted,
                question_tail,
            )
            expected_prompt_tokens = int(
                selected["token_counts"][args.model_label]["prompt"]
            )
            if len(original_offsets) != expected_prompt_tokens:
                raise ValueError(
                    f"prompt length drift for {source_id}"
                )
            answers = [str(value) for value in raw["answers"]]
            for capacity in capacities:
                if (source_id, capacity) in completed:
                    continue
                empty_formatted, _ = format_parts(
                    tokenizer,
                    "",
                    question,
                    answer_prefix,
                )
                empty_context_tokens = len(
                    tokenizer(
                        empty_formatted,
                        add_special_tokens=False,
                    )["input_ids"]
                )
                compressor_target = (
                    capacity
                    - original_question_tokens
                    - empty_context_tokens
                )
                if compressor_target < 1:
                    raise ValueError(
                        f"no context capacity for {source_id}/{capacity}"
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                selection_started = time.perf_counter()
                (
                    compressed,
                    fitted_context,
                    input_ids,
                    retained_context_tokens,
                    question_tokens,
                    actual_tokens,
                    compression_attempts,
                ) = compress_to_decoder_budget(
                    compressor,
                    tokenizer,
                    context,
                    question,
                    answer_prefix,
                    capacity,
                    compressor_target,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                selection_seconds = (
                    time.perf_counter() - selection_started
                )

                input_ids = input_ids.to(
                    next(model.parameters()).device
                )
                retained_indices = list(range(actual_tokens))
                torch.cuda.reset_peak_memory_stats()
                generation_started = time.perf_counter()
                (
                    prediction,
                    cache,
                    _,
                    warnings,
                    generation_timing,
                ) = production_compact_decode(
                    model,
                    tokenizer,
                    input_ids,
                    retained_indices,
                    actual_tokens,
                    MAX_NEW_TOKENS[task],
                    512,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                generation_seconds = (
                    time.perf_counter() - generation_started
                )
                peak_cuda_memory_gb = (
                    torch.cuda.max_memory_allocated() / 1024**3
                )
                del cache
                _release_cuda()
                if actual_tokens > capacity:
                    raise ValueError("LLMLingua-2 exceeded KV cap")
                row = {
                    "id": source_id,
                    "task": task,
                    "method": METHOD,
                    "capacity": capacity,
                    "answers": answers,
                    "prediction": prediction,
                    "score": best_qa_f1(prediction, answers),
                    "context_tokens": (
                        len(original_offsets)
                        - original_question_tokens
                    ),
                    "question_tokens": question_tokens,
                    "prompt_tokens": len(original_offsets),
                    "retained_context_tokens_target": (
                        capacity - question_tokens
                    ),
                    "actual_retained_tokens": actual_tokens,
                    "actual_retained_ratio": (
                        actual_tokens / len(original_offsets)
                    ),
                    "actual_kv_bytes": (
                        actual_tokens * kv_bytes_per_token
                    ),
                    "seconds": (
                        selection_seconds + generation_seconds
                    ),
                    "selection_seconds": selection_seconds,
                    "generation_seconds": generation_seconds,
                    "peak_cuda_memory_gb": peak_cuda_memory_gb,
                    "selection_partition": {
                        "compressor": METHOD,
                        "compressor_revision": (
                            args.compressor_revision
                        ),
                        "compressor_target_tokens": compressor_target,
                        "compressor_origin_tokens": int(
                            compressed["origin_tokens"]
                        ),
                        "compressor_reported_tokens": int(
                            compressed["compressed_tokens"]
                        ),
                        "budget_normalization_attempts": (
                            compression_attempts
                        ),
                        "decoder_retained_context_tokens": (
                            retained_context_tokens
                        ),
                        "decoder_fitted_context_sha256": (
                            hashlib.sha256(
                                fitted_context.encode("utf-8")
                            ).hexdigest()
                        ),
                        "generation_timing": generation_timing,
                    },
                    "warnings": warnings,
                }
                if not math.isclose(
                    row["seconds"],
                    row["selection_seconds"]
                    + row["generation_seconds"],
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise ValueError("timing fields do not sum")
                rows.append(row)
                append_jsonl(results_path, row)
                completed.add((source_id, capacity))

        task_rows = [
            row for row in rows if row["task"] == task
        ]
        if len(task_rows) != count * len(capacities):
            raise RuntimeError(
                f"incomplete {task}: {len(task_rows)}"
            )

    task_records, macro_records = aggregate(rows, tasks)
    payload = {
        "protocol": "llmlingua2_same_compact_prefill_contract_v1",
        "selection_manifest_sha256": sha256_file(selection_path),
        "model": args.model_label,
        "model_path": args.model,
        "model_revision": args.model_revision,
        "compressor": METHOD,
        "compressor_revision": args.compressor_revision,
        "tasks": tasks,
        "n_per_task": count,
        "selection_start_per_task": start,
        "capacities": capacities,
        "capacity_scope": (
            "total prefill KV including uncompressed question tail"
        ),
        "kv_bytes_per_token_bf16": kv_bytes_per_token,
        "task_records": task_records,
        "macro_records": macro_records,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "macro": macro_records}))


if __name__ == "__main__":
    main()
