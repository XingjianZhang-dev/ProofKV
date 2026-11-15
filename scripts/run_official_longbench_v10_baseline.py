#!/usr/bin/env python3
"""Model-compatible official baselines on the frozen LongBench-v10 slice."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import string
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from proofkv.utils.io import append_jsonl
from freeze_longbench_structured_selection import official_fields, row_id


OFFICIAL_COMMITS = {
    "official_compresskv": "12f920fbecba67e16806a6d8eac0ad389ddcad6f",
    "official_headkv_reasonkv": "0862a0955fe82e9ff611d59541918e02c5def625",
}
MODEL_LABELS = {
    "official_compresskv": "qwen25",
    "official_headkv_reasonkv": "mistral02",
}
MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_parts(
    tokenizer: Any,
    context: str,
    question: str,
    answer_prefix: str,
) -> tuple[str, str]:
    if tokenizer.chat_template is None:
        formatted_context = (
            str(getattr(tokenizer, "bos_token", "") or "") + context
        )
        question_suffix = "\n"
    else:
        separator = "#" * (len(context) + 10)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        formatted_context, question_suffix = rendered.split(separator, 1)
    return formatted_context, question + question_suffix + answer_prefix


def tokenize_parts(
    tokenizer: Any,
    formatted_context: str,
    question_tail: str,
) -> tuple[torch.Tensor, list[tuple[int, int]], int, int]:
    context_encoding = tokenizer(
        formatted_context,
        add_special_tokens=False,
        return_tensors="pt",
    )
    question_encoding = tokenizer(
        question_tail,
        add_special_tokens=False,
        return_tensors="pt",
    )
    context_ids = context_encoding["input_ids"]
    question_ids = question_encoding["input_ids"]
    return (
        torch.cat((context_ids, question_ids), dim=1),
        [],
        int(context_ids.shape[1]),
        int(question_ids.shape[1]),
    )


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_qa_f1(prediction: str, answers: list[str]) -> float:
    return max(
        (
            qa_f1(prediction.lstrip(), str(answer))
            for answer in answers
        ),
        default=0.0,
    )


def environment_versions() -> dict[str, Any]:
    packages = {}
    for package in (
        "torch",
        "transformers",
        "tokenizers",
        "flash-attn",
        "huggingface-hub",
        "nvtx",
        "protobuf",
        "sentencepiece",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=sorted(OFFICIAL_COMMITS),
        required=True,
    )
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capacities", default="64,128,256,512")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--max-samples-per-task", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def tree_python_hashes(root: Path) -> dict[str, str]:
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise ValueError(f"no Python sources below {root}")
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in files
    }


def official_auxiliary_hashes(
    method: str,
    official_repo: Path,
) -> dict[str, str]:
    if method != "official_headkv_reasonkv":
        return {}
    candidates = [
        official_repo / "csrc" / "build.py",
        official_repo / "csrc" / "csrc" / "cuda_api.cu",
        official_repo / "csrc" / "csrc" / "static_switch.h",
        official_repo / "csrc" / "include" / "cuda_api.h",
        official_repo / "csrc" / "build_a100" / "build_manifest.json",
        official_repo / "Important_Head" / "head.yml",
    ]
    binaries = sorted(
        (
            official_repo
            / "csrc"
            / "build_a100"
            / "lib"
        ).glob("tiny_api_cuda*.so")
    )
    candidates.extend(binaries)
    if not binaries or not all(path.is_file() for path in candidates):
        raise FileNotFoundError("HeadKV CUDA extension artifacts are incomplete")
    return {
        str(path.relative_to(official_repo)): sha256_file(path)
        for path in candidates
    }


def implementation_hashes() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        repo / "scripts" / "freeze_longbench_structured_selection.py",
    ]
    return {str(path): sha256_file(path) for path in paths}


def allocate_layers(
    layer_ratio: list[float],
    total_budget: int,
    minimum: int = 32,
    maximum: int = 1536,
) -> list[int]:
    importance = np.asarray(layer_ratio, dtype=float)
    if (
        importance.ndim != 1
        or not len(importance)
        or np.any(importance < 0)
        or not np.isfinite(importance).all()
    ):
        raise ValueError("invalid CompressKV layer-importance vector")
    if not np.isclose(importance.sum(), 1.0):
        importance = importance / importance.sum()
    if total_budget < minimum * len(importance):
        raise ValueError("total budget is below the per-layer minimum")
    budget = np.full_like(importance, minimum, dtype=float)
    budget += np.round(importance * (total_budget - budget.sum()))
    budget = np.clip(budget, minimum, maximum)
    difference = int(total_budget - budget.sum())
    while difference:
        if difference > 0:
            candidates = np.where(budget < maximum)[0]
            order = candidates[np.argsort(-importance[candidates])]
            direction = 1
        else:
            candidates = np.where(budget > minimum)[0]
            order = candidates[np.argsort(importance[candidates])]
            direction = -1
        if not len(order):
            raise ValueError(
                "could not allocate the exact CompressKV layer budget"
            )
        for index in order:
            budget[index] += direction
            difference -= direction
            if not difference:
                break
    budget = budget.astype(int)
    if int(budget.sum()) != total_budget:
        raise ValueError("CompressKV layer-budget sum is not exact")
    if int(budget.min()) < minimum or int(budget.max()) > maximum:
        raise ValueError("CompressKV layer budget is outside bounds")
    return budget.tolist()


def headkv_internal_capacity(
    target_native_tokens: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    window_size: int,
) -> int:
    if num_attention_heads % num_key_value_heads:
        raise ValueError("HeadKV requires an integral GQA expansion factor")
    factor = num_attention_heads // num_key_value_heads
    internal = target_native_tokens // factor - 1
    if internal <= window_size:
        raise ValueError("target capacity is too small for HeadKV's window")
    return internal


def configure_patch(method: str, official_repo: Path) -> None:
    sys.path.insert(0, str(official_repo))
    if method == "official_compresskv":
        from methods.compresskv.monkeypatch import replace_qwen2

        replace_qwen2()
    else:
        from headkv.monkeypatch import replace_mistral

        replace_mistral("ReasonKV")


def load_official_model(
    args: argparse.Namespace,
) -> tuple[Any, Any, dict[str, Any]]:
    official_repo = Path(args.official_repo)
    configure_patch(args.method, official_repo)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        revision=args.model_revision,
        use_fast=args.method == "official_compresskv",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if args.method == "official_compresskv"
        else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        use_cache=True,
    ).cuda().eval()
    config = model.config
    if args.method == "official_compresskv":
        config.window_size = 8
        config.kernel_size = 5
        config.pooling = "avgpool"
        config.first_k = 4
        layer_path = (
            official_repo
            / "scores"
            / "Qwen2.5-7B-Instruct_layer_score.jsonl"
        )
        head_path = (
            official_repo
            / "scores"
            / "Qwen2.5-7B-Instruct_head_idx.json"
        )
        layer_importance = json.loads(layer_path.read_text())["avg_score"]
        head_payload = json.loads(head_path.read_text())
        config.important_heads = [
            head_payload[str(index)] for index in range(len(head_payload))
        ]
        static = {
            "dtype": "bfloat16",
            "window_size": 8,
            "kernel_size": 5,
            "pooling": "avgpool",
            "first_k": 4,
            "layer_score_sha256": sha256_file(layer_path),
            "head_index_sha256": sha256_file(head_path),
            "layer_importance": layer_importance,
        }
    else:
        config.window_size = 8
        config.head_choice = "reason"
        config.beta = 1.005
        config.temp = 1.0
        config.kernel_size = 7
        config.skip = 0
        config.normalize = True
        config.pooling = "maxpool"
        config.floor = 0.2
        static = {
            "dtype": "float16",
            "window_size": 8,
            "head_choice": "reason",
            "beta": 1.005,
            "temp": 1.0,
            "kernel_size": 7,
            "skip": 0,
            "normalize": True,
            "pooling": "maxpool",
            "floor": 0.2,
        }
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, static


def reset_headkv_clusters(model: Any) -> None:
    for layer in model.model.layers:
        attention = layer.self_attn
        if hasattr(attention, "kv_cluster"):
            delattr(attention, "kv_cluster")


def reset_compresskv_clusters(model: Any) -> None:
    """Rebuild the official lazy cluster after every capacity change.

    The upstream attention patch caches ``kv_cluster`` on each layer. Merely
    changing ``model.config`` would therefore leave later capacities executing
    with the first capacity's selector state.
    """

    for layer in model.model.layers:
        attention = layer.self_attn
        if hasattr(attention, "kv_cluster"):
            delattr(attention, "kv_cluster")


def configure_capacity(
    model: Any,
    method: str,
    capacity: int,
    static: dict[str, Any],
) -> dict[str, Any]:
    config = model.config
    if method == "official_compresskv":
        layer_capacities = allocate_layers(
            list(static["layer_importance"]),
            capacity * int(config.num_hidden_layers),
        )
        config.max_capacity_prompt = capacity
        config.max_capacity_prompt_layer_adaptive = layer_capacities
        reset_compresskv_clusters(model)
        return {
            "layer_capacities": layer_capacities,
            "layer_capacity_mean": sum(layer_capacities)
            / len(layer_capacities),
        }
    internal = headkv_internal_capacity(
        capacity,
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(static["window_size"]),
    )
    config.base_capacity = internal
    reset_headkv_clusters(model)
    return {
        "target_native_equivalent_tokens": capacity,
        "internal_query_head_capacity": internal,
        "gqa_expansion_factor": (
            int(config.num_attention_heads)
            // int(config.num_key_value_heads)
        ),
    }


def kv_bytes_per_native_token(model: Any) -> int:
    config = model.config
    head_dim = int(
        getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
    )
    return (
        int(config.num_hidden_layers)
        * int(config.num_key_value_heads)
        * head_dim
        * 2
        * torch.tensor([], dtype=model.dtype).element_size()
    )


def compresskv_runtime_capacity_audit(
    model: Any,
    expected_layer_capacities: list[int],
) -> dict[str, Any]:
    """Fail if any live official cluster uses a stale capacity vector."""

    expected = [int(value) for value in expected_layer_capacities]
    observed_hashes: set[str] = set()
    clusters = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        cluster = getattr(attention, "kv_cluster", None)
        if cluster is None:
            raise ValueError("CompressKV runtime cluster was not initialized")
        observed = [
            int(value)
            for value in cluster.max_capacity_prompt_layer_adaptive
        ]
        if observed != expected:
            raise ValueError(
                "CompressKV runtime cluster retained a stale layer budget"
            )
        observed_hashes.add(
            hashlib.sha256(
                json.dumps(observed, separators=(",", ":")).encode()
            ).hexdigest()
        )
        clusters += 1
    if clusters != int(model.config.num_hidden_layers):
        raise ValueError("CompressKV runtime cluster count differs")
    if len(observed_hashes) != 1:
        raise ValueError("CompressKV runtime layers disagree on capacity")
    return {
        "runtime_capacity_cluster_count": clusters,
        "runtime_layer_capacities_sha256": observed_hashes.pop(),
    }


def compresskv_physical_budget(
    prompt_tokens: int,
    layer_capacities: list[int],
    model: Any,
) -> dict[str, float]:
    config = model.config
    retained_by_layer = [
        min(prompt_tokens, int(value)) for value in layer_capacities
    ]
    retained = sum(retained_by_layer) / len(retained_by_layer)
    bytes_per_token = kv_bytes_per_native_token(model)
    return {
        "actual_retained_tokens": retained,
        "actual_retained_ratio": retained / prompt_tokens,
        "actual_kv_bytes": retained * bytes_per_token,
        "layer_retained_token_min": min(retained_by_layer),
        "layer_retained_token_max": max(retained_by_layer),
    }


def headkv_physical_budget(
    prompt_tokens: int,
    model: Any,
) -> dict[str, float]:
    config = model.config
    entries = 0
    for layer in model.model.layers:
        cluster = layer.self_attn.kv_cluster
        entries += int(cluster.head_capacity[layer.self_attn.layer_idx].sum())
        entries += int(cluster.window_size * config.num_attention_heads)
    retained = entries / (
        int(config.num_hidden_layers) * int(config.num_key_value_heads)
    )
    return {
        "actual_retained_tokens": retained,
        "actual_retained_ratio": retained / prompt_tokens,
        "actual_kv_bytes": retained * kv_bytes_per_native_token(model),
        "query_head_kv_entries": entries,
    }


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    keys = [
        (row["task"], row["id"], row["method"], int(row["capacity"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("existing official-baseline rows contain duplicates")
    return rows


def main() -> None:
    args = parse_args()
    if args.model_label != MODEL_LABELS[args.method]:
        raise ValueError("official method and model label are incompatible")
    capacities = [
        int(value) for value in args.capacities.split(",") if value.strip()
    ]
    if capacities != sorted(set(capacities)):
        raise ValueError("capacities must be unique and sorted")
    selection_path = Path(args.selection_manifest)
    selection = json.loads(selection_path.read_text())
    if capacities != list(selection["capacities"]):
        raise ValueError("capacities differ from the frozen selection")
    tasks = list(selection["tasks"])
    if args.tasks:
        requested = [
            value.strip() for value in args.tasks.split(",") if value.strip()
        ]
        if not requested or not set(requested).issubset(tasks):
            raise ValueError("tasks must be a nonempty frozen-task subset")
        tasks = requested
    frozen_n = int(selection["max_samples_per_task"])
    execution_n = (
        frozen_n
        if args.max_samples_per_task is None
        else int(args.max_samples_per_task)
    )
    if not 1 <= execution_n <= frozen_n:
        raise ValueError("invalid sample count")

    official_repo = Path(args.official_repo)
    commit = git_commit(official_repo)
    if commit != OFFICIAL_COMMITS[args.method]:
        raise ValueError(f"unexpected official commit: {commit}")
    source_root = (
        official_repo / "methods" / "compresskv"
        if args.method == "official_compresskv"
        else official_repo / "headkv"
    )
    official_hashes = tree_python_hashes(source_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    metrics_path = (
        out_dir / "metrics.json"
        if execution_n == frozen_n and tasks == selection["tasks"]
        else out_dir / f"pilot_metrics_n{execution_n}.json"
    )
    manifest_path = out_dir / "run_manifest.json"
    manifest = {
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"out_dir", "max_samples_per_task"}
        },
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_protocol": selection["protocol"],
        "official_commit": commit,
        "official_source_sha256": official_hashes,
        "official_auxiliary_sha256": official_auxiliary_hashes(
            args.method,
            official_repo,
        ),
        "implementation_sha256": implementation_hashes(),
        "environment": environment_versions(),
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("run manifest differs from resumed execution")
    else:
        if results_path.exists():
            raise ValueError("cannot resume results without a run manifest")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if metrics_path.exists():
        raise FileExistsError(metrics_path)

    model, tokenizer, static = load_official_model(args)
    rows = load_existing(results_path)
    completed = {
        (row["task"], row["id"], row["method"], int(row["capacity"]))
        for row in rows
    }
    capacity_configurations: dict[str, Any] = {}
    for capacity in capacities:
        capacity_configurations[str(capacity)] = configure_capacity(
            model,
            args.method,
            capacity,
            static,
        )
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
            for selected in tqdm(
                task_selection["selected"][:execution_n],
                desc=f"{args.method}:{task}:{capacity}",
            ):
                raw = source_rows[int(selected["index"])]
                source_id = row_id(task, int(selected["index"]), raw)
                if source_id != selected["id"]:
                    raise ValueError("frozen row identity mismatch")
                key = (task, source_id, args.method, capacity)
                if key in completed:
                    continue
                context, question, answer_prefix = official_fields(task, raw)
                formatted_context, question_tail = format_parts(
                    tokenizer,
                    context,
                    question,
                    answer_prefix,
                )
                input_ids, _, _, question_tokens = tokenize_parts(
                    tokenizer,
                    formatted_context,
                    question_tail,
                )
                prompt_tokens = int(input_ids.shape[1])
                expected_prompt = int(
                    selected["token_counts"][args.model_label]["prompt"]
                )
                if prompt_tokens != expected_prompt:
                    raise ValueError(
                        f"prompt drift for {source_id}: "
                        f"{prompt_tokens} != {expected_prompt}"
                    )
                input_ids = input_ids.to(model.device)
                attention_mask = torch.ones_like(input_ids)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS[task],
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                    )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                prediction = tokenizer.decode(
                    output[0, prompt_tokens:],
                    skip_special_tokens=True,
                )
                answers = [str(value) for value in raw["answers"]]
                if args.method == "official_compresskv":
                    runtime_capacity = compresskv_runtime_capacity_audit(
                        model,
                        capacity_configurations[str(capacity)][
                            "layer_capacities"
                        ],
                    )
                    physical = compresskv_physical_budget(
                        prompt_tokens,
                        capacity_configurations[str(capacity)][
                            "layer_capacities"
                        ],
                        model,
                    )
                else:
                    runtime_capacity = {}
                    physical = headkv_physical_budget(prompt_tokens, model)
                if float(physical["actual_retained_tokens"]) > capacity + 1e-6:
                    raise ValueError(
                        f"{args.method} exceeded capacity {capacity}: "
                        f"{physical['actual_retained_tokens']}"
                    )
                row = {
                    "id": source_id,
                    "task": task,
                    "method": args.method,
                    "capacity": capacity,
                    "answers": answers,
                    "prediction": prediction,
                    "score": best_qa_f1(prediction, answers),
                    "context_tokens": prompt_tokens - question_tokens,
                    "question_tokens": question_tokens,
                    "prompt_tokens": prompt_tokens,
                    "retained_context_tokens_target": None,
                    **runtime_capacity,
                    **physical,
                    "seconds": seconds,
                    "selection_seconds": 0.0,
                    "generation_seconds": seconds,
                    "peak_cuda_memory_gb": (
                        torch.cuda.max_memory_allocated() / 1024**3
                    ),
                }
                append_jsonl(results_path, row)
                rows.append(row)
                completed.add(key)

    expected = len(tasks) * execution_n * len(capacities)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), int(row["capacity"]))].append(row)
    records = [
        {
            "task": task,
            "capacity": capacity,
            "n": len(values),
            "score": sum(float(row["score"]) for row in values) / len(values),
            "actual_retained_tokens_mean": sum(
                float(row["actual_retained_tokens"]) for row in values
            )
            / len(values),
            "actual_kv_bytes_mean": sum(
                float(row["actual_kv_bytes"]) for row in values
            )
            / len(values),
            "seconds_mean": sum(float(row["seconds"]) for row in values)
            / len(values),
        }
        for (task, capacity), values in sorted(grouped.items())
    ]
    metrics = {
        "status": "PASS",
        "protocol": "frozen_longbench_v10_model_compatible_official_baseline",
        "method": args.method,
        "model": args.model_label,
        "model_revision": args.model_revision,
        "tasks": tasks,
        "capacities": capacities,
        "n_per_task": execution_n,
        "rows": len(rows),
        "selection_manifest_sha256": sha256_file(selection_path),
        "official_commit": commit,
        "official_static_configuration": {
            key: value
            for key, value in static.items()
            if key != "layer_importance"
        },
        "capacity_configurations": capacity_configurations,
        "task_capacity_records": records,
        "max_capacity_excess_tokens": max(
            float(row["actual_retained_tokens"]) - int(row["capacity"])
            for row in rows
        ),
        "results_sha256": sha256_file(results_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "method": args.method,
                "rows": len(rows),
                "out": str(metrics_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
