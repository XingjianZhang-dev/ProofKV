#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import re
import string
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

try:
    from scripts import eval_deployable_ruler_lite as local_ruler
    from scripts import eval_official_ruler_mini as official_ruler
except ImportError:  # pragma: no cover - direct script execution from scripts/
    import eval_deployable_ruler_lite as local_ruler
    import eval_official_ruler_mini as official_ruler

from proofkv.config import CompassConfig, Route
from proofkv.controllers import CompassKVController
from proofkv.eval.generation import load_model_tokenizer
from proofkv.eval.metrics import contains_answer, normalize_answer
from proofkv.patch import patch_model, unpatch_model
from proofkv.physical_kv import (
    compact_cache_stats,
    gather_legacy_past_key_values,
    legacy_cache_layers,
    make_decode_cache_position,
    make_cache_position,
    make_compact_attention_mask,
    make_position_ids,
    prepare_compact_prefill_inputs,
    compact_prefill_cache_stats,
)
from proofkv.routing import make_uniform_route_table
from proofkv.utils.io import append_jsonl, ensure_dir
from proofkv.utils.seed import set_seed


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SUPPORTED_SOURCES = {"local_ruler_lite", "official_ruler"}
SUPPORTED_TASKS = {"exact_retrieval", "multi_hop", "niah", "variable_tracking"}
SINK = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline physical KV replay prototype for CompassKV.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--context-length", type=int, default=8000)
    p.add_argument("--budget-ratio", type=float, default=0.08)
    p.add_argument("--n-per-task", type=int, default=2)
    p.add_argument("--tasks", default="exact_retrieval,multi_hop")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--source", choices=sorted(SUPPORTED_SOURCES), default="local_ruler_lite")
    p.add_argument("--include-compact-prefill", action="store_true")
    return p.parse_args()


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cache_seq_len(past_key_values: Any) -> int:
    layers = legacy_cache_layers(past_key_values)
    return int(layers[0][0].shape[2])


def _decode_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(list(token_ids), skip_special_tokens=True).strip()


def _first_number(text: str) -> str | None:
    match = re.search(r"\d+", text)
    return match.group(0) if match else None


def _canonical_text(text: str) -> str:
    collapsed = " ".join(text.lower().strip().split())
    return collapsed.rstrip(string.punctuation + " ")


def _first_answer_like_token(text: str) -> str | None:
    num = _first_number(text)
    if num is not None:
        return num
    match = re.search(r"\b[A-Za-z][A-Za-z0-9_-]*\b", text)
    return match.group(0).lower() if match else None


def _first_answer_like_exact(a: str, b: str) -> float:
    aa = _first_answer_like_token(a)
    bb = _first_answer_like_token(b)
    if aa is None or bb is None:
        return 0.0
    return float(aa == bb)


def _answer_set(text: str) -> set[str] | None:
    numbers = re.findall(r"\d+", text)
    if numbers:
        return set(numbers)
    uppercase_tokens = re.findall(r"\b[A-Z][A-Z0-9_-]{1,}\b", text)
    if uppercase_tokens:
        return {tok.lower() for tok in uppercase_tokens}
    return None


def _answer_set_exact(a: str, b: str) -> float | None:
    aa = _answer_set(a)
    bb = _answer_set(b)
    if aa is None or bb is None:
        return None
    return float(aa == bb)


def _digit_prefix_match(a: str, b: str) -> float:
    da = _first_number(a)
    db = _first_number(b)
    if not da or not db:
        return 0.0
    return float(da == db or da.startswith(db) or db.startswith(da))


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + int(ca != cb)))
        prev = cur
    return prev[-1]


def _normalized_edit(a: str, b: str) -> float:
    return _edit_distance(a, b) / max(1, max(len(a), len(b)))


def _forward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    past_key_values: Any | None,
    cache_position: torch.Tensor | None,
    warnings: List[str],
) -> Any:
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        # Prefill and one-token decode only consume the final-position logits.
        # Avoid materializing [batch, sequence, vocabulary] logits, which can
        # dominate peak memory at long context without changing the KV cache.
        "logits_to_keep": 1,
    }
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    if cache_position is not None:
        kwargs["cache_position"] = cache_position
    try:
        return model(**kwargs)
    except TypeError:
        if "cache_position" not in kwargs:
            raise
        kwargs.pop("cache_position")
        warnings.append("model.forward rejected cache_position; retried without it")
        return model(**kwargs)


def _step_with_cache(
    model: Any,
    input_ids: torch.Tensor,
    past_key_values: Any,
    original_position: int,
    warnings: List[str],
    cache_position_start: int | None = None,
) -> Any:
    device = input_ids.device
    cache_len = _cache_seq_len(past_key_values)
    attention_mask = make_compact_attention_mask(input_ids.shape[0], cache_len, input_ids.shape[1], device)
    position_ids = make_position_ids(original_position, input_ids.shape[1], device)
    if cache_position_start is None:
        cache_position = make_cache_position(original_position, input_ids.shape[1], device)
    else:
        cache_position = make_decode_cache_position(cache_position_start, input_ids.shape[1], device)
    return _forward(model, input_ids, attention_mask, position_ids, past_key_values, cache_position, warnings)


def _full_prefill(model: Any, input_ids: torch.Tensor, warnings: List[str]) -> Any:
    device = input_ids.device
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    position_ids = make_position_ids(0, input_ids.shape[1], device)
    cache_position = make_cache_position(0, input_ids.shape[1], device)
    return _forward(model, input_ids, attention_mask, position_ids, None, cache_position, warnings)


def _greedy_full_decode(model: Any, tokenizer: Any, input_ids: torch.Tensor, max_new_tokens: int) -> tuple[str, Any, List[str]]:
    warnings: List[str] = []
    with torch.inference_mode():
        out = _full_prefill(model, input_ids, warnings)
        past = out.past_key_values
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated: List[int] = []
        for step in range(max_new_tokens):
            token = int(next_id[0, 0].item())
            generated.append(token)
            if tokenizer.eos_token_id is not None and token == int(tokenizer.eos_token_id):
                break
            if step + 1 >= max_new_tokens:
                break
            out = _step_with_cache(model, next_id, past, input_ids.shape[1] + step, warnings)
            past = out.past_key_values
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    return _decode_text(tokenizer, generated), past, warnings


def _greedy_bootstrap_decode(
    model: Any,
    tokenizer: Any,
    past_key_values: Any,
    bootstrap_ids: torch.Tensor,
    bootstrap_position: int,
    max_new_tokens: int,
    cache_position_start: int | None = None,
) -> tuple[str, Any, List[str]]:
    warnings: List[str] = []
    next_input = bootstrap_ids
    past = past_key_values
    generated: List[int] = []
    with torch.inference_mode():
        for step in range(max_new_tokens):
            compact_pos = None if cache_position_start is None else cache_position_start + step
            out = _step_with_cache(model, next_input, past, bootstrap_position + step, warnings, compact_pos)
            past = out.past_key_values
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            token = int(next_id[0, 0].item())
            generated.append(token)
            if tokenizer.eos_token_id is not None and token == int(tokenizer.eos_token_id):
                break
            next_input = next_id
    return _decode_text(tokenizer, generated), past, warnings


def _greedy_decode_from_prefill_logits(
    model: Any,
    tokenizer: Any,
    past_key_values: Any,
    logits: torch.Tensor,
    first_original_position: int,
    first_cache_position: int,
    max_new_tokens: int,
) -> tuple[str, Any, List[str]]:
    warnings: List[str] = []
    past = past_key_values
    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated: List[int] = []
    with torch.inference_mode():
        for step in range(max_new_tokens):
            token = int(next_id[0, 0].item())
            generated.append(token)
            if tokenizer.eos_token_id is not None and token == int(tokenizer.eos_token_id):
                break
            if step + 1 >= max_new_tokens:
                break
            out = _step_with_cache(
                model,
                next_id,
                past,
                first_original_position + step,
                warnings,
                first_cache_position + step,
            )
            past = out.past_key_values
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    return _decode_text(tokenizer, generated), past, warnings


def _compact_prefill(
    model: Any,
    input_ids: torch.Tensor,
    retained_indices: Sequence[int],
    warnings: List[str],
) -> tuple[Any, Any]:
    prepared = prepare_compact_prefill_inputs(input_ids, retained_indices)
    if prepared.input_ids.shape[1] <= 0:
        raise ValueError("compact prefill needs at least one retained prompt token")
    out = _forward(
        model,
        prepared.input_ids,
        prepared.attention_mask,
        prepared.position_ids,
        None,
        prepared.cache_position,
        warnings,
    )
    return out, prepared


def _compact_prefill_decode(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    retained_indices: Sequence[int],
    prompt_len: int,
    max_new_tokens: int,
) -> tuple[str, Any, Dict[str, Any], List[str], Dict[str, float]]:
    warnings: List[str] = []
    prefill_start = time.perf_counter()
    prefill_out, prepared = _compact_prefill(model, input_ids, retained_indices, warnings)
    if int(prepared.retained_indices[-1].item()) != int(prompt_len - 1):
        raise ValueError("compact prefill generation requires the final prompt token to be retained")
    prefill_seconds = time.perf_counter() - prefill_start
    stats = compact_prefill_cache_stats(prompt_len, prefill_out.past_key_values)
    decode_start = time.perf_counter()
    text, final_cache, decode_warnings = _greedy_decode_from_prefill_logits(
        model,
        tokenizer,
        prefill_out.past_key_values,
        prefill_out.logits,
        prompt_len,
        int(prepared.input_ids.shape[1]),
        max_new_tokens,
    )
    decode_seconds = time.perf_counter() - decode_start
    warnings.extend(decode_warnings)
    warnings.append("compact prefill cache API used: model-native compact prefill")
    return text, final_cache, stats, warnings, {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": prefill_seconds + decode_seconds,
    }


def _clone_legacy_cache(past_key_values: Any) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple((key.clone(), value.clone()) for key, value in legacy_cache_layers(past_key_values))


def _dynamic_cache_constructor(legacy_cache: Any) -> Any | None:
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache(ddp_cache_data=_clone_legacy_cache(legacy_cache))
    except Exception:
        return None


def _physical_decode(
    model: Any,
    tokenizer: Any,
    full_cache: Any,
    retained_indices: Sequence[int],
    prompt_len: int,
    bootstrap_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str, Any, Dict[str, Any], List[str], str]:
    retained_without_bootstrap = [idx for idx in retained_indices if idx != prompt_len - 1]
    compact_legacy = gather_legacy_past_key_values(full_cache, retained_without_bootstrap, return_dynamic_cache=False)
    stats = compact_cache_stats(full_cache, compact_legacy)
    attempts: List[Tuple[str, Callable[[], Any]]] = []
    dynamic = _dynamic_cache_constructor(compact_legacy)
    if dynamic is not None:
        attempts.append(("DynamicCache(ddp_cache_data)", lambda: _dynamic_cache_constructor(compact_legacy)))
    attempts.append(("legacy_tuple", lambda: _clone_legacy_cache(compact_legacy)))

    errors: List[str] = []
    for name, maker in attempts:
        cache = maker()
        if cache is None:
            continue
        try:
            text, final_cache, warnings = _greedy_bootstrap_decode(
                model,
                tokenizer,
                cache,
                bootstrap_ids,
                prompt_len - 1,
                max_new_tokens,
                cache_position_start=int(stats["compact_seq_len"]),
            )
            warnings.append(f"physical replay cache API used: {name}")
            return text, final_cache, stats, warnings, name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Physical compact KV replay requires a compatible cache API: " + " | ".join(errors))


def _make_controller(model: Any, context_length: int, budget_ratio: float, recent: int, guidepost_budget: int) -> CompassKVController:
    cfg = CompassConfig(
        sink_size=SINK,
        recent_size=recent,
        guidepost_count=guidepost_budget,
        block_size=16,
        candidates_per_block=1,
        selector="uniform",
        default_route="navigation",
        target_budget_ratio=budget_ratio,
        dtype="bfloat16",
        seed=0,
    )
    table = make_uniform_route_table(
        int(model.config.num_hidden_layers),
        int(model.config.num_attention_heads),
        Route.NAV,
        {"method": "compass_prefill_guidepost", "context_length": context_length, "physical_replay": True},
    )
    return CompassKVController(cfg, table)


def _token_offsets(tokenizer: Any, prompt: str) -> List[Tuple[int, int]]:
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    return [(int(s), int(e)) for s, e in enc["offset_mapping"]]


def _prompt_tokens(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def _build_samples(args: argparse.Namespace, tokenizer: Any, out_dir: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unsupported = sorted(set(tasks) - SUPPORTED_TASKS)
    if unsupported:
        raise ValueError(f"Unsupported tasks for physical replay: {unsupported}")
    if args.source == "local_ruler_lite":
        allowed = {"exact_retrieval", "multi_hop"}
        if not set(tasks).issubset(allowed):
            raise ValueError(f"local_ruler_lite tasks must be in {sorted(allowed)}")
        samples, filler_tokens, max_prompt_tokens = local_ruler.make_context_fit_suite(
            tokenizer,
            n_per_task=args.n_per_task,
            context_length=args.context_length,
            seed=args.seed,
            tasks=set(tasks),
        )
        rows = [
            {
                "sample_id": f"{sample.task_type}_{idx}",
                "task_type": sample.task_type,
                "prompt": sample.prompt,
                "answers": [sample.answer],
                "metadata": sample.metadata,
            }
            for idx, sample in enumerate(samples)
        ]
        return rows, {"filler_tokens": filler_tokens, "max_prompt_tokens": max_prompt_tokens}

    allowed = {"niah", "variable_tracking"}
    if not set(tasks).issubset(allowed):
        raise ValueError(f"official_ruler tasks must be in {sorted(allowed)}")
    official_args = SimpleNamespace(
        model=args.model,
        context_length=args.context_length,
        n_per_task=args.n_per_task,
        seed=args.seed,
    )
    raw_rows = official_ruler.make_samples(official_args, out_dir, tasks)
    rows = [
        {
            "sample_id": row["sample_id"],
            "task_type": row["task_type"],
            "prompt": row["prompt"],
            "answers": row["outputs"],
            "metadata": {"source_file": row.get("source_file"), "source_row": row.get("source_row")},
        }
        for row in raw_rows
    ]
    max_prompt_tokens = max(_prompt_tokens(tokenizer, row["prompt"]) for row in rows) if rows else 0
    return rows, {"filler_tokens": None, "max_prompt_tokens": max_prompt_tokens}


def _plan_for_sample(
    args: argparse.Namespace,
    sample: Dict[str, Any],
    tokenizer: Any,
    recent: int,
    guidepost_budget: int,
) -> Any:
    if args.source == "local_ruler_lite":
        return local_ruler.plan_for(
            "compass_prefill_guidepost",
            sample["sample_id"],
            SimpleNamespace(prompt=sample["prompt"]),
            tokenizer,
            recent,
            guidepost_budget,
        )
    return official_ruler.plan_for(
        "compass_prefill_guidepost",
        sample["sample_id"],
        sample["prompt"],
        tokenizer,
        recent,
        guidepost_budget,
    )


def _score(args: argparse.Namespace, prediction: str, sample: Dict[str, Any]) -> float | None:
    if args.source == "local_ruler_lite":
        return contains_answer(prediction, sample["answers"][0])
    if args.source == "official_ruler":
        return official_ruler.score_prediction(prediction, sample["answers"])
    return None


def _aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_type"]].append(row)
    records = []
    for task, vals in sorted(grouped.items()):
        def avg(key: str) -> float:
            nums = [float(v[key]) for v in vals if v.get(key) is not None]
            return sum(nums) / max(1, len(nums))

        record = {
            "task": task,
            "n": len(vals),
            "reference_correctness": avg("reference_correctness"),
            "physical_correctness": avg("physical_correctness"),
            "physical_vs_reference_exact": avg("physical_vs_reference_exact"),
            "physical_vs_reference_canonical_text_exact": avg("physical_vs_reference_canonical_text_exact"),
            "physical_vs_reference_numeric_answer_exact": avg("physical_vs_reference_numeric_answer_exact"),
            "physical_vs_reference_answer_set_exact": avg("physical_vs_reference_answer_set_exact"),
            "physical_vs_reference_semantic_match": avg("physical_vs_reference_semantic_match"),
            "physical_vs_reference_format_only_mismatch": avg("physical_vs_reference_format_only_mismatch"),
            "physical_vs_reference_digit_prefix": avg("physical_vs_reference_digit_prefix"),
            "physical_vs_reference_normalized_edit": avg("physical_vs_reference_normalized_edit"),
            "retained_ratio": avg("retained_ratio"),
            "compact_byte_ratio": avg("compact_byte_ratio"),
            "peak_cuda_memory_gb": max(
                [float(v["peak_cuda_memory_gb"]) for v in vals if v.get("peak_cuda_memory_gb") is not None],
                default=None,
            ),
        }
        if any(v.get("compact_prefill_correctness") is not None for v in vals):
            record.update(
                {
                    "compact_prefill_correctness": avg("compact_prefill_correctness"),
                    "compact_prefill_vs_reference_exact": avg("compact_prefill_vs_reference_exact"),
                    "compact_prefill_vs_reference_semantic_match": avg("compact_prefill_vs_reference_semantic_match"),
                    "compact_prefill_byte_ratio": avg("compact_prefill_byte_ratio"),
                    "compact_prefill_prefill_seconds": avg("compact_prefill_prefill_seconds"),
                    "compact_prefill_decode_seconds": avg("compact_prefill_decode_seconds"),
                    "compact_prefill_total_seconds": avg("compact_prefill_total_seconds"),
                }
            )
        records.append(record)
    return records


def _write_summary(out_dir: Path, payload: Dict[str, Any]) -> None:
    has_compact_prefill = bool(payload.get("include_compact_prefill"))
    cache_note = "offline compact K/V replay after prompt encoding"
    if has_compact_prefill:
        cache_note += " plus compact prompt replay"
    lines = [
        "# Physical KV Replay",
        "",
        f"- Model: `{payload['model']}`; source={payload['source']}; context={payload['context_length']}; budget={payload['budget_ratio']:.0%}.",
        f"- n/task={payload['n_per_task']}; tasks={','.join(payload['tasks'])}; max_new_tokens={payload['max_new_tokens']}.",
        f"- Cache note: {cache_note}.",
        "",
    ]
    if has_compact_prefill:
        lines += [
            "| task | n | ref_mask | after_prefill | compact_prefill | retained | replay_byte | compact_prefill_byte | prefill_s | decode_s | peak_GB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines += [
            "| task | n | ref_score | physical_score | exact_match | semantic | numeric | digit_prefix | retained | byte_ratio | peak_GB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    for r in payload["records"]:
        peak = r.get("peak_cuda_memory_gb")
        if has_compact_prefill:
            lines.append(
                f"| {r['task']} | {r['n']} | {r['reference_correctness']:.3f} | {r['physical_correctness']:.3f} | "
                f"{r.get('compact_prefill_correctness', 0.0):.3f} | {r['retained_ratio']:.3f} | "
                f"{r['compact_byte_ratio']:.3f} | {r.get('compact_prefill_byte_ratio', 0.0):.3f} | "
                f"{r.get('compact_prefill_prefill_seconds', 0.0):.2f} | "
                f"{r.get('compact_prefill_decode_seconds', 0.0):.2f} | "
                f"{'n/a' if peak is None else f'{peak:.2f}'} |"
            )
        else:
            lines.append(
                f"| {r['task']} | {r['n']} | {r['reference_correctness']:.3f} | {r['physical_correctness']:.3f} | "
                f"{r['physical_vs_reference_exact']:.3f} | {r['physical_vs_reference_semantic_match']:.3f} | "
                f"{r['physical_vs_reference_numeric_answer_exact']:.3f} | "
                f"{r['physical_vs_reference_digit_prefix']:.3f} | {r['retained_ratio']:.3f} | "
                f"{r['compact_byte_ratio']:.3f} | {'n/a' if peak is None else f'{peak:.2f}'} |"
            )
    lines += [
        "",
        f"- All physical decodes ran: {str(payload['all_physical_decodes_ran']).lower()}.",
        f"- Cache API modes: {', '.join(payload['cache_api_modes']) if payload['cache_api_modes'] else 'none'}.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    set_seed(args.seed)

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    retained_total, recent, guidepost_budget = local_ruler.budget_parts(args.context_length, args.budget_ratio)

    model, tokenizer = load_model_tokenizer(
        args.model,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )
    device = _model_device(model)
    samples, sample_meta = _build_samples(args, tokenizer, out_dir)
    rows: List[Dict[str, Any]] = []

    for sample in tqdm(samples, desc="physical replay"):
        prompt = sample["prompt"]
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            raise ValueError(f"Prompt too short for bootstrap replay: {sample['sample_id']}")
        seq_len = int(input_ids.shape[1])
        plan = _plan_for_sample(args, sample, tokenizer, recent, guidepost_budget)
        retained_indices = sorted(int(i) for i in plan.retained_indices)
        retained_ratio = len(retained_indices) / max(1, seq_len)
        bootstrap_ids = input_ids[:, -1:]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        unpatch_model(model)
        start = time.perf_counter()
        full_start = time.perf_counter()
        full_text, full_final_cache, full_warnings = _greedy_full_decode(model, tokenizer, input_ids, args.max_new_tokens)
        full_seconds = time.perf_counter() - full_start
        del full_final_cache
        _release_cuda()

        controller = _make_controller(model, args.context_length, args.budget_ratio, recent, guidepost_budget)
        controller.set_fixed_guidepost_indices(plan.guidepost_indices)
        patch_model(model, controller)
        prefix_ids = input_ids[:, :-1]
        reference_start = time.perf_counter()
        ref_prefill = _full_prefill(model, prefix_ids, full_warnings)
        reference_text, reference_final_cache, reference_warnings = _greedy_bootstrap_decode(
            model,
            tokenizer,
            ref_prefill.past_key_values,
            bootstrap_ids,
            seq_len - 1,
            args.max_new_tokens,
        )
        reference_seconds = time.perf_counter() - reference_start
        unpatch_model(model)
        del ref_prefill, reference_final_cache
        _release_cuda()

        replay_prefill_start = time.perf_counter()
        full_prefill = _full_prefill(model, input_ids, full_warnings)
        replay_full_prefill_seconds = time.perf_counter() - replay_prefill_start
        replay_decode_start = time.perf_counter()
        physical_text, _, stats, physical_warnings, cache_api_mode = _physical_decode(
            model,
            tokenizer,
            full_prefill.past_key_values,
            retained_indices,
            seq_len,
            bootstrap_ids,
            args.max_new_tokens,
        )
        replay_decode_seconds = time.perf_counter() - replay_decode_start
        del full_prefill
        _release_cuda()

        compact_prefill_text = None
        compact_prefill_correctness = None
        compact_prefill_raw_exact = None
        compact_prefill_semantic_match = None
        compact_prefill_stats: Dict[str, Any] | None = None
        compact_prefill_timings: Dict[str, float] | None = None
        compact_prefill_warnings: List[str] = []
        if args.include_compact_prefill:
            compact_prefill_text, compact_prefill_final_cache, compact_prefill_stats, compact_prefill_warnings, compact_prefill_timings = _compact_prefill_decode(
                model,
                tokenizer,
                input_ids,
                retained_indices,
                seq_len,
                args.max_new_tokens,
            )
            del compact_prefill_final_cache
            _release_cuda()

        seconds = time.perf_counter() - start
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None

        reference_correctness = _score(args, reference_text, sample)
        physical_correctness = _score(args, physical_text, sample)
        raw_exact = float(normalize_answer(physical_text) == normalize_answer(reference_text))
        canonical_text_exact = float(_canonical_text(physical_text) == _canonical_text(reference_text))
        numeric_answer_exact = _first_answer_like_exact(physical_text, reference_text)
        answer_set_exact = _answer_set_exact(physical_text, reference_text)
        semantic_match = float(
            (reference_correctness == 1.0 and physical_correctness == 1.0)
            or numeric_answer_exact == 1.0
            or answer_set_exact == 1.0
        )
        if compact_prefill_text is not None:
            compact_prefill_correctness = _score(args, compact_prefill_text, sample)
            compact_prefill_raw_exact = float(normalize_answer(compact_prefill_text) == normalize_answer(reference_text))
            compact_prefill_numeric_exact = _first_answer_like_exact(compact_prefill_text, reference_text)
            compact_prefill_answer_set_exact = _answer_set_exact(compact_prefill_text, reference_text)
            compact_prefill_semantic_match = float(
                (reference_correctness == 1.0 and compact_prefill_correctness == 1.0)
                or compact_prefill_numeric_exact == 1.0
                or compact_prefill_answer_set_exact == 1.0
            )
        row = {
            "sample_id": sample["sample_id"],
            "task_type": sample["task_type"],
            "source": args.source,
            "answers": sample["answers"],
            "prompt_tokens": seq_len,
            "full_text": full_text,
            "reference_mask_compass_text": reference_text,
            "physical_compact_text": physical_text,
            "compact_prefill_text": compact_prefill_text,
            "full_correctness": _score(args, full_text, sample),
            "reference_correctness": reference_correctness,
            "physical_correctness": physical_correctness,
            "compact_prefill_correctness": compact_prefill_correctness,
            "physical_vs_reference_exact": raw_exact,
            "physical_vs_reference_canonical_text_exact": canonical_text_exact,
            "physical_vs_reference_numeric_answer_exact": numeric_answer_exact,
            "physical_vs_reference_answer_set_exact": answer_set_exact,
            "physical_vs_reference_semantic_match": semantic_match,
            "physical_vs_reference_format_only_mismatch": float(raw_exact == 0.0 and semantic_match == 1.0),
            "physical_vs_reference_digit_prefix": _digit_prefix_match(physical_text, reference_text),
            "physical_vs_reference_normalized_edit": _normalized_edit(
                normalize_answer(physical_text),
                normalize_answer(reference_text),
            ),
            "retained_token_count": len(retained_indices),
            "after_prefill_compact_token_count": stats["compact_seq_len"],
            "retained_ratio": retained_ratio,
            "full_kv_bytes": stats["full_bytes"],
            "compact_kv_bytes": stats["compact_bytes"],
            "compact_byte_ratio": stats["byte_ratio"],
            "compact_prefill_token_count": None if compact_prefill_stats is None else compact_prefill_stats["compact_seq_len"],
            "compact_prefill_full_kv_bytes_estimated": None if compact_prefill_stats is None else compact_prefill_stats["full_bytes_estimated"],
            "compact_prefill_kv_bytes": None if compact_prefill_stats is None else compact_prefill_stats["compact_bytes"],
            "compact_prefill_byte_ratio": None if compact_prefill_stats is None else compact_prefill_stats["byte_ratio"],
            "compact_prefill_vs_reference_exact": compact_prefill_raw_exact,
            "compact_prefill_vs_reference_semantic_match": compact_prefill_semantic_match,
            "full_seq_len": stats["full_seq_len"],
            "compact_seq_len": stats["compact_seq_len"],
            "layer_count": stats["layer_count"],
            "cache_dtype": stats["dtype"],
            "cache_device": stats["device"],
            "full_generation_seconds": full_seconds,
            "reference_mask_seconds": reference_seconds,
            "after_prefill_replay_full_prefill_seconds": replay_full_prefill_seconds,
            "after_prefill_replay_decode_seconds": replay_decode_seconds,
            "compact_prefill_prefill_seconds": None if compact_prefill_timings is None else compact_prefill_timings["prefill_seconds"],
            "compact_prefill_decode_seconds": None if compact_prefill_timings is None else compact_prefill_timings["decode_seconds"],
            "compact_prefill_total_seconds": None if compact_prefill_timings is None else compact_prefill_timings["total_seconds"],
            "generation_seconds": seconds,
            "peak_cuda_memory_gb": peak_gb,
            "cache_api_mode": cache_api_mode,
            "cache_api_warnings": full_warnings + reference_warnings + physical_warnings + compact_prefill_warnings,
            "selector": "compass_prefill_guidepost",
            "uses_decode_time_qk_selector": False,
        }
        rows.append(row)
        append_jsonl(results_path, row)

    unpatch_model(model)
    records = _aggregate(rows)
    payload = {
        "model": args.model,
        "context_length": args.context_length,
        "budget_ratio": args.budget_ratio,
        "retained_total": retained_total,
        "recent_size": recent,
        "guidepost_budget": guidepost_budget,
        "n_per_task": args.n_per_task,
        "tasks": tasks,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "source": args.source,
        "include_compact_prefill": args.include_compact_prefill,
        "sample_meta": sample_meta,
        "records": records,
        "cache_api_modes": sorted({row["cache_api_mode"] for row in rows}),
        "all_physical_decodes_ran": len(rows) == len(samples),
        "scope": "offline compact cache replay after prompt encoding"
        + (" plus compact prompt replay" if args.include_compact_prefill else ""),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_summary(out_dir, payload)
    print(json.dumps({"rows": len(rows), "out_dir": str(out_dir), "records": records}, sort_keys=True))


if __name__ == "__main__":
    main()
