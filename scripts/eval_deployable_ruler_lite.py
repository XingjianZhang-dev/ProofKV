#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

from proofkv.config import CompassConfig, Route
from proofkv.controllers import CompassKVController
from proofkv.data.synthetic import SyntheticSample, make_ruler_lite_suite
from proofkv.eval.generation import generate_text, load_model_tokenizer
from proofkv.eval.metrics import contains_answer
from proofkv.patch import patch_model, unpatch_model
from proofkv.prefill_selectors import GuidepostPlan, block_halo_guideposts, compass_prefill_guideposts
from proofkv.routing import make_uniform_route_table
from proofkv.utils.io import append_jsonl, ensure_dir
from proofkv.utils.seed import set_seed


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SINK = 4
BASE_RETAINED_TOTAL = 640
BASE_RECENT = 512
MAX_NEW = 8
SUPPORTED_METHODS = {"full", "random_block_halo", "uniform_block_halo", "compass_prefill_guidepost"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run deployable CompassKV methods on the local RULER-lite synthetic path.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--context-length", type=int, default=8000)
    p.add_argument("--budget-ratio", type=float, default=0.08)
    p.add_argument("--n-per-task", type=int, default=8)
    p.add_argument("--methods", default="full,random_block_halo,uniform_block_halo,compass_prefill_guidepost")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--out-dir", default="outputs/post_overnight_next_step/ruler_lite_plumbing/smoke")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tasks", default="exact_retrieval,multi_hop")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW)
    p.add_argument("--strict-token-fit", action="store_true")
    p.add_argument("--token-fit-safety-margin", type=int, default=64)
    return p.parse_args()


def budget_parts(context_length: int, budget_ratio: float) -> tuple[int, int, int]:
    retained_total = max(SINK + 1, int(round(context_length * budget_ratio)))
    remaining = max(1, retained_total - SINK)
    recent_frac = BASE_RECENT / max(1, BASE_RETAINED_TOTAL - SINK)
    recent = max(1, min(remaining, int(round(remaining * recent_frac))))
    guidepost_budget = max(0, remaining - recent)
    return retained_total, recent, guidepost_budget


def token_offsets(tokenizer: Any, prompt: str) -> List[Tuple[int, int]]:
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    return [(int(s), int(e)) for s, e in enc["offset_mapping"]]


def prompt_tokens(tokenizer: Any, prompt: str, warn: bool = True) -> int:
    kwargs = {"add_special_tokens": False}
    if not warn:
        kwargs["verbose"] = False
    try:
        return len(tokenizer(prompt, **kwargs)["input_ids"])
    except TypeError:
        kwargs.pop("verbose", None)
        return len(tokenizer(prompt, **kwargs)["input_ids"])


def strict_token_fit_limit(
    model_config: Any,
    context_length: int,
    max_new_tokens: int,
    safety_margin: int,
) -> tuple[int, int | None]:
    max_pos = getattr(model_config, "max_position_embeddings", None)
    max_pos_int = int(max_pos) if isinstance(max_pos, int) and max_pos > 0 else None
    base_limit = min(context_length, max_pos_int) if max_pos_int is not None else context_length
    return max(1, int(base_limit) - int(max_new_tokens) - int(safety_margin)), max_pos_int


def query_text(prompt: str) -> str:
    idx = prompt.rfind("\nQuestion:")
    return prompt[idx:] if idx >= 0 else prompt[-1000:]


def make_context_fit_suite(
    tokenizer: Any,
    n_per_task: int,
    context_length: int,
    seed: int,
    tasks: set[str],
) -> tuple[List[SyntheticSample], int, int]:
    filler_tokens = max(64, context_length)
    samples: List[SyntheticSample] = []
    max_prompt_tokens = 0
    for _ in range(8):
        all_samples = make_ruler_lite_suite(n_per_task=n_per_task, filler_tokens=filler_tokens, seed=seed)
        samples = [s for s in all_samples if s.task_type in tasks]
        max_prompt_tokens = max(prompt_tokens(tokenizer, s.prompt) for s in samples) if samples else 0
        if max_prompt_tokens <= context_length:
            break
        scale = max(0.1, 0.92 * context_length / max(1, max_prompt_tokens))
        next_filler = max(64, int(filler_tokens * scale))
        if next_filler == filler_tokens:
            next_filler = max(64, filler_tokens - 64)
        filler_tokens = next_filler
    return samples, filler_tokens, max_prompt_tokens


def make_strict_token_fit_suite(
    tokenizer: Any,
    n_per_task: int,
    context_length: int,
    seed: int,
    tasks: set[str],
    max_prompt_tokens: int,
) -> tuple[List[SyntheticSample], int, int, Dict[str, Any]]:
    filler_tokens = max(64, context_length)
    candidate_count = 0
    skipped_count = 0
    trials: List[Dict[str, Any]] = []
    task_order = sorted(tasks)

    for _ in range(8):
        all_samples = make_ruler_lite_suite(n_per_task=n_per_task, filler_tokens=filler_tokens, seed=seed)
        samples = [s for s in all_samples if s.task_type in tasks]
        lengths = [prompt_tokens(tokenizer, s.prompt, warn=False) for s in samples]
        candidate_count += len(samples)
        skipped_count += sum(int(length > max_prompt_tokens) for length in lengths)

        selected: List[SyntheticSample] = []
        selected_lengths: List[int] = []
        counts: Dict[str, int] = {task: 0 for task in task_order}
        for sample, length in zip(samples, lengths):
            if length > max_prompt_tokens or counts[sample.task_type] >= n_per_task:
                continue
            selected.append(sample)
            selected_lengths.append(length)
            counts[sample.task_type] += 1

        max_trial = max(lengths) if lengths else 0
        trials.append(
            {
                "filler_tokens": filler_tokens,
                "candidate_count": len(samples),
                "fit_count": sum(int(length <= max_prompt_tokens) for length in lengths),
                "skipped_count": sum(int(length > max_prompt_tokens) for length in lengths),
                "max_prompt_tokens": max_trial,
                "selected_counts": dict(counts),
            }
        )
        if all(counts.get(task, 0) >= n_per_task for task in task_order):
            max_selected = max(selected_lengths) if selected_lengths else 0
            stats = {
                "enabled": True,
                "max_prompt_tokens": max_prompt_tokens,
                "candidate_count": candidate_count,
                "skipped_count": skipped_count,
                "selected_count": len(selected),
                "max_selected_prompt_tokens": max_selected,
                "filler_tokens": filler_tokens,
                "trials": trials,
            }
            return selected, filler_tokens, max_selected, stats

        scale = max(0.1, 0.92 * max_prompt_tokens / max(1, max_trial))
        next_filler = max(64, int(filler_tokens * scale))
        if next_filler == filler_tokens:
            next_filler = max(64, filler_tokens - 64)
        filler_tokens = next_filler

    raise RuntimeError(
        "Strict token-fit could not select enough fitting samples: "
        f"counts={trials[-1]['selected_counts'] if trials else {}}, limit={max_prompt_tokens}"
    )


def plan_for(
    method: str,
    sample_id: str,
    sample: SyntheticSample,
    tokenizer: Any,
    recent: int,
    guidepost_budget: int,
) -> GuidepostPlan | None:
    seq_len = prompt_tokens(tokenizer, sample.prompt)
    if method == "random_block_halo":
        stable_seed = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)
        return block_halo_guideposts(seq_len, guidepost_budget, SINK, recent, mode="random", seed=stable_seed)
    if method == "uniform_block_halo":
        return block_halo_guideposts(seq_len, guidepost_budget, SINK, recent, mode="uniform")
    if method == "compass_prefill_guidepost":
        offsets = token_offsets(tokenizer, sample.prompt)
        return compass_prefill_guideposts(
            sample.prompt,
            query_text(sample.prompt),
            offsets,
            guidepost_budget,
            SINK,
            recent,
            block_size=16,
            halo=4,
            propagation_rounds=5,
        )
    return None


def install_method(
    model: Any,
    method: str,
    context_length: int,
    budget_ratio: float,
    recent: int,
    guidepost_budget: int,
) -> CompassKVController | None:
    unpatch_model(model)
    if method == "full":
        return None
    layers = int(model.config.num_hidden_layers)
    heads = int(model.config.num_attention_heads)
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
    table = make_uniform_route_table(layers, heads, Route.NAV, {"method": method, "context_length": context_length})
    controller = CompassKVController(cfg, table)
    patch_model(model, controller)
    return controller


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["task_type"])].append(row)
    records = []
    for (method, task), vals in sorted(grouped.items()):
        score = sum(float(v["score"]) for v in vals) / max(1, len(vals))
        retained = sum(float(v["retained_ratio"]) for v in vals) / max(1, len(vals))
        times = [float(v["generation_seconds"]) for v in vals if v.get("generation_seconds") is not None]
        peaks = [float(v["cuda_peak_allocated_gb"]) for v in vals if v.get("cuda_peak_allocated_gb") is not None]
        records.append(
            {
                "method": method,
                "task": task,
                "n": len(vals),
                "score": score,
                "retained_ratio": retained,
                "avg_generation_seconds": None if not times else sum(times) / len(times),
                "peak_cuda_memory_gb": None if not peaks else max(peaks),
            }
        )
    return {"records": records}


def write_summary(out_dir: Path, payload: Dict[str, Any]) -> None:
    records = payload["records"]
    dry_run = payload.get("dry_run", False)
    lines = [
        "# Deployable RULER-lite Plumbing",
        "",
        "- Script: `scripts/eval_deployable_ruler_lite.py`.",
        f"- Run mode: {'dry-run' if dry_run else 'generation'}.",
        f"- Generation status: {'skipped in dry-run mode' if dry_run else 'completed'}.",
        f"- Context/budget: {payload['context_length']} / {payload['budget_ratio']:.0%}; tasks={','.join(payload['tasks'])}.",
        f"- Prompt token max after fitting: {payload['max_prompt_tokens']}; synthetic filler tokens={payload['filler_tokens']}.",
    ]
    token_fit = payload.get("token_fit") or {}
    if token_fit.get("enabled"):
        lines.append(
            f"- Strict token-fit: limit={token_fit['max_prompt_tokens']}; selected={token_fit['selected_count']}; "
            f"skipped={token_fit['skipped_count']}; max_selected={token_fit['max_selected_prompt_tokens']}."
        )
    if records:
        lines += ["", "| method | task | score | retained | avg_s/sample | peak_GB |", "|---|---|---:|---:|---:|---:|"]
        for r in records:
            avg = r.get("avg_generation_seconds")
            peak = r.get("peak_cuda_memory_gb")
            lines.append(
                f"| {r['method']} | {r['task']} | {r['score']:.3f} | {r['retained_ratio']:.3f} | "
                f"{'n/a' if avg is None else f'{avg:.2f}'} | {'n/a' if peak is None else f'{peak:.2f}'} |"
            )
    beats = payload.get("compass_beats_uniform")
    if beats is not None:
        lines.append(f"- Deployable compass beats uniform_block_halo on at least one task: {str(beats).lower()}.")
    lines.append("- Scope: local RULER-lite exact retrieval and multi-hop.")
    lines.append(f"- RULER-lite ready: {str(payload.get('ruler_lite_ready', False)).lower()}.")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(methods) - SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"Unsupported deployable methods: {unknown}")
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    out_dir = ensure_dir(args.out_dir)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    set_seed(args.seed)
    retained_total, recent, guidepost_budget = budget_parts(args.context_length, args.budget_ratio)

    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_fit_stats: Dict[str, Any] = {"enabled": False}
    if args.strict_token_fit:
        model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        fit_limit, model_max_position_embeddings = strict_token_fit_limit(
            model_config,
            args.context_length,
            args.max_new_tokens,
            args.token_fit_safety_margin,
        )
        samples, filler_tokens, max_prompt_tokens, token_fit_stats = make_strict_token_fit_suite(
            tokenizer,
            args.n_per_task,
            args.context_length,
            args.seed,
            tasks,
            fit_limit,
        )
        token_fit_stats.update(
            {
                "model_max_position_embeddings": model_max_position_embeddings,
                "context_length": args.context_length,
                "max_new_tokens": args.max_new_tokens,
                "safety_margin": args.token_fit_safety_margin,
            }
        )
    else:
        samples, filler_tokens, max_prompt_tokens = make_context_fit_suite(
            tokenizer, args.n_per_task, args.context_length, args.seed, tasks
        )
    plan_rows = []
    for idx, sample in enumerate(samples):
        sample_id = f"{sample.task_type}_{idx}"
        for method in methods:
            plan = plan_for(method, sample_id, sample, tokenizer, recent, guidepost_budget)
            seq_len = prompt_tokens(tokenizer, sample.prompt)
            retained_ratio = 1.0 if method == "full" else (len(plan.retained_indices) / max(1, seq_len) if plan else retained_total / max(1, seq_len))
            plan_rows.append({"sample_id": sample_id, "task_type": sample.task_type, "method": method, "prompt_tokens": seq_len, "retained_ratio": retained_ratio})

    if args.dry_run:
        payload = {
            "dry_run": True,
            "model": args.model,
            "context_length": args.context_length,
            "budget_ratio": args.budget_ratio,
            "retained_total": retained_total,
            "recent_size": recent,
            "guidepost_budget": guidepost_budget,
            "n_per_task": args.n_per_task,
            "tasks": sorted(tasks),
            "methods": methods,
            "filler_tokens": filler_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "plan_rows": plan_rows,
            "records": [],
            "token_fit": token_fit_stats,
            "ruler_lite_ready": False,
        }
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        write_summary(out_dir, payload)
        print(json.dumps({"dry_run": True, "samples": len(samples), "methods": methods, "out_dir": str(out_dir)}, sort_keys=True))
        return

    model, _ = load_model_tokenizer(args.model, dtype=args.dtype, attn_implementation=args.attn_implementation, device_map="auto")
    rows: List[Dict[str, Any]] = []
    for method in methods:
        controller = install_method(model, method, args.context_length, args.budget_ratio, recent, guidepost_budget)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for idx, sample in enumerate(tqdm(samples, desc=f"eval {method}")):
            sample_id = f"{sample.task_type}_{idx}"
            plan = plan_for(method, sample_id, sample, tokenizer, recent, guidepost_budget)
            if controller is not None:
                if plan is None:
                    controller.clear_fixed_guidepost_indices()
                else:
                    controller.set_fixed_guidepost_indices(plan.guidepost_indices)
            seq_len = prompt_tokens(tokenizer, sample.prompt)
            start = time.perf_counter()
            pred = generate_text(model, tokenizer, sample.prompt, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0)
            seconds = time.perf_counter() - start
            score = contains_answer(pred, sample.answer)
            retained_ratio = 1.0 if method == "full" else (len(plan.retained_indices) / max(1, seq_len) if plan else retained_total / max(1, seq_len))
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
            row = {
                "sample_id": sample_id,
                "task_type": sample.task_type,
                "method": method,
                "answer": sample.answer,
                "prediction": pred,
                "score": score,
                "metadata": sample.metadata,
                "prompt_tokens": seq_len,
                "budget_target": args.budget_ratio if method != "full" else 1.0,
                "retained_ratio": retained_ratio,
                "physical_ratio_note": "intended retained span ratio; Python reference keeps original HF cache resident",
                "generation_seconds": seconds,
                "cuda_peak_allocated_gb": peak_gb,
            }
            rows.append(row)
            append_jsonl(results_path, row)
    unpatch_model(model)

    payload = {
        "dry_run": False,
        "model": args.model,
        "context_length": args.context_length,
        "budget_ratio": args.budget_ratio,
        "retained_total": retained_total,
        "recent_size": recent,
        "guidepost_budget": guidepost_budget,
        "n_per_task": args.n_per_task,
        "tasks": sorted(tasks),
        "methods": methods,
        "filler_tokens": filler_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "token_fit": token_fit_stats,
        "records": aggregate(rows)["records"],
        "ruler_lite_ready": True,
    }
    by_method_task = {(r["method"], r["task"]): r["score"] for r in payload["records"]}
    payload["compass_beats_uniform"] = any(
        by_method_task.get(("compass_prefill_guidepost", task), -1.0)
        > by_method_task.get(("uniform_block_halo", task), -1.0)
        for task in tasks
    )
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_summary(out_dir, payload)
    print(json.dumps({"dry_run": False, "rows": len(rows), "out_dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
