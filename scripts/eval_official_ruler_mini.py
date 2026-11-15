#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

from proofkv.config import CompassConfig, Route
from proofkv.controllers import CompassKVController
from proofkv.eval.generation import generate_text, load_model_tokenizer
from proofkv.patch import patch_model, unpatch_model
from proofkv.prefill_selectors import GuidepostPlan, block_halo_guideposts, compass_prefill_guideposts
from proofkv.routing import make_uniform_route_table
from proofkv.utils.io import append_jsonl, ensure_dir, read_jsonl
from proofkv.utils.seed import set_seed


RULER_ROOT = Path("third_party/NVIDIA_RULER")
RULER_DEPS = Path("third_party/ruler_deps")
SINK = 4
BASE_RETAINED_TOTAL = 640
BASE_RECENT = 512
DEFAULT_TASKS = "niah,variable_tracking"
SUPPORTED_METHODS = {"full", "uniform_block_halo", "compass_prefill_guidepost"}


NIAH_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
    "{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
    " The special magic {type_needle_v} for {query} mentioned in the provided text are"
)

VARIABLE_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
    "{context}\n"
    "Question: Find all variables that are assigned the value {query} in the text above."
    " Answer: According to the chain(s) of variable assignment in the text above, "
    "{num_v} variables are assigned the value {query}, they are: "
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mini official RULER synthetic gate using NVIDIA/RULER generators.")
    p.add_argument("--model", required=True)
    p.add_argument("--context-length", type=int, default=8000)
    p.add_argument("--budget-ratio", type=float, default=0.08)
    p.add_argument("--n-per-task", type=int, default=16)
    p.add_argument("--methods", default="full,uniform_block_halo,compass_prefill_guidepost")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--max-new-tokens", type=int, default=32)
    return p.parse_args()


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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


def prompt_tokens(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def selector_query(prompt: str) -> str:
    markers = ["\nQuestion:", "Question:", "\nWhat ", " What "]
    positions = [prompt.rfind(m) for m in markers]
    pos = max(positions)
    return prompt[pos:] if pos >= 0 else prompt[-1200:]


def score_prediction(prediction: str, refs: Sequence[str]) -> float:
    pred = prediction.lower()
    if not refs:
        return 0.0
    return sum(1.0 if str(ref).lower() in pred else 0.0 for ref in refs) / len(refs)


def env_with_ruler_deps() -> Dict[str, str]:
    env = os.environ.copy()
    additions = [str(RULER_DEPS)]
    existing = env.get("PYTHONPATH")
    if existing:
        additions.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(additions)
    return env


def run_generator(task: str, args: argparse.Namespace, data_dir: Path) -> Path:
    if not RULER_ROOT.exists():
        raise FileNotFoundError(f"Official RULER repo not found at {RULER_ROOT}")
    script_dir = RULER_ROOT / "scripts" / "data" / "synthetic"
    save_name = task
    common = [
        sys.executable,
        str(script_dir / f"{task}.py"),
        "--save_dir",
        str(data_dir),
        "--save_name",
        save_name,
        "--tokenizer_path",
        args.model,
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(args.context_length),
        "--num_samples",
        str(args.n_per_task),
        "--random_seed",
        str(args.seed),
    ]
    if task == "niah":
        cmd = common + [
            "--tokens_to_generate",
            "32",
            "--num_needle_k",
            "1",
            "--num_needle_v",
            "1",
            "--num_needle_q",
            "1",
            "--type_haystack",
            "noise",
            "--type_needle_k",
            "words",
            "--type_needle_v",
            "numbers",
            "--template",
            NIAH_TEMPLATE,
        ]
    elif task == "variable_tracking":
        cmd = common + [
            "--tokens_to_generate",
            "32",
            "--type_haystack",
            "noise",
            "--num_chains",
            "1",
            "--num_hops",
            "4",
            "--template",
            VARIABLE_TEMPLATE,
        ]
    else:
        raise ValueError(f"Unsupported official RULER mini task {task}")
    subprocess.run(cmd, check=True, env=env_with_ruler_deps())
    return data_dir / save_name / "validation.jsonl"


def make_samples(args: argparse.Namespace, out_dir: Path, tasks: Sequence[str]) -> List[Dict[str, Any]]:
    data_dir = ensure_dir(out_dir / "official_ruler_generated")
    samples: List[Dict[str, Any]] = []
    for task in tasks:
        path = run_generator(task, args, data_dir)
        rows = read_jsonl(path)
        for row in rows:
            prompt = str(row["input"]) + str(row.get("answer_prefix", ""))
            outputs = [str(x) for x in row.get("outputs", [])]
            samples.append(
                {
                    "task_type": task,
                    "sample_id": f"{task}_{row.get('index', len(samples))}",
                    "prompt": prompt,
                    "outputs": outputs,
                    "source_row": row,
                    "source_file": str(path),
                }
            )
    return samples


def plan_for(
    method: str,
    sample_id: str,
    prompt: str,
    tokenizer: Any,
    recent: int,
    guidepost_budget: int,
) -> GuidepostPlan | None:
    seq_len = prompt_tokens(tokenizer, prompt)
    if method == "uniform_block_halo":
        return block_halo_guideposts(seq_len, guidepost_budget, SINK, recent, mode="uniform")
    if method == "compass_prefill_guidepost":
        offsets = token_offsets(tokenizer, prompt)
        return compass_prefill_guideposts(
            prompt,
            selector_query(prompt),
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


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    return records


def write_summary(out_dir: Path, payload: Dict[str, Any]) -> None:
    records = payload["records"]
    lines = [
        "# Official RULER Mini Gate",
        "",
        f"- Source: `third_party/NVIDIA_RULER`, commit `{payload['ruler_commit']}`.",
        f"- Data generation: {payload['data_generation']} using tasks {','.join(payload['tasks'])}.",
        f"- Model: `{payload['model']}`; context={payload['context_length']}; budget={payload['budget_ratio']:.0%}; n/task={payload['n_per_task']}.",
        f"- Gate positive: {str(payload['gate_positive']).lower()}.",
        "",
        "| method | task | score | retained | avg_s/sample | peak_GB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in records:
        avg = r.get("avg_generation_seconds")
        peak = r.get("peak_cuda_memory_gb")
        lines.append(
            f"| {r['method']} | {r['task']} | {r['score']:.3f} | {r['retained_ratio']:.3f} | "
            f"{'n/a' if avg is None else f'{avg:.2f}'} | {'n/a' if peak is None else f'{peak:.2f}'} |"
        )
    lines += [
        "",
        f"- Full average score: {payload['full_average_score']:.3f}.",
        f"- Compass average score: {payload['compass_average_score']:.3f}.",
        f"- Uniform average score: {payload['uniform_average_score']:.3f}.",
        f"- Compass retains >=70% of Full average: {str(payload['compass_ge_70pct_full']).lower()}.",
        "- Scope: official RULER NIAH and variable-tracking slices.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def average_for(records: Sequence[Dict[str, Any]], method: str) -> float:
    vals = [float(r["score"]) for r in records if r["method"] == method]
    return sum(vals) / max(1, len(vals))


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(methods) - SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unsupported_tasks = sorted(set(tasks) - {"niah", "variable_tracking"})
    if unsupported_tasks:
        raise ValueError(f"Unsupported official RULER mini tasks: {unsupported_tasks}")

    out_dir = ensure_dir(args.out_dir)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    set_seed(args.seed)
    retained_total, recent, guidepost_budget = budget_parts(args.context_length, args.budget_ratio)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = make_samples(args, out_dir, tasks)
    max_prompt_tokens = max(prompt_tokens(tokenizer, s["prompt"]) for s in samples) if samples else 0

    model, _ = load_model_tokenizer(
        args.model,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )
    rows: List[Dict[str, Any]] = []
    for method in methods:
        controller = install_method(model, method, args.context_length, args.budget_ratio, recent, guidepost_budget)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for sample in tqdm(samples, desc=f"eval {method}"):
            sample_id = f"{sample['sample_id']}_{method}"
            plan = plan_for(method, sample_id, sample["prompt"], tokenizer, recent, guidepost_budget)
            if controller is not None:
                if plan is None:
                    controller.clear_fixed_guidepost_indices()
                else:
                    controller.set_fixed_guidepost_indices(plan.guidepost_indices)
            seq_len = prompt_tokens(tokenizer, sample["prompt"])
            start = time.perf_counter()
            pred = generate_text(
                model,
                tokenizer,
                sample["prompt"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
            seconds = time.perf_counter() - start
            score = score_prediction(pred, sample["outputs"])
            retained_ratio = 1.0 if method == "full" else (len(plan.retained_indices) / max(1, seq_len) if plan else retained_total / max(1, seq_len))
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
            row = {
                "sample_id": sample["sample_id"],
                "task_type": sample["task_type"],
                "method": method,
                "answers": sample["outputs"],
                "prediction": pred,
                "score": score,
                "prompt_tokens": seq_len,
                "budget_target": args.budget_ratio if method != "full" else 1.0,
                "retained_ratio": retained_ratio,
                "physical_ratio_note": "reference-mask/theoretical retained-token ratio; no physical KV cache reduction",
                "generation_seconds": seconds,
                "cuda_peak_allocated_gb": peak_gb,
                "source": {
                    "repo": "https://github.com/NVIDIA/RULER",
                    "local_path": str(RULER_ROOT),
                    "commit": git_commit(RULER_ROOT),
                    "generated_file": sample["source_file"],
                },
            }
            rows.append(row)
            append_jsonl(results_path, row)
    unpatch_model(model)

    records = aggregate(rows)
    full_avg = average_for(records, "full")
    uniform_avg = average_for(records, "uniform_block_halo")
    compass_avg = average_for(records, "compass_prefill_guidepost")
    compass_beats_uniform_tasks = [
        task
        for task in tasks
        if next((r["score"] for r in records if r["method"] == "compass_prefill_guidepost" and r["task"] == task), -1.0)
        > next((r["score"] for r in records if r["method"] == "uniform_block_halo" and r["task"] == task), -1.0)
    ]
    gate_positive = (
        full_avg >= 0.5
        and (len(compass_beats_uniform_tasks) >= max(1, len(tasks) // 2))
        and (compass_avg >= 0.7 * full_avg or compass_avg - uniform_avg >= 0.30)
    )
    payload = {
        "model": args.model,
        "context_length": args.context_length,
        "budget_ratio": args.budget_ratio,
        "retained_total": retained_total,
        "recent_size": recent,
        "guidepost_budget": guidepost_budget,
        "n_per_task": args.n_per_task,
        "tasks": tasks,
        "methods": methods,
        "max_prompt_tokens": max_prompt_tokens,
        "records": records,
        "ruler_repo": "https://github.com/NVIDIA/RULER",
        "ruler_local_path": str(RULER_ROOT),
        "ruler_commit": git_commit(RULER_ROOT),
        "data_generation": "official NVIDIA/RULER synthetic generators invoked from third_party/NVIDIA_RULER",
        "full_average_score": full_avg,
        "uniform_average_score": uniform_avg,
        "compass_average_score": compass_avg,
        "compass_beats_uniform_tasks": compass_beats_uniform_tasks,
        "compass_ge_70pct_full": compass_avg >= 0.7 * full_avg if full_avg > 0 else False,
        "gate_positive": gate_positive,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_summary(out_dir, payload)
    print(json.dumps({"rows": len(rows), "gate_positive": gate_positive, "out_dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
