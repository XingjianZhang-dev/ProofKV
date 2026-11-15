#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_fair_contract_external_20260726 import (
    PROOF_METHOD,
    ROOT_METHOD,
    TASKS,
    clustered_inference,
    load_jsonl,
    sha256_file,
)


LLMLINGUA_METHOD = "llmlingua2_official"
CAPACITIES = (128, 256, 512)
MODELS = ("qwen25", "mistral02")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof-qwen-dir", required=True)
    parser.add_argument("--proof-mistral-dir", required=True)
    parser.add_argument("--llmlingua-qwen-dir", required=True)
    parser.add_argument("--llmlingua-mistral-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    parser.add_argument("--permutation-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_method_rows(
    model: str,
    run_dir: Path,
    method: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    for path in (results_path, manifest_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = []
    for source in load_jsonl(results_path):
        if source["method"] != method:
            continue
        if int(source["capacity"]) not in CAPACITIES:
            continue
        row = dict(source)
        row["model_label"] = model
        rows.append(row)
    return rows, {
        "results_sha256": sha256_file(results_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "metrics_sha256": sha256_file(metrics_path),
    }


def main() -> None:
    args = parse_args()
    proof_dirs = {
        "qwen25": Path(args.proof_qwen_dir),
        "mistral02": Path(args.proof_mistral_dir),
    }
    llmlingua_dirs = {
        "qwen25": Path(args.llmlingua_qwen_dir),
        "mistral02": Path(args.llmlingua_mistral_dir),
    }
    rows: list[dict[str, Any]] = []
    proof_timing_seeds: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}
    hashes: dict[str, dict[str, dict[str, str]]] = {}
    for model in MODELS:
        proof_rows, proof_hashes = _load_method_rows(
            model,
            proof_dirs[model],
            PROOF_METHOD,
        )
        llmlingua_rows, llmlingua_hashes = _load_method_rows(
            model,
            llmlingua_dirs[model],
            LLMLINGUA_METHOD,
        )
        rows.extend(proof_rows)
        rows.extend(llmlingua_rows)
        for source in load_jsonl(
            proof_dirs[model] / "results.jsonl"
        ):
            if (
                source["method"] == ROOT_METHOD
                and int(source["capacity"]) == 64
            ):
                timing_key = (
                    model,
                    str(source["task"]),
                    str(source["id"]),
                )
                if timing_key in proof_timing_seeds:
                    raise ValueError(
                        f"duplicate ProofKV timing seed: {timing_key}"
                    )
                proof_timing_seeds[timing_key] = source
        hashes[model] = {
            "proofkv": proof_hashes,
            "llmlingua2": llmlingua_hashes,
        }

    by_key: dict[
        tuple[str, str, str, int, str],
        dict[str, Any],
    ] = {}
    for row in rows:
        key = (
            str(row["model_label"]),
            str(row["task"]),
            str(row["id"]),
            int(row["capacity"]),
            str(row["method"]),
        )
        if key in by_key:
            raise ValueError(f"duplicate row: {key}")
        if (
            not math.isfinite(float(row["score"]))
            or not 0.0 <= float(row["score"]) <= 1.0
            or float(row["actual_retained_tokens"])
            > int(row["capacity"]) + 1e-6
        ):
            raise ValueError(f"invalid row: {key}")
        by_key[key] = row

    cells: list[dict[str, Any]] = []
    clusters: dict[tuple[str, str, int], np.ndarray] = {}
    retention_delta: list[float] = []
    for model in MODELS:
        for task in TASKS:
            ids = sorted(
                {
                    key[2]
                    for key in by_key
                    if key[0] == model and key[1] == task
                }
            )
            if len(ids) != 46:
                raise ValueError(
                    f"{model}/{task}: expected 46 ids, found {len(ids)}"
                )
            for capacity in CAPACITIES:
                proof = [
                    by_key[
                        (
                            model,
                            task,
                            source_id,
                            capacity,
                            PROOF_METHOD,
                        )
                    ]
                    for source_id in ids
                ]
                baseline = [
                    by_key[
                        (
                            model,
                            task,
                            source_id,
                            capacity,
                            LLMLINGUA_METHOD,
                        )
                    ]
                    for source_id in ids
                ]
                difference = np.asarray(
                    [
                        float(a["score"]) - float(b["score"])
                        for a, b in zip(proof, baseline)
                    ],
                    dtype=np.float64,
                )
                clusters[(model, task, capacity)] = difference
                retention_delta.extend(
                    float(a["actual_retained_tokens"])
                    - float(b["actual_retained_tokens"])
                    for a, b in zip(proof, baseline)
                )
                scorer_once = [
                    float(
                        proof_timing_seeds[
                            (model, task, source_id)
                        ]["selection_seconds"]
                    )
                    for source_id in ids
                ]
                proof_compile = [
                    float(row["selection_seconds"])
                    for row in proof
                ]
                standalone_proof_selection = [
                    scorer + compile_time
                    for scorer, compile_time in zip(
                        scorer_once,
                        proof_compile,
                    )
                ]
                cells.append(
                    {
                        "model": model,
                        "task": task,
                        "capacity": capacity,
                        "n": len(ids),
                        "llmlingua2_f1": 100.0
                        * float(
                            np.mean(
                                [float(row["score"]) for row in baseline]
                            )
                        ),
                        "proofkv_f1": 100.0
                        * float(
                            np.mean(
                                [float(row["score"]) for row in proof]
                            )
                        ),
                        "delta_f1": 100.0 * float(difference.mean()),
                        "llmlingua2_total_seconds": float(
                            np.mean(
                                [float(row["seconds"]) for row in baseline]
                            )
                        ),
                        "proofkv_total_seconds": float(
                            np.mean(
                                [
                                    selection_time
                                    + float(row["generation_seconds"])
                                    for selection_time, row in zip(
                                        standalone_proof_selection,
                                        proof,
                                    )
                                ]
                            )
                        ),
                        "llmlingua2_selection_seconds": float(
                            np.mean(
                                [
                                    float(row["selection_seconds"])
                                    for row in baseline
                                ]
                            )
                        ),
                        "proofkv_selection_seconds": float(
                            np.mean(standalone_proof_selection)
                        ),
                        "proofkv_cached_compile_seconds": float(
                            np.mean(proof_compile)
                        ),
                        "proofkv_scorer_once_seconds": float(
                            np.mean(scorer_once)
                        ),
                        "llmlingua2_retained_tokens": float(
                            np.mean(
                                [
                                    float(row["actual_retained_tokens"])
                                    for row in baseline
                                ]
                            )
                        ),
                        "proofkv_retained_tokens": float(
                            np.mean(
                                [
                                    float(row["actual_retained_tokens"])
                                    for row in proof
                                ]
                            )
                        ),
                    }
                )

    endpoint = clustered_inference(
        clusters,
        args.bootstrap_replicates,
        args.permutation_replicates,
        np.random.default_rng(args.seed),
    )
    endpoint["passes_confirmatory_gate"] = (
        endpoint["delta_f1_points"] > 0.0
        and endpoint["ci95_low"] > 0.0
        and endpoint["permutation_p_two_sided"] < 0.05
    )
    payload = {
        "status": (
            "PASS"
            if endpoint["passes_confirmatory_gate"]
            else "NEGATIVE_OR_INCONCLUSIVE"
        ),
        "protocol": "llmlingua2_same_compact_prefill_contract_V2",
        "methods": {
            "prompt_compressor": LLMLINGUA_METHOD,
            "proofkv": PROOF_METHOD,
        },
        "endpoint": endpoint,
        "retention_contract": {
            "same_total_kv_cap": True,
            "mean_proof_minus_llmlingua_tokens": float(
                np.mean(retention_delta)
            ),
            "max_abs_proof_minus_llmlingua_tokens": float(
                np.max(np.abs(retention_delta))
            ),
        },
        "timing_contract": {
            "proofkv_standalone_selection_seconds": (
                "capacity-64 learned_root_topk selection_seconds, which "
                "contains the once-per-query Evidence Index and learned "
                "overlay construction under the runner's fixed method "
                "order, plus the requested-capacity ProofKV cached "
                "compiler selection_seconds"
            ),
            "reason": (
                "The formal quality runner caches the shared learned "
                "overlay across methods and capacities. Reporting only "
                "the ProofKV row would omit scorer/index construction."
            ),
            "llmlingua2_selection_seconds": (
                "all budget-normalization compressor attempts for the "
                "requested capacity"
            ),
        },
        "cells": cells,
        "source_hashes": hashes,
        "inference": {
            "bootstrap_replicates": args.bootstrap_replicates,
            "permutation_replicates": args.permutation_replicates,
            "seed": args.seed,
            "cluster_grain": "model_task_capacity",
            "task_model_capacity_weighting": "equal",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "endpoint": endpoint,
                "out": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
