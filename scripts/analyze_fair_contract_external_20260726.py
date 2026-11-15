#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT_METHOD = "learned_root_topk"
PROOF_METHOD = "learned_and_or_control_anchor75_aux"
FORMAL_CAPACITIES = (64, 128, 256, 512)
PRIMARY_CAPACITIES = (128, 256, 512)
TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
)
EXTERNAL_TASKS = ("qasper", "multifieldqa_en")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-dir", required=True)
    parser.add_argument("--mistral-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    parser.add_argument("--permutation-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def holm_adjust(records: list[dict[str, Any]]) -> None:
    order = sorted(
        range(len(records)),
        key=lambda index: records[index]["permutation_p_two_sided"],
    )
    running = 0.0
    total = len(records)
    for rank, index in enumerate(order):
        adjusted = min(
            1.0,
            (total - rank)
            * records[index]["permutation_p_two_sided"],
        )
        running = max(running, adjusted)
        records[index]["holm_p_two_sided"] = running


def clustered_inference(
    clusters: dict[tuple[str, str, int], np.ndarray],
    bootstrap_replicates: int,
    permutation_replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not clusters:
        raise ValueError("no paired clusters")
    ordered = [clusters[key] for key in sorted(clusters)]
    observed = float(np.mean([values.mean() for values in ordered]))
    boot = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        boot[index] = np.mean(
            [
                values[
                    rng.integers(0, len(values), size=len(values))
                ].mean()
                for values in ordered
            ]
        )
    extreme = 0
    for _ in range(permutation_replicates):
        randomized = np.mean(
            [
                (
                    values
                    * rng.choice(
                        np.asarray((-1.0, 1.0)),
                        size=len(values),
                    )
                ).mean()
                for values in ordered
            ]
        )
        extreme += int(abs(randomized) >= abs(observed))
    return {
        "delta_f1_points": 100.0 * observed,
        "ci95_low": 100.0 * float(np.quantile(boot, 0.025)),
        "ci95_high": 100.0 * float(np.quantile(boot, 0.975)),
        "permutation_p_two_sided": (
            (extreme + 1) / (permutation_replicates + 1)
        ),
        "clusters": len(ordered),
        "paired_examples": sum(len(values) for values in ordered),
    }


def select_clusters(
    paired: dict[tuple[str, str, int], np.ndarray],
    tasks: Iterable[str],
) -> dict[tuple[str, str, int], np.ndarray]:
    task_set = set(tasks)
    return {
        key: values
        for key, values in paired.items()
        if key[1] in task_set and key[2] in PRIMARY_CAPACITIES
    }


def main() -> None:
    args = parse_args()
    run_dirs = {
        "qwen25": Path(args.qwen_dir),
        "mistral02": Path(args.mistral_dir),
    }
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    for model, run_dir in run_dirs.items():
        results_path = run_dir / "results.jsonl"
        manifest_path = run_dir / "run_manifest.json"
        metrics_path = run_dir / "metrics.json"
        for path in (results_path, manifest_path, metrics_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        model_rows = load_jsonl(results_path)
        for row in model_rows:
            row = dict(row)
            row["model_label"] = model
            rows.append(row)
        source_hashes[model] = {
            "results_sha256": sha256_file(results_path),
            "run_manifest_sha256": sha256_file(manifest_path),
            "metrics_sha256": sha256_file(metrics_path),
        }

    expected_methods = {ROOT_METHOD, PROOF_METHOD}
    observed_methods = {str(row["method"]) for row in rows}
    if observed_methods != expected_methods:
        raise ValueError(
            f"method coverage differs: {observed_methods}"
        )
    observed_tasks = {str(row["task"]) for row in rows}
    if observed_tasks != set(TASKS):
        raise ValueError(f"task coverage differs: {observed_tasks}")
    observed_capacities = {int(row["capacity"]) for row in rows}
    if observed_capacities != set(FORMAL_CAPACITIES):
        raise ValueError(
            f"capacity coverage differs: {observed_capacities}"
        )

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
        score = float(row["score"])
        retained = float(row["actual_retained_tokens"])
        if (
            not math.isfinite(score)
            or not math.isfinite(retained)
            or not 0.0 <= score <= 1.0
            or retained <= 0
        ):
            raise ValueError(f"invalid measurement: {key}")
        if retained > int(row["capacity"]) + 1e-6:
            raise ValueError(f"capacity exceeded: {key}")
        by_key[key] = row

    paired: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    cells: list[dict[str, Any]] = []
    retention_differences: list[float] = []
    for model in sorted(run_dirs):
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
            for capacity in FORMAL_CAPACITIES:
                root_rows = [
                    by_key[(model, task, source_id, capacity, ROOT_METHOD)]
                    for source_id in ids
                ]
                proof_rows = [
                    by_key[(model, task, source_id, capacity, PROOF_METHOD)]
                    for source_id in ids
                ]
                differences = np.asarray(
                    [
                        float(proof["score"]) - float(root["score"])
                        for root, proof in zip(root_rows, proof_rows)
                    ],
                    dtype=np.float64,
                )
                paired[(model, task, capacity)].extend(
                    differences.tolist()
                )
                retention_delta = [
                    float(proof["actual_retained_tokens"])
                    - float(root["actual_retained_tokens"])
                    for root, proof in zip(root_rows, proof_rows)
                ]
                retention_differences.extend(retention_delta)
                cells.append(
                    {
                        "model": model,
                        "task": task,
                        "capacity": capacity,
                        "n": len(ids),
                        "independent_root_f1": 100.0
                        * float(
                            np.mean(
                                [
                                    float(row["score"])
                                    for row in root_rows
                                ]
                            )
                        ),
                        "proofkv_f1": 100.0
                        * float(
                            np.mean(
                                [
                                    float(row["score"])
                                    for row in proof_rows
                                ]
                            )
                        ),
                        "delta_f1": 100.0 * float(differences.mean()),
                        "root_retained_tokens": float(
                            np.mean(
                                [
                                    float(row["actual_retained_tokens"])
                                    for row in root_rows
                                ]
                            )
                        ),
                        "proofkv_retained_tokens": float(
                            np.mean(
                                [
                                    float(row["actual_retained_tokens"])
                                    for row in proof_rows
                                ]
                            )
                        ),
                        "root_generation_seconds": float(
                            np.mean(
                                [
                                    float(row["generation_seconds"])
                                    for row in root_rows
                                ]
                            )
                        ),
                        "proofkv_generation_seconds": float(
                            np.mean(
                                [
                                    float(row["generation_seconds"])
                                    for row in proof_rows
                                ]
                            )
                        ),
                    }
                )
    paired_arrays = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in paired.items()
    }
    rng = np.random.default_rng(args.seed)
    endpoints = [
        {
            "endpoint": "all_tasks_same_scorer_contract",
            **clustered_inference(
                select_clusters(paired_arrays, TASKS),
                args.bootstrap_replicates,
                args.permutation_replicates,
                rng,
            ),
        },
        {
            "endpoint": "parser_free_external_tasks",
            **clustered_inference(
                select_clusters(paired_arrays, EXTERNAL_TASKS),
                args.bootstrap_replicates,
                args.permutation_replicates,
                rng,
            ),
        },
    ]
    holm_adjust(endpoints)
    for endpoint in endpoints:
        endpoint["passes_confirmatory_gate"] = (
            endpoint["delta_f1_points"] > 0.0
            and endpoint["ci95_low"] > 0.0
            and endpoint["holm_p_two_sided"] < 0.05
        )

    payload = {
        "status": (
            "PASS"
            if all(
                endpoint["passes_confirmatory_gate"]
                for endpoint in endpoints
            )
            else "NEGATIVE_OR_INCONCLUSIVE"
        ),
        "protocol": "fair_same_scorer_parser_free_external_v1",
        "formal_examples_per_model_task": 46,
        "methods": {
            "independent_span_control": ROOT_METHOD,
            "proofkv": PROOF_METHOD,
        },
        "training_data_boundary": {
            "scorer_training_families": [
                "hotpotqa",
                "2wikimqa",
                "musique",
                "controlled_topology",
            ],
            "parser_free_external_tasks": list(EXTERNAL_TASKS),
            "ruler_specific_adapter_used": False,
            "audit_amendment": (
                "2wikimqa removed from the external endpoint before any "
                "2wikimqa formal row was generated; it remains in the "
                "all-task endpoint"
            ),
        },
        "endpoints": endpoints,
        "retention_contract": {
            "same_capacity_cap": True,
            "mean_proof_minus_root_tokens": float(
                np.mean(retention_differences)
            ),
            "max_abs_proof_minus_root_tokens": float(
                np.max(np.abs(retention_differences))
            ),
        },
        "cells": cells,
        "source_hashes": source_hashes,
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
                "endpoints": endpoints,
                "out": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
