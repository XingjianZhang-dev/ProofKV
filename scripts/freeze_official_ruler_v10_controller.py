#!/usr/bin/env python
"""Freeze official-RULER v10 routes from development runs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


RISK_FEATURES = [
    "overlay_confidence",
    "overlay_top_proof_probability",
    "overlay_proof_probability_margin",
    "overlay_root_top_probability",
    "overlay_root_probability_margin",
    "overlay_terminal_top_probability",
    "overlay_terminal_probability_margin",
    "overlay_num_proofs",
    "overlay_top_proof_spans",
    "proof_core_fraction",
    "coverage_auxiliary_fraction",
    "context_log_tokens",
    "question_budget_fraction",
]
BASELINES = [
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
]
CANDIDATES = [
    "learned_and_or_control_anchor25_aux",
    "learned_and_or_control_anchor50_aux",
    "learned_and_or_control_anchor75_aux",
    "ruler_chain_adapter_v10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose physical fallbacks and control-preserving v10 routes on "
            "official-RULER development data, then freeze before formal output."
        )
    )
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="MODEL_LABEL=/absolute/development/run/directory",
    )
    parser.add_argument(
        "--formal-output-dir",
        action="append",
        required=True,
        help="A formal run directory that must contain no result rows.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--one-sided-lower-quantile", type=float, default=0.10)
    parser.add_argument("--minimum-mean-delta", type=float, default=0.02)
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
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_runs(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        label, separator, directory = value.partition("=")
        if not separator or not label or not directory or label in output:
            raise ValueError(f"invalid or duplicate --run: {value}")
        output[label] = Path(directory)
    return output


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of no values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_lower(
    values: list[float],
    replicates: int,
    probability: float,
    seed: int,
) -> float:
    if not values or replicates < 1:
        raise ValueError("bootstrap requires values and positive replicates")
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        estimates.append(
            sum(values[rng.randrange(len(values))] for _ in values)
            / len(values)
        )
    return quantile(estimates, probability)


def constant_utility_model() -> dict[str, Any]:
    return {
        "feature_names": list(RISK_FEATURES),
        "feature_mean": [0.0] * len(RISK_FEATURES),
        "feature_scale": [1.0] * len(RISK_FEATURES),
        "coefficients": [0.0] * len(RISK_FEATURES),
        "intercept": 1.0,
        "training_examples": 0,
        "semantics": "constant-one prospective static route",
    }


def selected_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        str(row["id"])
        for task in manifest["tasks"]
        for row in manifest["selections"][task]["selected"]
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.one_sided_lower_quantile < 0.5:
        raise ValueError("one-sided lower quantile must be in (0, 0.5)")
    development_path = Path(args.development_manifest)
    formal_path = Path(args.formal_manifest)
    development = json.loads(development_path.read_text())
    formal = json.loads(formal_path.read_text())
    if development["tasks"] != formal["tasks"]:
        raise ValueError("development/formal official RULER tasks differ")
    if development["capacities"] != formal["capacities"]:
        raise ValueError("development/formal capacities differ")
    overlap = selected_ids(development) & selected_ids(formal)
    if overlap:
        raise ValueError(
            f"development/formal official RULER selections overlap: {len(overlap)}"
        )
    for value in args.formal_output_dir:
        results = Path(value) / "results.jsonl"
        if results.exists() and any(line.strip() for line in results.read_text().splitlines()):
            raise ValueError(
                f"formal output already exists before controller freeze: {results}"
            )
    runs = parse_runs(args.run)
    tasks = [str(value) for value in development["tasks"]]
    capacities = [int(value) for value in development["capacities"]]
    policies = {}
    development_summary = {}
    expected_per_task = int(development["max_samples_per_task"])
    for model_number, (model_label, run_dir) in enumerate(sorted(runs.items())):
        results_path = run_dir / "results.jsonl"
        rows = load_jsonl(results_path)
        keys = [
            (str(row["id"]), str(row["method"]), str(row["capacity"]))
            for row in rows
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate development rows: {run_dir}")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["id"])].append(row)
        expected_ids = selected_ids(development)
        if set(grouped) != expected_ids:
            raise ValueError(f"development IDs differ for {run_dir}")
        expected_configs = 1 + len(capacities) * (
            len(BASELINES) + len(CANDIDATES)
        )
        if any(len(values) != expected_configs for values in grouped.values()):
            raise ValueError(f"incomplete development matrix: {run_dir}")
        for task_number, task in enumerate(tasks):
            task_ids = sorted(
                row_id
                for row_id in grouped
                if row_id.startswith(f"{task}:")
            )
            if len(task_ids) != expected_per_task:
                raise ValueError(f"{run_dir}: incomplete {task} development IDs")
            for capacity_number, capacity in enumerate(capacities):
                score_by_method = {
                    method: [
                        float(
                            next(
                                row
                                for row in grouped[row_id]
                                if row["method"] == method
                                and str(row["capacity"]) == str(capacity)
                            )["score"]
                        )
                        for row_id in task_ids
                    ]
                    for method in BASELINES + CANDIDATES
                }
                fallback = max(
                    BASELINES,
                    key=lambda method: (
                        sum(score_by_method[method])
                        / len(score_by_method[method]),
                        -BASELINES.index(method),
                    ),
                )
                candidate_records = {}
                passing = []
                for candidate_number, candidate in enumerate(CANDIDATES):
                    deltas = [
                        candidate_score - fallback_score
                        for candidate_score, fallback_score in zip(
                            score_by_method[candidate],
                            score_by_method[fallback],
                        )
                    ]
                    mean = sum(deltas) / len(deltas)
                    lower = bootstrap_lower(
                        deltas,
                        args.bootstrap_replicates,
                        args.one_sided_lower_quantile,
                        args.seed
                        + 100000 * model_number
                        + 1000 * task_number
                        + 10 * capacity_number
                        + candidate_number,
                    )
                    record = {
                        "mean_delta": mean,
                        "bootstrap_lower": lower,
                        "samples": len(deltas),
                        "eligible": (
                            mean >= args.minimum_mean_delta
                            and lower >= 0.0
                        ),
                    }
                    candidate_records[candidate] = record
                    if record["eligible"]:
                        passing.append(candidate)
                selected_candidate = (
                    max(
                        passing,
                        key=lambda method: (
                            candidate_records[method]["mean_delta"],
                            candidate_records[method]["bootstrap_lower"],
                            -CANDIDATES.index(method),
                        ),
                    )
                    if passing
                    else None
                )
                key = f"{model_label}:{task}:{capacity}"
                policy = {
                    "fallback_method": fallback,
                    "eligible_candidate_methods": (
                        [selected_candidate]
                        if selected_candidate is not None
                        else []
                    ),
                    "candidate_models": (
                        {selected_candidate: constant_utility_model()}
                        if selected_candidate is not None
                        else {}
                    ),
                    "utility_margin": (
                        0.0
                        if selected_candidate is not None
                        else 1.0e9
                    ),
                    "upgraded_route": (
                        {
                            "route_type": "static_anchor",
                            "family": (
                                "ruler_chain_adapter"
                                if selected_candidate
                                == "ruler_chain_adapter_v10"
                                else "control_anchor"
                            ),
                            "method": selected_candidate,
                            "development_gate": candidate_records[
                                selected_candidate
                            ],
                        }
                        if selected_candidate is not None
                        else None
                    ),
                    "development_candidate_records": candidate_records,
                }
                policies[key] = policy
                development_summary[key] = {
                    "fallback_method": fallback,
                    "fallback_score": (
                        sum(score_by_method[fallback])
                        / len(score_by_method[fallback])
                    ),
                    "selected_candidate": selected_candidate,
                    "candidate_records": candidate_records,
                }
    payload = {
        "status": "PASS",
        "protocol": "proofkv_v10_official_ruler_development_frozen_controller_v1",
        "predecode_only": True,
        "domain_conditioned": True,
        "tasks": tasks,
        "capacities": capacities,
        "model_labels": sorted(runs),
        "feature_names": list(RISK_FEATURES),
        "candidate_methods": list(CANDIDATES),
        "final_policy_by_model_task_capacity": policies,
        "upgraded_architecture": {
            "status": "PASS",
            "predecode_only": True,
            "protocol": "control_preservation_only_v10",
            "external_policy": (
                "development-gated static control anchor, visible RULER "
                "chain adapter, or strongest development physical fallback"
            ),
        },
        "development_gate": {
            "bootstrap_replicates": args.bootstrap_replicates,
            "one_sided_lower_quantile": args.one_sided_lower_quantile,
            "minimum_mean_delta": args.minimum_mean_delta,
            "seed": args.seed,
        },
        "development_manifest": str(development_path.resolve()),
        "development_manifest_sha256": sha256_file(development_path),
        "formal_manifest": str(formal_path.resolve()),
        "formal_manifest_sha256": sha256_file(formal_path),
        "formal_outputs_unobserved_at_freeze": True,
        "formal_output_directories": [
            str(Path(value).resolve())
            for value in args.formal_output_dir
        ],
        "development_run_results_sha256": {
            label: sha256_file(directory / "results.jsonl")
            for label, directory in sorted(runs.items())
        },
        "development_summary": development_summary,
    }
    out = Path(args.out)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out.exists():
        if out.read_text() != serialized:
            raise ValueError("existing frozen RULER v10 controller differs")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(serialized)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "controller": str(out.resolve()),
                "controller_sha256": sha256_file(out),
                "policies": len(policies),
                "proof_policies": sum(
                    policy["upgraded_route"] is not None
                    for policy in policies.values()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
