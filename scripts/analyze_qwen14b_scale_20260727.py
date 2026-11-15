#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = "learned_root_topk"
PROOF = "learned_and_or_control_anchor75_aux"
LL2 = "llmlingua2_official"
CAPACITIES = (128, 256, 512)
TASKS = ("qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique")
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
SELECTION_PROTOCOL = "proofkv_qwen25_14b_scale_transfer_20260727"
PROOF_METHODS = "full,learned_root_topk,learned_and_or_control_anchor75_aux"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the frozen Qwen2.5-14B scale-transfer endpoint."
    )
    parser.add_argument("--proof-results", required=True)
    parser.add_argument("--ll2-results")
    parser.add_argument("--selection-manifest")
    parser.add_argument("--proof-manifest")
    parser.add_argument("--ll2-manifest")
    parser.add_argument("--expected-per-task", type=int, default=46)
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def validate_protocol_binding(
    *,
    args: argparse.Namespace,
    proof_rows: list[dict[str, Any]],
    ll2_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    paths = {
        "selection": Path(args.selection_manifest)
        if args.selection_manifest
        else None,
        "proof": Path(args.proof_manifest) if args.proof_manifest else None,
        "ll2": Path(args.ll2_manifest) if args.ll2_manifest else None,
    }
    if not args.allow_incomplete and any(path is None for path in paths.values()):
        raise ValueError(
            "final audit requires selection, proof, and LL2 run manifests"
        )
    if paths["selection"] is None:
        return {"status": "NOT_REQUESTED"}, {}

    selection = json.loads(paths["selection"].read_text())
    proof_manifest = json.loads(paths["proof"].read_text())
    ll2_manifest = json.loads(paths["ll2"].read_text())
    selection_sha = sha256(paths["selection"])
    checks = {
        "selection_protocol": selection.get("protocol") == SELECTION_PROTOCOL,
        "selection_outcome_blind": selection.get("selection_is_outcome_blind")
        is True,
        "selection_disjoint_complement": selection.get(
            "selection_is_disjoint_complement"
        )
        is True,
        "selection_tasks": tuple(selection.get("tasks", ())) == TASKS,
        "selection_capacities": tuple(selection.get("capacities", ()))
        == CAPACITIES,
        "selection_tokenizer_parity": selection.get("scale_transfer", {}).get(
            "token_counts_equal_qwen25_7b"
        )
        is True,
        "proof_selection_sha": proof_manifest.get("selection_manifest_sha256")
        == selection_sha,
        "ll2_selection_sha": ll2_manifest.get("selection_manifest_sha256")
        == selection_sha,
        "proof_model_revision": proof_manifest.get("arguments", {}).get(
            "model_revision"
        )
        == MODEL_REVISION,
        "ll2_model_revision": ll2_manifest.get("arguments", {}).get(
            "model_revision"
        )
        == MODEL_REVISION,
        "proof_methods": proof_manifest.get("arguments", {}).get("methods")
        == PROOF_METHODS,
        "proof_capacities": proof_manifest.get("arguments", {}).get("capacities")
        == "128,256,512",
        "ll2_capacities": ll2_manifest.get("arguments", {}).get("capacities")
        == "128,256,512",
        "proof_implementation_hashes": len(
            proof_manifest.get("implementation_sha256", {})
        )
        >= 8,
        "proof_scorer_hashes": len(
            proof_manifest.get("scorer_checkpoint_sha256", {})
        )
        >= 6,
        "ll2_implementation_hashes": len(
            ll2_manifest.get("implementation_sha256", {})
        )
        >= 3,
        "ll2_compressor_hashes": len(
            ll2_manifest.get("compressor_model_sha256", {})
        )
        >= 7,
    }
    if not all(checks.values()):
        raise ValueError(
            "protocol binding failed: "
            + ", ".join(key for key, value in checks.items() if not value)
        )

    expected_ids: dict[str, set[str]] = {}
    expected = int(args.expected_per_task)
    for task in TASKS:
        selected = selection["selections"][task]["selected"]
        expected_ids[task] = {
            str(row["id"]) for row in selected[1 : 1 + expected]
        }
        if len(expected_ids[task]) != expected:
            raise ValueError(f"{task}: frozen formal ID set has wrong size")

    allowed = set().union(*expected_ids.values())
    for label, rows in (("proof", proof_rows), ("ll2", ll2_rows)):
        observed = {str(row["id"]) for row in rows}
        if not observed <= allowed:
            raise ValueError(f"{label}: rows outside frozen formal ID set")

    return (
        {
            "status": "PASS",
            "checks": checks,
            "selection_manifest_sha256": selection_sha,
            "proof_manifest_sha256": sha256(paths["proof"]),
            "ll2_manifest_sha256": sha256(paths["ll2"]),
            "model_revision": MODEL_REVISION,
        },
        expected_ids,
    )


def unique_map(
    rows: list[dict[str, Any]],
    methods: set[str],
) -> dict[tuple[str, str, int | str], dict[str, Any]]:
    output: dict[tuple[str, str, int | str], dict[str, Any]] = {}
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            continue
        capacity: int | str = (
            str(row["capacity"])
            if method == "full"
            else int(row["capacity"])
        )
        key = (str(row["id"]), method, capacity)
        if key in output:
            raise ValueError(f"duplicate row: {key}")
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite score: {key}")
        output[key] = row
    return output


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_endpoint(
    left: dict[tuple[str, str, int | str], dict[str, Any]],
    right: dict[tuple[str, str, int | str], dict[str, Any]],
    left_method: str,
    right_method: str,
    replicates: int,
) -> dict[str, Any]:
    clusters: dict[tuple[str, int], list[float]] = defaultdict(list)
    pairs = 0
    for task in TASKS:
        for capacity in CAPACITIES:
            left_rows = {
                key[0]: row
                for key, row in left.items()
                if key[1] == left_method
                and key[2] == capacity
                and row["task"] == task
            }
            right_rows = {
                key[0]: row
                for key, row in right.items()
                if key[1] == right_method
                and key[2] == capacity
                and row["task"] == task
            }
            shared = sorted(set(left_rows) & set(right_rows))
            if shared:
                clusters[(task, capacity)] = [
                    float(left_rows[row_id]["score"])
                    - float(right_rows[row_id]["score"])
                    for row_id in shared
                ]
                pairs += len(shared)
    if not clusters:
        raise ValueError(f"no paired rows for {left_method} and {right_method}")
    cluster_keys = sorted(clusters)
    point = mean(mean(clusters[key]) for key in cluster_keys)

    rng = random.Random(20260727)
    bootstrap = []
    for _ in range(replicates):
        cluster_means = []
        for key in cluster_keys:
            values = clusters[key]
            cluster_means.append(
                mean(values[rng.randrange(len(values))] for _ in values)
            )
        bootstrap.append(mean(cluster_means))

    cluster_means = [mean(clusters[key]) for key in cluster_keys]
    if len(cluster_means) <= 20:
        randomization = [
            mean(sign * value for sign, value in zip(signs, cluster_means))
            for signs in itertools.product((-1.0, 1.0), repeat=len(cluster_means))
        ]
    else:
        randomization = []
        for _ in range(replicates):
            randomization.append(
                mean(
                    (-1.0 if rng.random() < 0.5 else 1.0) * value
                    for value in cluster_means
                )
            )
    p_value = (
        sum(abs(value) >= abs(point) - 1e-15 for value in randomization) + 1
    ) / (len(randomization) + 1)
    return {
        "left": left_method,
        "right": right_method,
        "delta_f1": 100.0 * point,
        "ci95": [
            100.0 * percentile(bootstrap, 0.025),
            100.0 * percentile(bootstrap, 0.975),
        ],
        "p_value": p_value,
        "pairs": pairs,
        "clusters": len(cluster_keys),
        "cluster_sizes": {
            f"{task}:{capacity}": len(clusters[(task, capacity)])
            for task, capacity in cluster_keys
        },
    }


def holm_adjust(endpoints: list[dict[str, Any]]) -> None:
    ordered = sorted(
        enumerate(endpoints),
        key=lambda item: float(item[1]["p_value"]),
    )
    running = 0.0
    total = len(ordered)
    for rank, (index, endpoint) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(endpoint["p_value"]))
        running = max(running, adjusted)
        endpoints[index]["holm_p"] = running


def method_means(
    rows: dict[tuple[str, str, int | str], dict[str, Any]]
) -> dict[str, dict[str, float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (_, method, capacity), row in rows.items():
        label = str(capacity)
        values[(method, label)].append(float(row["score"]))
    return {
        method: {
            capacity: 100.0 * mean(scores)
            for (candidate, capacity), scores in sorted(values.items())
            if candidate == method
        }
        for method in sorted({method for method, _ in values})
    }


def main() -> None:
    args = parse_args()
    proof_rows = read_jsonl(Path(args.proof_results))
    ll2_rows = read_jsonl(Path(args.ll2_results)) if args.ll2_results else []
    protocol_binding, expected_ids = validate_protocol_binding(
        args=args,
        proof_rows=proof_rows,
        ll2_rows=ll2_rows,
    )
    proof_map = unique_map(proof_rows, {"full", ROOT, PROOF})
    ll2_map: dict[tuple[str, str, int | str], dict[str, Any]] = {}
    if ll2_rows:
        ll2_map = unique_map(ll2_rows, {LL2})

    capacity_violations = []
    for mapping in (proof_map, ll2_map):
        for (row_id, method, capacity), row in mapping.items():
            if isinstance(capacity, int) and not (
                0 < int(row["actual_retained_tokens"]) <= capacity
            ):
                capacity_violations.append(
                    {
                        "id": row_id,
                        "method": method,
                        "capacity": capacity,
                        "actual_retained_tokens": int(
                            row["actual_retained_tokens"]
                        ),
                    }
                )
            if int(row.get("actual_kv_bytes", 0)) <= 0:
                capacity_violations.append(
                    {
                        "id": row_id,
                        "method": method,
                        "capacity": capacity,
                        "actual_kv_bytes": row.get("actual_kv_bytes"),
                    }
                )
    if capacity_violations:
        raise ValueError(f"capacity/physical-KV violations: {capacity_violations[:3]}")

    root_endpoint = paired_endpoint(
        proof_map,
        proof_map,
        PROOF,
        ROOT,
        args.replicates,
    )
    endpoints = [root_endpoint]
    if ll2_map:
        endpoints.append(
            paired_endpoint(
                proof_map,
                ll2_map,
                PROOF,
                LL2,
                args.replicates,
            )
        )
    holm_adjust(endpoints)

    exact_token_differences = []
    if ll2_map:
        for key, proof_row in proof_map.items():
            row_id, method, capacity = key
            if method != PROOF or not isinstance(capacity, int):
                continue
            ll2_row = ll2_map.get((row_id, LL2, capacity))
            if ll2_row is None:
                continue
            exact_token_differences.append(
                int(proof_row["actual_retained_tokens"])
                - int(ll2_row["actual_retained_tokens"])
            )

    expected = int(args.expected_per_task)
    expected_compact = expected * len(TASKS) * len(CAPACITIES)
    expected_full = expected * len(TASKS)
    counts = defaultdict(int)
    for _, method, _ in proof_map:
        counts[method] += 1
    for _, method, _ in ll2_map:
        counts[method] += 1
    complete = (
        counts[PROOF] == expected_compact
        and counts[ROOT] == expected_compact
        and counts["full"] == expected_full
        and (not ll2_map or counts[LL2] == expected_compact)
    )
    if not complete and not args.allow_incomplete:
        raise ValueError(
            f"incomplete formal rows: {dict(counts)}; "
            f"expected compact={expected_compact}, full={expected_full}"
        )
    if expected_ids:
        for task in TASKS:
            task_expected = expected_ids[task]
            for method, capacity in (
                ("full", "full"),
                *(
                    (method, capacity)
                    for method in (PROOF, ROOT)
                    for capacity in CAPACITIES
                ),
                *(
                    ((LL2, capacity) for capacity in CAPACITIES)
                    if ll2_map
                    else ()
                ),
            ):
                mapping = ll2_map if method == LL2 else proof_map
                observed = {
                    row_id
                    for (row_id, candidate, candidate_capacity), row in mapping.items()
                    if candidate == method
                    and candidate_capacity == capacity
                    and row["task"] == task
                }
                if args.allow_incomplete:
                    if not observed <= task_expected:
                        raise ValueError(
                            f"{task}/{method}/{capacity}: unexpected formal IDs"
                        )
                elif observed != task_expected:
                    raise ValueError(
                        f"{task}/{method}/{capacity}: incomplete formal ID set"
                    )

    full_scores = {
        key[0]: float(row["score"])
        for key, row in proof_map.items()
        if key[1] == "full"
    }
    proof_512 = {
        key[0]: float(row["score"])
        for key, row in proof_map.items()
        if key[1] == PROOF and key[2] == 512
    }
    shared_full = sorted(set(full_scores) & set(proof_512))
    full_mean = mean(full_scores[row_id] for row_id in shared_full)
    proof_512_mean = mean(proof_512[row_id] for row_id in shared_full)
    endpoint_positive = all(
        endpoint["delta_f1"] > 0.0
        and endpoint["ci95"][0] > 0.0
        and endpoint["holm_p"] < 0.05
        for endpoint in endpoints
    )
    exact_tokens = (
        not exact_token_differences
        or max(map(abs, exact_token_differences)) == 0
    )
    status = (
        "PASS"
        if complete and endpoint_positive and exact_tokens and len(endpoints) == 2
        else "INTERIM_POSITIVE"
        if endpoint_positive and exact_tokens
        else "FAIL"
    )
    payload = {
        "status": status,
        "complete": complete,
        "protocol_binding": protocol_binding,
        "capacity_audit": {
            "status": "PASS",
            "violations": 0,
            "compact_rows_checked": sum(
                method != "full"
                for _, method, _ in itertools.chain(proof_map, ll2_map)
            ),
        },
        "counts": dict(counts),
        "expected_compact_rows_per_method": expected_compact,
        "expected_full_rows": expected_full,
        "endpoints": endpoints,
        "exact_token_difference": {
            "pairs": len(exact_token_differences),
            "mean": mean(exact_token_differences)
            if exact_token_differences
            else None,
            "max_abs": max(map(abs, exact_token_differences))
            if exact_token_differences
            else None,
        },
        "f1_by_method_capacity": {
            **method_means(proof_map),
            **method_means(ll2_map),
        },
        "proof_512_vs_full": {
            "pairs": len(shared_full),
            "proof_f1": 100.0 * proof_512_mean,
            "full_f1": 100.0 * full_mean,
            "absolute_delta": 100.0 * (proof_512_mean - full_mean),
            "retention_percent": (
                100.0 * proof_512_mean / full_mean
                if full_mean > 0.0
                else None
            ),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "complete": complete,
                "counts": dict(counts),
                "endpoints": endpoints,
                "proof_512_vs_full": payload["proof_512_vs_full"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
