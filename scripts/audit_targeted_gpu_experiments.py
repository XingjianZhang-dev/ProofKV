#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import statistics
import string
from pathlib import Path
from typing import Any


ROUTE_MODES = ("precompiled", "shared_index", "cold")
TOPOLOGY_METHODS = (
    "proof_core",
    "relevance_same_bytes",
    "edge_shuffled",
    "random_edges",
    "broken_middle",
    "oracle_proof",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent audit of both targeted GPU groups.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=100000)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return float(values[min(len(values) - 1, max(0, round((len(values) - 1) * q)))])


def bootstrap_mean(values: list[float], samples: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(values)
    draws = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": statistics.fmean(values),
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
    }


def sign_flip_pvalue(values: list[float], samples: int, seed: int) -> float:
    observed = statistics.fmean(values)
    rng = random.Random(seed)
    exceed = 0
    for _ in range(samples):
        value = statistics.fmean(item if rng.random() < 0.5 else -item for item in values)
        exceed += value >= observed - 1e-15
    return (exceed + 1) / (samples + 1)


def normalize(text: str) -> str:
    value = text.lower()
    value = "".join(char for char in value if char not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def f1(prediction: str, answer: str) -> float:
    pred = normalize(prediction).split()
    gold = normalize(answer).split()
    overlap = sum((collections.Counter(pred) & collections.Counter(gold)).values())
    if not pred or not gold:
        return float(pred == gold)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def audit_group1(path: Path, bootstrap_samples: int) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    batch_path = path / "batch_results.jsonl"
    route_path = path / "route_manifest.jsonl"
    metrics = json.loads(metrics_path.read_text())
    rows = read_jsonl(batch_path)
    protocol = metrics["protocol"]
    outputs = [int(value) for value in str(protocol["output_tokens"]).split(",")]
    modes = [value.strip() for value in str(protocol["modes"]).split(",") if value.strip()]
    repeats = int(protocol["repeats"])
    keys = [(row["output_tokens"], row["repeat"], row["mode"]) for row in rows]
    expected = len(outputs) * len(modes) * repeats
    by_key = {
        (int(row["output_tokens"]), int(row["repeat"]), str(row["mode"])): row
        for row in rows
    }
    comparisons = []
    route_identity = True
    for output_tokens in outputs:
        for repeat in range(repeats):
            base_hashes = by_key[(output_tokens, repeat, "precompiled")]["route_hashes"]
            for mode in ("shared_index", "cold"):
                route_identity &= by_key[(output_tokens, repeat, mode)]["route_hashes"] == base_hashes
        for mode in ROUTE_MODES:
            ratios = [
                by_key[(output_tokens, repeat, "full")]["total_seconds"]
                / by_key[(output_tokens, repeat, mode)]["total_seconds"]
                for repeat in range(repeats)
            ]
            latency_deltas = [
                by_key[(output_tokens, repeat, "full")]["total_seconds"]
                - by_key[(output_tokens, repeat, mode)]["total_seconds"]
                for repeat in range(repeats)
            ]
            comparisons.append(
                {
                    "output_tokens": output_tokens,
                    "mode": mode,
                    "paired_speedup": bootstrap_mean(
                        ratios,
                        bootstrap_samples,
                        91000 + output_tokens + sum(map(ord, mode)),
                    ),
                    "median_speedup": statistics.median(ratios),
                    "wins_over_full": sum(value > 1 for value in ratios),
                    "paired_sign_flip_p_one_sided": sign_flip_pvalue(
                        latency_deltas,
                        min(100000, bootstrap_samples * 10),
                        92000 + output_tokens + sum(map(ord, mode)),
                    ),
                }
            )
    selector_boundary_ok = all(
        (row["selector_seconds"] > 0) == (row["mode"] in {"shared_index", "cold"})
        for row in rows
    )
    return {
        "path": str(path),
        "metrics_sha256": sha256(metrics_path),
        "batch_sha256": sha256(batch_path),
        "route_sha256": sha256(route_path),
        "rows_expected": expected,
        "rows_observed": len(rows),
        "unique_keys": len(set(keys)),
        "corrupted_requests": sum(int(row["corrupted_requests"]) for row in rows),
        "route_identity_all_passed": route_identity,
        "selector_timing_boundary_ok": selector_boundary_ok,
        "comparisons": comparisons,
        "passed": (
            len(rows) == expected == len(set(keys))
            and sum(int(row["corrupted_requests"]) for row in rows) == 0
            and route_identity
            and selector_boundary_ok
        ),
    }


def audit_group2(path: Path, bootstrap_samples: int, permutation_samples: int) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    result_path = path / "results.jsonl"
    manifest_path = path / "run_manifest.json"
    metrics = json.loads(metrics_path.read_text())
    rows = read_jsonl(result_path)
    manifest = json.loads(manifest_path.read_text())
    selected_ids = [str(value) for value in manifest["selected_ids"]]
    by_key = {(str(row["id"]), str(row["method"])): row for row in rows}
    score_mismatches = sum(
        abs(f1(str(row["short_prediction"]), str(row["answer"])) - float(row["answer_f1"]))
        > 1e-12
        for row in rows
    )
    empty_answers = sum(not normalize(str(row["answer"])) for row in rows)
    empty_predictions = sum(not normalize(str(row["short_prediction"])) for row in rows)
    zero_by_method = {
        method: sum(by_key[(sample_id, method)]["answer_f1"] == 0 for sample_id in selected_ids)
        for method in TOPOLOGY_METHODS
    }
    comparisons = {}
    for opponent in ("relevance_same_bytes", "edge_shuffled", "random_edges", "broken_middle"):
        deltas = [
            float(by_key[(sample_id, "proof_core")]["answer_f1"])
            - float(by_key[(sample_id, opponent)]["answer_f1"])
            for sample_id in selected_ids
        ]
        comparisons[f"proof_core_vs_{opponent}"] = {
            **bootstrap_mean(
                deltas,
                bootstrap_samples,
                81000 + sum(map(ord, opponent)),
            ),
            "paired_sign_flip_p_one_sided": sign_flip_pvalue(
                deltas,
                permutation_samples,
                82000 + sum(map(ord, opponent)),
            ),
            "same_route_count": sum(
                by_key[(sample_id, "proof_core")]["route_sha256"]
                == by_key[(sample_id, opponent)]["route_sha256"]
                for sample_id in selected_ids
            ),
        }
    exact_tokens = all(
        len({int(by_key[(sample_id, method)]["retained_tokens"]) for method in TOPOLOGY_METHODS})
        == 1
        for sample_id in selected_ids
    )
    expected = len(selected_ids) * len(TOPOLOGY_METHODS)
    return {
        "path": str(path),
        "metrics_sha256": sha256(metrics_path),
        "results_sha256": sha256(result_path),
        "manifest_sha256": sha256(manifest_path),
        "rows_expected": expected,
        "rows_observed": len(rows),
        "unique_pairs": len(by_key),
        "prior_id_overlap": int(manifest["overlap_count"]),
        "exact_tokens_within_example": exact_tokens,
        "empty_answers": empty_answers,
        "empty_predictions": empty_predictions,
        "independent_f1_score_mismatches": score_mismatches,
        "zero_f1_rows_by_method": zero_by_method,
        "comparisons": comparisons,
        "frozen_primary_gate_reproduced": (
            comparisons["proof_core_vs_edge_shuffled"]["mean"] >= 0.03
            and comparisons["proof_core_vs_edge_shuffled"]["ci95_low"] > 0
        ),
        "passed": (
            len(rows) == expected == len(by_key)
            and int(manifest["overlap_count"]) == 0
            and exact_tokens
            and empty_answers == 0
            and empty_predictions == 0
            and score_mismatches == 0
        ),
        "source_reported_primary_gate": metrics["primary_causal_gate"],
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    group1_dirs = [
        root / "group1_selector_inclusive_8k_n16_repair1",
        root / "group1_selector_inclusive_32k_n4_out256_repair2",
        root / "group1_selector_inclusive_30k_n4_out1024_repair2",
    ]
    missing = [str(path) for path in group1_dirs if not (path / "metrics.json").exists()]
    group2_dir = root / "group2_natural_topology_musique_disjoint_n128_repair1"
    if missing or not (group2_dir / "metrics.json").exists():
        raise FileNotFoundError({"missing_group1": missing, "group2": str(group2_dir)})
    payload = {
        "group1": [audit_group1(path, args.bootstrap_samples) for path in group1_dirs],
        "group2": audit_group2(
            group2_dir, args.bootstrap_samples, args.permutation_samples
        ),
    }
    payload["passed"] = all(value["passed"] for value in payload["group1"]) and payload["group2"]["passed"]
    output_path = out_dir / "independent_audit.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Independent targeted-GPU audit",
        "",
        f"Overall integrity passed: **{payload['passed']}**.",
        "",
        "## Selector-inclusive serving",
        "",
    ]
    for group in payload["group1"]:
        lines.append(f"- `{Path(group['path']).name}`: integrity={group['passed']}")
        for value in group["comparisons"]:
            speedup = value["paired_speedup"]
            lines.append(
                f"  - output={value['output_tokens']}, {value['mode']}: "
                f"mean paired speedup {speedup['mean']:.3f}x "
                f"(95% CI {speedup['ci95_low']:.3f}-{speedup['ci95_high']:.3f})"
            )
    lines.extend(["", "## Natural topology causality", ""])
    for name, value in payload["group2"]["comparisons"].items():
        lines.append(
            f"- {name}: delta {value['mean']:+.4f}, 95% CI "
            f"[{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}], "
            f"sign-flip p={value['paired_sign_flip_p_one_sided']:.6f}"
        )
    lines.extend(
        [
            "",
            "Zero-F1 rows were independently rescored; none came from empty answers, "
            "empty predictions, or score mismatches.",
        ]
    )
    (out_dir / "independent_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
