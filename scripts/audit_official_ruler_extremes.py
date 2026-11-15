#!/usr/bin/env python3
"""Audit extreme low-budget official-RULER results without reusing model outcomes.

This audit is intentionally separate from the frozen formal audit.  It is meant
to make a suspicious-looking zero baseline reproducible: replay the official
metric, verify the hard KV budget, show that the full-context model solves the
same rows, and report recovery as the cache capacity increases.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from official_ruler_v10_common import score_prediction


PHYSICAL_METHODS = (
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run(value: str) -> tuple[str, str, Path]:
    label, separator, remainder = value.partition("=")
    candidate, separator_2, directory = remainder.partition("=")
    if (
        not separator
        or not separator_2
        or not label
        or not candidate
        or not directory
    ):
        raise ValueError(
            "--run must be LABEL=CANDIDATE_METHOD=/absolute/run/directory"
        )
    return label, candidate, Path(directory)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    keys = [
        (str(row["id"]), str(row["method"]), str(row["capacity"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate result keys in {path}")
    return rows


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def audit_run(
    label: str,
    candidate_method: str,
    run_dir: Path,
    task: str,
    capacity: int,
    minimum_samples: int,
    expect_all_physical_zero: bool,
    minimum_full_mean: float = 90.0,
) -> dict[str, Any]:
    results_path = run_dir / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    rows = read_rows(results_path)
    task_rows = [row for row in rows if str(row["task"]) == task]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        grouped[str(row["id"])].append(row)

    required = {
        ("full", "full"),
        (candidate_method, str(capacity)),
        *((method, str(capacity)) for method in PHYSICAL_METHODS),
    }
    complete: list[dict[tuple[str, str], dict[str, Any]]] = []
    for values in grouped.values():
        index = {
            (str(row["method"]), str(row["capacity"])): row
            for row in values
        }
        if required.issubset(index):
            complete.append(index)
    if len(complete) < minimum_samples:
        raise ValueError(
            f"{label}: only {len(complete)} complete samples; "
            f"{minimum_samples} required"
        )

    replay_errors = 0
    capacity_excess = 0
    raw_empty = 0
    scores: dict[str, list[float]] = {
        "full": [],
        candidate_method: [],
        **{method: [] for method in PHYSICAL_METHODS},
    }
    actual_tokens: dict[str, list[float]] = {
        candidate_method: [],
        **{method: [] for method in PHYSICAL_METHODS},
    }
    for index in complete:
        for method in scores:
            row = index[
                ("full", "full")
                if method == "full"
                else (method, str(capacity))
            ]
            prediction = str(row["prediction"])
            raw_empty += int(not prediction.strip())
            replayed = score_prediction(prediction, row["answers"])
            replay_errors += int(
                not math.isclose(
                    float(row["score"]),
                    replayed,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            )
            scores[method].append(100.0 * float(row["score"]))
            if method != "full":
                tokens = float(row["actual_retained_tokens"])
                actual_tokens[method].append(tokens)
                capacity_excess += int(tokens > capacity + 1.0e-6)

    if replay_errors or capacity_excess or raw_empty:
        raise ValueError(
            f"{label}: replay_errors={replay_errors}, "
            f"capacity_excess={capacity_excess}, empty={raw_empty}"
        )
    full_mean = mean(scores["full"])
    if (
        full_mean < minimum_full_mean
        or any(value <= 0.0 for value in scores["full"])
    ):
        raise ValueError(
            f"{label}: full-context sanity gate failed "
            f"({full_mean:.3f} < {minimum_full_mean:.3f} or a zero row)"
        )
    if expect_all_physical_zero and any(
        not math.isclose(value, 0.0, abs_tol=1.0e-12)
        for method in PHYSICAL_METHODS
        for value in scores[method]
    ):
        raise ValueError(f"{label}: a physical baseline is nonzero")
    if any(value <= 0.0 for value in scores[candidate_method]):
        raise ValueError(f"{label}: candidate failed an audited row")

    capacity_recovery = {}
    observed_capacities = sorted(
        {
            int(row["capacity"])
            for row in task_rows
            if isinstance(row.get("capacity"), int)
        }
    )
    for observed_capacity in observed_capacities:
        method_scores = {}
        for method in PHYSICAL_METHODS:
            values = [
                100.0 * float(row["score"])
                for row in task_rows
                if str(row["method"]) == method
                and row["capacity"] == observed_capacity
            ]
            if len(values) == len(grouped):
                method_scores[method] = mean(values)
        if method_scores:
            capacity_recovery[str(observed_capacity)] = method_scores

    return {
        "label": label,
        "candidate_method": candidate_method,
        "run_dir": str(run_dir.resolve()),
        "task": task,
        "capacity": capacity,
        "complete_samples": len(complete),
        "score_means": {
            method: mean(values) for method, values in scores.items()
        },
        "all_physical_predictions_zero": all(
            math.isclose(value, 0.0, abs_tol=1.0e-12)
            for method in PHYSICAL_METHODS
            for value in scores[method]
        ),
        "candidate_nonzero_on_every_sample": all(
            value > 0.0 for value in scores[candidate_method]
        ),
        "actual_retained_token_ranges": {
            method: [min(values), max(values)]
            for method, values in actual_tokens.items()
        },
        "capacity_recovery": capacity_recovery,
        "results_sha256": sha256_file(results_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=CANDIDATE_METHOD=/absolute/run/directory",
    )
    parser.add_argument("--task", default="ruler_variable_tracking")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--minimum-samples", type=int, default=10)
    parser.add_argument("--minimum-full-mean", type=float, default=90.0)
    parser.add_argument("--expect-all-physical-zero", action="store_true")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed = [parse_run(value) for value in args.run]
    labels = [label for label, _, _ in parsed]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate run label")
    runs = [
        audit_run(
            label,
            candidate,
            directory,
            args.task,
            args.capacity,
            args.minimum_samples,
            args.expect_all_physical_zero,
            args.minimum_full_mean,
        )
        for label, candidate, directory in parsed
    ]
    payload = {
        "status": "PASS",
        "protocol": "official_ruler_extreme_low_budget_audit_v1",
        "task": args.task,
        "capacity": args.capacity,
        "expect_all_physical_zero": args.expect_all_physical_zero,
        "physical_methods": list(PHYSICAL_METHODS),
        "runs": runs,
        "audit_implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "runs": len(runs),
                "samples": sum(run["complete_samples"] for run in runs),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
