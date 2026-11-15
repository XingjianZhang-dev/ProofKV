#!/usr/bin/env python3
"""Fail-closed preflight for current-code official-baseline smoke runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from audit_official_longbench_v10_baselines import (
    CAPACITIES,
    CONFIGURATIONS,
    best_qa_f1,
    capacity_vector_sha256,
    load_jsonl,
    sha256_file,
    validate_auxiliary_hashes,
    validate_source_hashes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--qwen-run-dir", required=True)
    parser.add_argument("--mistral-run-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def audit_run(
    model: str,
    run_dir: Path,
    selection: dict[str, Any],
    selection_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    configuration = CONFIGURATIONS[model]
    method = str(configuration["method"])
    results_path = run_dir / "results.jsonl"
    metrics_path = run_dir / "pilot_metrics_n1.json"
    manifest_path = run_dir / "run_manifest.json"
    for path in (results_path, metrics_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = load_jsonl(results_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [str(task) for task in selection["tasks"]]
    expected_ids = {
        task: str(selection["selections"][task]["selected"][0]["id"])
        for task in tasks
    }
    expected_keys = {
        (task, expected_ids[task], capacity)
        for task in tasks
        for capacity in CAPACITIES
    }
    observed_keys: set[tuple[str, str, int]] = set()
    score_replays = 0
    nonzero_scores = 0
    runtime_capacity_hashes: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"{model} preflight row {index}"
        if row.get("method") != method:
            raise ValueError(f"{prefix}: wrong method")
        key = (str(row["task"]), str(row["id"]), int(row["capacity"]))
        if key in observed_keys or key not in expected_keys:
            raise ValueError(f"{prefix}: duplicate or unexpected key {key}")
        observed_keys.add(key)
        prediction = row.get("prediction")
        answers = row.get("answers")
        if not isinstance(prediction, str) or not prediction.strip():
            raise ValueError(f"{prefix}: empty or non-string prediction")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"{prefix}: answers are missing")
        score = float(row["score"])
        replayed = best_qa_f1(
            prediction,
            [str(answer) for answer in answers],
        )
        if not math.isclose(score, replayed, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{prefix}: QA-F1 replay differs")
        score_replays += 1
        nonzero_scores += int(score > 0)
        capacity = int(row["capacity"])
        retained = float(row["actual_retained_tokens"])
        prompt_tokens = int(row["prompt_tokens"])
        ratio = float(row["actual_retained_ratio"])
        kv_bytes = float(row["actual_kv_bytes"])
        values = (
            score,
            retained,
            ratio,
            kv_bytes,
            float(row["seconds"]),
            float(row["generation_seconds"]),
            float(row["selection_seconds"]),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{prefix}: non-finite value")
        if retained <= 0 or retained > capacity + 1e-6:
            raise ValueError(f"{prefix}: physical capacity violation")
        if prompt_tokens <= capacity:
            raise ValueError(f"{prefix}: compression is not exercised")
        if not math.isclose(
            ratio,
            retained / prompt_tokens,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{prefix}: retained ratio differs")
        expected_bytes = (
            retained * configuration["kv_bytes_per_native_token"]
        )
        if not math.isclose(
            kv_bytes,
            expected_bytes,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{prefix}: physical KV bytes differ")
        if not math.isclose(
            float(row["seconds"]),
            float(row["generation_seconds"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ) or not math.isclose(
            float(row["selection_seconds"]),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{prefix}: timing decomposition differs")
        if method == "official_compresskv":
            expected_capacities = metrics["capacity_configurations"][
                str(capacity)
            ]["layer_capacities"]
            expected_runtime_hash = capacity_vector_sha256(
                expected_capacities
            )
            if (
                int(row.get("runtime_capacity_cluster_count", -1))
                != len(expected_capacities)
                or row.get("runtime_layer_capacities_sha256")
                != expected_runtime_hash
            ):
                raise ValueError(
                    f"{prefix}: live CompressKV capacity state differs"
                )
            runtime_capacity_hashes.add(expected_runtime_hash)
    if observed_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError(f"{model}: incomplete preflight crossing")
    if nonzero_scores == 0:
        raise ValueError(f"{model}: all preflight scores are zero")
    if method == "official_compresskv" and len(runtime_capacity_hashes) != len(
        CAPACITIES
    ):
        raise ValueError(
            f"{model}: preflight runtime capacities are not distinct"
        )

    expected_metrics = {
        "status": "PASS",
        "protocol": "frozen_longbench_v10_model_compatible_official_baseline",
        "method": method,
        "model": model,
        "n_per_task": 1,
        "rows": len(expected_keys),
        "selection_manifest_sha256": selection_sha256,
        "official_commit": configuration["commit"],
        "results_sha256": sha256_file(results_path),
    }
    for field, expected in expected_metrics.items():
        if metrics.get(field) != expected:
            raise ValueError(
                f"{model}: metrics {field}={metrics.get(field)!r}, "
                f"expected {expected!r}"
            )
    if tuple(int(value) for value in metrics["capacities"]) != CAPACITIES:
        raise ValueError(f"{model}: metrics capacities differ")
    if float(metrics["max_capacity_excess_tokens"]) > 1e-6:
        raise ValueError(f"{model}: metrics report a capacity violation")

    arguments = manifest["arguments"]
    if arguments["method"] != method or arguments["model_label"] != model:
        raise ValueError(f"{model}: manifest method/model differs")
    if manifest["selection_manifest_sha256"] != selection_sha256:
        raise ValueError(f"{model}: manifest selection hash differs")
    if manifest["official_commit"] != configuration["commit"]:
        raise ValueError(f"{model}: manifest commit differs")
    runner_hashes = [
        value
        for path, value in manifest["implementation_sha256"].items()
        if str(path).endswith("/run_official_longbench_v10_baseline.py")
    ]
    if runner_hashes != [runner_sha256]:
        raise ValueError(f"{model}: smoke did not use the current runner")
    packages = manifest["environment"]["packages"]
    for package, expected in configuration["packages"].items():
        if packages.get(package) != expected:
            raise ValueError(
                f"{model}: package {package}={packages.get(package)!r}, "
                f"expected {expected!r}"
            )
    official_repo = Path(arguments["official_repo"])
    validate_source_hashes(
        official_repo,
        method,
        manifest["official_source_sha256"],
    )
    validate_auxiliary_hashes(
        official_repo,
        manifest["official_auxiliary_sha256"],
    )
    return {
        "status": "PASS",
        "model": model,
        "method": method,
        "rows": len(rows),
        "nonzero_scores": nonzero_scores,
        "score_replays": score_replays,
        "runtime_capacity_hashes": sorted(runtime_capacity_hashes),
        "max_capacity_excess_tokens": max(
            float(row["actual_retained_tokens"]) - int(row["capacity"])
            for row in rows
        ),
        "results_sha256": sha256_file(results_path),
        "metrics_sha256": sha256_file(metrics_path),
        "manifest_sha256": sha256_file(manifest_path),
        "official_commit": configuration["commit"],
    }


def main() -> None:
    args = parse_args()
    selection_path = Path(args.selection_manifest)
    runner_path = Path(args.runner)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_sha256 = sha256_file(selection_path)
    runner_sha256 = sha256_file(runner_path)
    runs = [
        audit_run(
            model,
            run_dir,
            selection,
            selection_sha256,
            runner_sha256,
        )
        for model, run_dir in (
            ("qwen25", Path(args.qwen_run_dir)),
            ("mistral02", Path(args.mistral_run_dir)),
        )
    ]
    payload = {
        "status": "PASS",
        "protocol": "official_longbench_v10_current_code_smoke_preflight_v1",
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": selection_sha256,
        "runner": str(runner_path.resolve()),
        "runner_implementation_sha256": runner_sha256,
        "audit_implementation_sha256": sha256_file(Path(__file__).resolve()),
        "runs": runs,
        "checks": [
            "current runner hash equals each runtime manifest",
            "official commits, sources, extensions, and package versions",
            "exact five-task x four-capacity smoke crossing",
            "independent QA-F1 replay and nonzero sanity",
            "physical retained-token, KV-byte, and timing consistency",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "runs": len(runs),
                "rows": sum(run["rows"] for run in runs),
                "out": str(out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
