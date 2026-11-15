#!/usr/bin/env python
"""Independent structural and semantic audit of formal official-RULER v10 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from official_ruler_v10_common import score_prediction


PRIMARY_METHODS = {
    "full",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
    "confirmation_stable_proofkv_v10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="MODEL_LABEL=/absolute/formal/run/directory",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_runs(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        label, separator, directory = value.partition("=")
        if not separator or not label or not directory or label in output:
            raise ValueError(f"invalid or duplicate --run: {value}")
        output[label] = Path(directory)
    return output


def selected_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        str(row["id"])
        for task in manifest["tasks"]
        for row in manifest["selections"][task]["selected"]
    }


def main() -> None:
    args = parse_args()
    selection_path = Path(args.selection_manifest)
    controller_path = Path(args.controller)
    selection = json.loads(selection_path.read_text())
    controller = json.loads(controller_path.read_text())
    if selection["tasks"] != controller["tasks"]:
        raise ValueError("selection/controller task domains differ")
    capacities = tuple(int(value) for value in selection["capacities"])
    if tuple(controller["capacities"]) != capacities:
        raise ValueError("formal official-RULER capacities differ")
    expected_ids = selected_ids(selection)
    expected_per_id = 1 + 5 * len(capacities)
    policies = controller["final_policy_by_model_task_capacity"]
    run_records = []
    total_routes = Counter()
    for model_label, run_dir in sorted(parse_runs(args.run).items()):
        results_path = run_dir / "results.jsonl"
        metrics_path = run_dir / "metrics.json"
        manifest_path = run_dir / "run_manifest.json"
        if not all(path.is_file() for path in [results_path, metrics_path, manifest_path]):
            raise FileNotFoundError(f"incomplete formal run: {run_dir}")
        rows = [
            json.loads(line)
            for line in results_path.read_text().splitlines()
            if line.strip()
        ]
        metrics = json.loads(metrics_path.read_text())
        run_manifest = json.loads(manifest_path.read_text())
        keys = [
            (str(row["id"]), str(row["method"]), str(row["capacity"]))
            for row in rows
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate formal rows: {run_dir}")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["id"])].append(row)
        if set(grouped) != expected_ids:
            raise ValueError(f"formal IDs differ: {run_dir}")
        if any(len(values) != expected_per_id for values in grouped.values()):
            raise ValueError(f"incomplete formal matrix: {run_dir}")
        if set(metrics["methods"]) != PRIMARY_METHODS:
            raise ValueError(f"formal primary methods differ: {run_dir}")
        if run_manifest["selection_manifest_sha256"] != sha256_file(selection_path):
            raise ValueError(f"selection hash differs: {run_dir}")
        if run_manifest["risk_controller_sha256"] != sha256_file(controller_path):
            raise ValueError(f"controller hash differs: {run_dir}")
        raw_empty = 0
        whitespace = 0
        capacity_excess = 0
        replay_errors = 0
        model_routes = Counter()
        for source_id, values in grouped.items():
            index = {
                (str(row["method"]), str(row["capacity"])): row
                for row in values
            }
            expected_configs = {("full", "full")} | {
                (method, str(capacity))
                for method in PRIMARY_METHODS - {"full"}
                for capacity in capacities
            }
            if set(index) != expected_configs:
                raise ValueError(f"formal config crossing differs: {source_id}")
            for row in values:
                prediction = row.get("prediction")
                if not isinstance(prediction, str):
                    raise ValueError(f"non-string prediction: {source_id}")
                raw_empty += int(prediction == "")
                whitespace += int(prediction != "" and not prediction.strip())
                observed = float(row["score"])
                replayed = score_prediction(prediction, row["answers"])
                replay_errors += int(
                    not math.isclose(
                        observed,
                        replayed,
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-9,
                    )
                )
                if str(row["capacity"]) != "full":
                    capacity_excess += int(
                        float(row["actual_retained_tokens"])
                        > int(row["capacity"]) + 1.0e-6
                    )
                    if not math.isclose(
                        float(row["seconds"]),
                        float(row["selection_seconds"])
                        + float(row["generation_seconds"]),
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-6,
                    ):
                        raise ValueError(f"timing decomposition differs: {source_id}")
            task = str(values[0]["task"])
            for capacity in capacities:
                policy = policies[f"{model_label}:{task}:{capacity}"]
                routed = index[
                    ("confirmation_stable_proofkv_v10", str(capacity))
                ]
                partition = routed["selection_partition"]
                route = str(partition["utility_route"])
                model_routes[route] += 1
                total_routes[route] += 1
                if route == "physical_fallback":
                    fallback = str(policy["fallback_method"])
                    source = index[(fallback, str(capacity))]
                    if (
                        policy["upgraded_route"] is not None
                        or partition["utility_fallback_method"] != fallback
                        or routed["prediction"] != source["prediction"]
                        or not math.isclose(
                            float(routed["score"]),
                            float(source["score"]),
                            abs_tol=1.0e-12,
                        )
                    ):
                        raise ValueError(f"fallback replay differs: {source_id}")
                elif route == "proof_coverage":
                    expected_method = policy["upgraded_route"]["method"]
                    if (
                        partition["utility_selected_method"]
                        != expected_method
                        or policy["eligible_candidate_methods"]
                        != [expected_method]
                    ):
                        raise ValueError(f"proof route differs: {source_id}")
                else:
                    raise ValueError(f"unknown formal route: {route}")
        if replay_errors or capacity_excess:
            raise ValueError(
                f"{run_dir}: replay_errors={replay_errors}, "
                f"capacity_excess={capacity_excess}"
            )
        run_records.append(
            {
                "model": model_label,
                "run_dir": str(run_dir.resolve()),
                "rows": len(rows),
                "samples": len(grouped),
                "routes": dict(model_routes),
                "raw_empty_predictions": raw_empty,
                "whitespace_only_predictions": whitespace,
                "implementation_files": len(
                    run_manifest["implementation_sha256"]
                ),
                "results_sha256": sha256_file(results_path),
                "metrics_sha256": sha256_file(metrics_path),
                "run_manifest_sha256": sha256_file(manifest_path),
            }
        )
    payload = {
        "status": "PASS",
        "protocol": "proofkv_v10_official_ruler_formal_audit_v1",
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256_file(selection_path),
        "controller": str(controller_path.resolve()),
        "controller_sha256": sha256_file(controller_path),
        "tasks": selection["tasks"],
        "capacities": list(capacities),
        "primary_methods": sorted(PRIMARY_METHODS),
        "routes": dict(total_routes),
        "runs": run_records,
        "audit_implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "models": len(run_records),
                "rows": sum(record["rows"] for record in run_records),
                "routes": dict(total_routes),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
