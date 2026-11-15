#!/usr/bin/env python
"""Apply the frozen point-estimate gate to the excluded confirmation slice."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from audit_longbench_structured_common import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def confirmation_summaries(
    statistics: dict[str, Any],
) -> dict[str, Any]:
    if statistics.get("analysis_role") != "confirmation":
        raise ValueError("statistics are not labeled as confirmation")
    if statistics.get("excluded_from_final_effect_estimates") is not True:
        raise ValueError("confirmation is not excluded from final estimates")
    model_deltas = {}
    capacity_values: dict[str, list[float]] = defaultdict(list)
    task_values: dict[str, list[float]] = defaultdict(list)
    proof_routes = 0
    utility_rows = 0
    for run in statistics["runs"]:
        if (
            int(run["selection_start_per_task"]) != 28
            or int(run["samples_per_task"]) != 10
        ):
            raise ValueError("statistics use the wrong confirmation slice")
        capacity_deltas = []
        for record in run["capacities"]:
            capacity = str(record["capacity"])
            delta = float(record["utility_delta_vs_fallback"])
            capacity_deltas.append(delta)
            capacity_values[capacity].append(delta)
            proof_routes += int(record["proof_routes"])
            utility_rows += int(record["utility_rows"])
            for task_record in record["task_breakdown"]:
                task_values[str(task_record["task"])].append(
                    float(task_record["utility_delta_vs_fallback"])
                )
        model_deltas[str(run["model"])] = mean(capacity_deltas)
    capacity_deltas = {
        capacity: mean(values)
        for capacity, values in sorted(capacity_values.items())
    }
    task_deltas = {
        task: mean(values) for task, values in sorted(task_values.items())
    }
    global_from_tasks = mean(task_deltas.values())
    global_reported = float(
        statistics["primary_inference"]["paired_task_macro_delta"]
    )
    if not math.isclose(
        global_from_tasks,
        global_reported,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("global primary delta differs from task reconstruction")
    return {
        "global_task_macro_delta_f1": global_reported,
        "model_macro_delta_f1": model_deltas,
        "capacity_macro_delta_f1": capacity_deltas,
        "task_macro_delta_f1": task_deltas,
        "proof_routes": proof_routes,
        "utility_rows": utility_rows,
        "overall_proof_route_coverage": (
            proof_routes / utility_rows if utility_rows else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol)
    audit_path = Path(args.audit)
    statistics_path = Path(args.statistics)
    protocol = json.loads(protocol_path.read_text())
    audit = json.loads(audit_path.read_text())
    statistics = json.loads(statistics_path.read_text())
    if (
        protocol.get("protocol")
        != "longbench_architecture_confirmation_acceptance_gate_v1"
    ):
        raise ValueError("unexpected confirmation protocol")
    requirements = protocol["requirements"]
    summaries = confirmation_summaries(statistics)
    checks = {
        "independent_audit_status": (
            audit.get("status") == requirements["independent_audit_status"]
        ),
        "global_task_macro_delta": (
            summaries["global_task_macro_delta_f1"]
            >= float(requirements["minimum_global_task_macro_delta_f1"])
        ),
        "each_model_macro_delta": all(
            value
            >= float(requirements["minimum_each_model_macro_delta_f1"])
            for value in summaries["model_macro_delta_f1"].values()
        ),
        "each_capacity_macro_delta": all(
            value
            >= float(requirements["minimum_each_capacity_macro_delta_f1"])
            for value in summaries["capacity_macro_delta_f1"].values()
        ),
        "each_task_macro_delta": all(
            value
            >= float(requirements["minimum_each_task_macro_delta_f1"])
            for value in summaries["task_macro_delta_f1"].values()
        ),
        "overall_proof_route_coverage": (
            summaries["overall_proof_route_coverage"]
            >= float(requirements["minimum_overall_proof_route_coverage"])
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": protocol["protocol"],
        "protocol_sha256": sha256_file(protocol_path),
        "audit_sha256": sha256_file(audit_path),
        "statistics_sha256": sha256_file(statistics_path),
        "requirements": requirements,
        "checks": checks,
        "summaries": summaries,
        "failure_action": protocol["failure_action"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, separators=(",", ":")))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
