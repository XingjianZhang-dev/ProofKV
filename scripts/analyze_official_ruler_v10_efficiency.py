#!/usr/bin/env python3
"""Specialize the frozen v10 efficiency audit to RULER capacities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import analyze_longbench_v10_efficiency as base


RULER_CAPACITIES = (128, 256, 512, 1024)
ROWS_PER_FORMAL_RUN = 1_974


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--utility-method", default=base.UTILITY_METHOD)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def specialize_payload(
    payload: dict[str, Any],
    wrapper_path: Path,
    base_path: Path,
) -> dict[str, Any]:
    if (
        payload.get("status") != "PASS"
        or payload.get("protocol") != "proofkv_v10_physical_efficiency_audit_v1"
    ):
        raise ValueError("base efficiency audit did not pass")
    reports = payload.get("input_runs", [])
    if not reports:
        raise ValueError("base efficiency audit contains no runs")
    for report in reports:
        if tuple(int(value) for value in report["capacities"]) != RULER_CAPACITIES:
            raise ValueError("RULER efficiency capacities differ")
        if int(report["rows"]) != ROWS_PER_FORMAL_RUN:
            raise ValueError("RULER efficiency row count differs")
    if int(payload["validation"]["rows"]) != ROWS_PER_FORMAL_RUN * len(reports):
        raise ValueError("RULER efficiency total rows differ")
    base_implementation_sha256 = payload["analysis_implementation_sha256"]
    if base_implementation_sha256 != base.sha256_file(base_path):
        raise ValueError("base efficiency implementation hash differs")
    payload["base_protocol"] = payload["protocol"]
    payload["protocol"] = (
        "proofkv_v10_official_ruler_physical_efficiency_audit_v1"
    )
    payload["base_analysis_implementation_sha256"] = (
        base_implementation_sha256
    )
    payload["analysis_implementation_sha256"] = base.sha256_file(wrapper_path)
    payload["base_analyzer"] = str(base_path.resolve())
    payload["ruler_capacities"] = list(RULER_CAPACITIES)
    payload["specialization"] = (
        "The frozen generic analyzer is executed unchanged after setting its "
        "capacity contract to the prospectively frozen official-RULER "
        "128/256/512/1024 total-KV capacities."
    )
    return payload


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    temporary = out.with_suffix(out.suffix + ".base.tmp")
    if temporary.exists():
        temporary.unlink()
    old_capacities = base.CAPACITIES
    old_argv = sys.argv
    base_path = Path(base.__file__).resolve()
    wrapper_path = Path(__file__).resolve()
    try:
        base.CAPACITIES = RULER_CAPACITIES
        sys.argv = [
            str(base_path),
            *[
                value
                for run_dir in args.run_dir
                for value in ("--run-dir", run_dir)
            ],
            "--utility-method",
            args.utility_method,
            "--out",
            str(temporary),
        ]
        base.main()
    finally:
        base.CAPACITIES = old_capacities
        sys.argv = old_argv
    payload = specialize_payload(
        json.loads(temporary.read_text(encoding="utf-8")),
        wrapper_path,
        base_path,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.unlink()
    print(
        json.dumps(
            {
                "status": "PASS",
                "runs": len(payload["input_runs"]),
                "rows": payload["validation"]["rows"],
                "out": str(out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
