#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_fair_contract_external_20260726 import sha256_file


REQUEST_COUNTS = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def projected_reuse_rows(
    payload: dict[str, Any],
    source: str,
) -> list[dict[str, Any]]:
    records = {
        (int(row["output_tokens"]), str(row["mode"])): row
        for row in payload["records"]
    }
    output_lengths = sorted(
        {
            int(row["output_tokens"])
            for row in payload["records"]
        }
    )
    rows = []
    for output_tokens in output_lengths:
        full = records[(output_tokens, "full")]
        shared = records[(output_tokens, "shared_index")]
        cold = records[(output_tokens, "cold")]
        full_seconds = float(full["total_seconds_p50"])
        shared_seconds = float(shared["total_seconds_p50"])
        cold_seconds = float(cold["total_seconds_p50"])
        index_once = cold_seconds - shared_seconds
        if index_once < 0:
            raise ValueError("cold latency is below shared-index latency")
        for requests in REQUEST_COUNTS:
            full_total = requests * full_seconds
            proof_total = index_once + requests * shared_seconds
            rows.append(
                {
                    "source": source,
                    "samples_per_batch": int(
                        payload["protocol"]["samples"]
                    ),
                    "context_label": Path(source).parent.name,
                    "output_tokens": output_tokens,
                    "requests_per_document": requests,
                    "full_total_seconds": full_total,
                    "proofkv_total_seconds": proof_total,
                    "full_over_proofkv_speedup": (
                        full_total / proof_total
                    ),
                    "measured_index_once_seconds": index_once,
                    "measured_full_per_request_batch_seconds": (
                        full_seconds
                    ),
                    "measured_shared_per_request_batch_seconds": (
                        shared_seconds
                    ),
                    "q1_matches_measured_cold": (
                        abs(proof_total - cold_seconds) < 1e-9
                        if requests == 1
                        else None
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    hashes = {}
    for value in args.metrics:
        path = Path(value)
        payload = json.loads(path.read_text())
        if not payload.get("passed_integrity"):
            raise ValueError(f"serving integrity failed: {path}")
        if not payload.get("route_identity_all_passed"):
            raise ValueError(f"route identity failed: {path}")
        rows.extend(projected_reuse_rows(payload, str(path)))
        hashes[str(path)] = sha256_file(path)
    if not all(
        row["full_over_proofkv_speedup"] > 1.0
        for row in rows
        if row["requests_per_document"] >= 2
    ):
        raise ValueError("a Q>=2 serving regime is not faster")
    payload = {
        "status": "PASS",
        "protocol": "measured_component_reuse_amortization_v1",
        "request_counts": list(REQUEST_COUNTS),
        "interpretation": (
            "Q=1 reproduces measured cold-index total latency; Q>1 "
            "amortizes the same measured index component over repeated "
            "queries to the same document batch"
        ),
        "rows": rows,
        "source_hashes": hashes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "rows": len(rows)}))


if __name__ == "__main__":
    main()
