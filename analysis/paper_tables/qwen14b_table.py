#!/usr/bin/env python3
"""Generate the frozen Qwen2.5-14B scale-transfer table.

This generator fails closed: the table is emitted only when the formal audit
is complete, both paired endpoints pass after Holm correction, and the
ProofKV/LLMLingua-2 retained-token difference is exactly zero.
"""

from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "paper" / "records" / "gpu_results_20260727" / "qwen14b_scale_20260727" / "formal_audit.json"
TABLE_DIR = Path(os.environ.get("PROOFKV_TABLE_DIR", ROOT / "paper" / "tables"))
OUTPUT = TABLE_DIR / "qwen14b_scale_transfer.tex"
SUMMARY = TABLE_DIR / "qwen14b_scale_generated_summary.json"
PROOF = "learned_and_or_control_anchor75_aux"
ROOT_RANKER = "learned_root_topk"
LL2 = "llmlingua2_official"
CAPACITIES = ("128", "256", "512")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def endpoint(
    payload: dict[str, Any], right: str
) -> dict[str, Any]:
    matches = [
        row
        for row in payload["endpoints"]
        if row["left"] == PROOF and row["right"] == right
    ]
    require(len(matches) == 1, f"missing unique ProofKV/{right} endpoint")
    return matches[0]


def format_row(
    label: str, values: dict[str, float], *, emphasize: bool = False
) -> str:
    scores = [float(values[capacity]) for capacity in CAPACITIES]
    rendered = [f"{score:.2f}" for score in scores]
    aggregate = f"{mean(scores):.2f}"
    if emphasize:
        rendered = [rf"\textbf{{{value}}}" for value in rendered]
        aggregate = rf"\textbf{{{aggregate}}}"
    cells = " & ".join(rendered)
    return f"{label} & {cells} & {aggregate} \\\\"


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    require(payload.get("status") == "PASS", "formal scale audit must PASS")
    require(payload.get("complete") is True, "formal scale audit must be complete")
    require(
        payload.get("protocol_binding", {}).get("status") == "PASS"
        and all(payload["protocol_binding"].get("checks", {}).values()),
        "model, manifest, implementation, and scorer bindings must pass",
    )
    require(
        payload.get("capacity_audit", {}).get("status") == "PASS"
        and int(payload["capacity_audit"].get("violations", -1)) == 0,
        "compact-capacity and physical-KV audit must pass",
    )
    require(
        payload.get("counts")
        == {
            "full": 230,
            PROOF: 690,
            ROOT_RANKER: 690,
            LL2: 690,
        },
        "formal scale row counts must match the frozen 46x5x3 design",
    )
    token_audit = payload["exact_token_difference"]
    require(
        int(token_audit["pairs"]) == 690
        and float(token_audit["mean"]) == 0.0
        and int(token_audit["max_abs"]) == 0,
        "ProofKV and LLMLingua-2 must match retained tokens exactly",
    )

    root_endpoint = endpoint(payload, ROOT_RANKER)
    ll2_endpoint = endpoint(payload, LL2)
    for row in (root_endpoint, ll2_endpoint):
        require(int(row["pairs"]) == 690, "each endpoint must contain 690 pairs")
        require(int(row["clusters"]) == 15, "each endpoint must contain 15 clusters")
        require(float(row["delta_f1"]) > 0.0, "endpoint delta must be positive")
        require(float(row["ci95"][0]) > 0.0, "endpoint interval must be positive")
        require(float(row["holm_p"]) < 0.05, "endpoint must pass Holm correction")

    values = payload["f1_by_method_capacity"]
    for method in (PROOF, ROOT_RANKER, LL2):
        require(
            set(values[method]) == set(CAPACITIES),
            f"{method} must contain all frozen capacities",
        )
    full_value = float(values["full"]["full"])
    full_values = {capacity: full_value for capacity in CAPACITIES}

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & 128 & 256 & 512 & Mean \\",
        r"\midrule",
        format_row("Full context", full_values),
        format_row("Official LL-2", values[LL2]),
        format_row("Independent Root", values[ROOT_RANKER]),
        format_row(r"\textbf{\proofkv}", values[PROOF], emphasize=True),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "status": "PASS",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "counts": payload["counts"],
        "protocol_binding": payload["protocol_binding"],
        "capacity_audit": payload["capacity_audit"],
        "exact_token_difference": token_audit,
        "endpoints": {
            "proof_minus_root": root_endpoint,
            "proof_minus_ll2": ll2_endpoint,
        },
        "f1_by_method_capacity": values,
        "proof_512_vs_full": payload["proof_512_vs_full"],
        "table": OUTPUT.name,
    }
    SUMMARY.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_sha256": summary["source_sha256"],
                "proof_minus_root": root_endpoint["delta_f1"],
                "proof_minus_ll2": ll2_endpoint["delta_f1"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
