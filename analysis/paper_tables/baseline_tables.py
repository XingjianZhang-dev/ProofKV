#!/usr/bin/env python3
"""Generate the targeted contract tables from immutable formal audits."""

from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RECORDS = REPO / "results" / "paper" / "records"
TABLES = Path(os.environ.get("PROOFKV_TABLE_DIR", REPO / "paper" / "tables"))
SOURCES = {
    "fair_contract_external_formal_audit.json": RECORDS / "gpu_results_20260726/fair_contract_external_20260726/formal_audit.json",
    "llmlingua2_contract_formal_audit.json": RECORDS / "gpu_results_20260726/llmlingua2_contract_20260726/formal_audit.json",
    "transparent_longbench_baselines.json": RECORDS / "gpu_results_20260726/transparent_longbench_baselines_20260726.json",
    "reuse_q_projection.json": REPO / "results" / "paper" / "reuse_q_projection.json",
}
EXPECTED_HASHES = {
    "fair_contract_external_formal_audit.json":
        "efb85257a174398f2102c76046ba47f0e599678ac1e75f872609933cdd5e6765",
    "llmlingua2_contract_formal_audit.json":
        "57dcca24d7d28e5252ae19b4da31e70da0732f57532e68240803663a6d4f880d",
    "transparent_longbench_baselines.json":
        "3518f0d7c0b61bd27fb70f4647c6158e235277db6aa23ebf3fb53ca4b1219007",
    "reuse_q_projection.json":
        "4b0badd1c22b567ad91cc18cfeda5984f409710edd99c89d7b9cd20b0e88f6e6",
}
METHODS = (
    "full",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
    "llmlingua2_official",
    "learned_root_topk",
    "learned_and_or_control_anchor75_aux",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str) -> dict:
    path = SOURCES[name]
    actual = sha256(path)
    if actual != EXPECTED_HASHES[name]:
        raise ValueError(f"{name}: expected {EXPECTED_HASHES[name]}, found {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError(f"{name}: formal status is not PASS")
    return payload


def transparent_table(payload: dict) -> str:
    if len(payload["rows"]) != 6:
        raise ValueError("transparent table must contain six model-capacity rows")
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Model & KV & Full & Stream & Snap & Pyr. & Chunk & LL-2 & Root & \proofkv \\",
        r"\midrule",
    ]
    for model_index, model in enumerate(("qwen25", "mistral02")):
        rows = [row for row in payload["rows"] if row["model"] == model]
        if [row["capacity"] for row in rows] != [128, 256, 512]:
            raise ValueError(f"{model}: capacity coverage differs")
        for row_index, row in enumerate(rows):
            scores = row["f1"]
            if set(scores) != set(METHODS):
                raise ValueError(f"{model}/{row['capacity']}: method coverage differs")
            model_name = (
                "Qwen2.5--7B" if model == "qwen25" else "Mistral--7B v0.2"
            )
            if row_index:
                model_name = ""
            compact_best = max(scores[method] for method in METHODS if method != "full")
            cells = []
            for method in METHODS:
                cell = f"{scores[method]:.1f}"
                if (
                    method == "learned_and_or_control_anchor75_aux"
                    and abs(scores[method] - compact_best) < 1e-9
                ):
                    cell = rf"\textbf{{{cell}}}"
                cells.append(cell)
            lines.append(
                f"{model_name} & {row['capacity']} & "
                + " & ".join(cells)
                + r" \\"
            )
        if model_index == 0:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def endpoint_table(fair: dict, llmlingua: dict) -> str:
    endpoints = {row["endpoint"]: row for row in fair["endpoints"]}
    same = endpoints["all_tasks_same_scorer_contract"]
    external = endpoints["parser_free_external_tasks"]
    ll2 = llmlingua["endpoint"]
    if not all(
        row["passes_confirmatory_gate"] and row["ci95_low"] > 0
        for row in (same, external, ll2)
    ):
        raise ValueError("at least one targeted endpoint failed")
    rows = (
        ("Same scorer: typed proof $-$ root top-$k$", same, True),
        ("Parser-free external transfer", external, True),
        ("Exact-token: \\proofkv $-$ LL-2", ll2, False),
    )
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Confirmatory endpoint & $\Delta$F1 & 95\% CI & Pairs & $p$ \\",
        r"\midrule",
    ]
    for label, row, holm in rows:
        p_value = row["holm_p_two_sided"] if holm else row["permutation_p_two_sided"]
        lines.append(
            f"{label} & {row['delta_f1_points']:.2f} & "
            f"$[{row['ci95_low']:.2f},{row['ci95_high']:.2f}]$ & "
            f"{row['paired_examples']:,} & ${p_value:.1e}$ \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def reuse_table(payload: dict) -> str:
    labels = (
        ("group1_selector_inclusive_8k_n16_repair1", 256, "8k / 256"),
        ("group1_selector_inclusive_8k_n16_repair1", 1024, "8k / 1024"),
        ("group1_selector_inclusive_32k_n4_out256_repair2", 256, "32k / 256"),
        ("group1_selector_inclusive_30k_n4_out1024_repair2", 1024, "30k / 1024"),
    )
    by_key = {
        (
            row["context_label"],
            row["output_tokens"],
            row["requests_per_document"],
        ): row
        for row in payload["rows"]
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Prompt / output & $Q{=}1$ & $Q{=}2$ & $Q{=}4$ & $Q{=}8$ \\",
        r"\midrule",
    ]
    for context, output, label in labels:
        values = [
            by_key[(context, output, q)]["full_over_proofkv_speedup"]
            for q in (1, 2, 4, 8)
        ]
        lines.append(
            f"{label} & " + " & ".join(f"{value:.2f}$\\times$" for value in values)
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def main() -> None:
    fair = load("fair_contract_external_formal_audit.json")
    llmlingua = load("llmlingua2_contract_formal_audit.json")
    transparent = load("transparent_longbench_baselines.json")
    reuse = load("reuse_q_projection.json")

    TABLES.mkdir(exist_ok=True)
    outputs = {
        "contract_transparent.tex": transparent_table(transparent),
        "contract_endpoints.tex": endpoint_table(fair, llmlingua),
        "reuse_q.tex": reuse_table(reuse),
    }
    for name, text in outputs.items():
        (TABLES / name).write_text(text, encoding="utf-8")

    summary = {
        "status": "PASS",
        "source_hashes": EXPECTED_HASHES,
        "generated_hashes": {
            name: sha256(TABLES / name) for name in outputs
        },
        "assertions": {
            "all_three_confirmatory_intervals_positive": True,
            "same_scorer_capacity_cap": fair["retention_contract"][
                "same_capacity_cap"
            ],
            "llmlingua2_exact_token_match": (
                llmlingua["retention_contract"][
                    "max_abs_proof_minus_llmlingua_tokens"
                ]
                == 0.0
            ),
            "transparent_rows": len(transparent["rows"]),
            "reuse_q1_matches_measured_cold": all(
                row["q1_matches_measured_cold"] is True
                for row in reuse["rows"]
                if row["requests_per_document"] == 1
            ),
        },
    }
    (TABLES / "targeted_20260726_generated_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
