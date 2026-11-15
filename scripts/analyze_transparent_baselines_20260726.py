#!/usr/bin/env python
"""Build a fail-closed, reader-visible LongBench baseline comparison.

The original paper collapsed four named physical baselines into a development-
selected "Frozen physical" column.  This audit joins those historical formal
rows to the new same-scorer and LLMLingua2 runs on the exact 46 untouched IDs
used by the new protocols.  It emits both a machine-readable ledger and a
compact LaTeX table; no missing cell or post-hoc per-task baseline selection is
permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
)
CAPACITIES = (128, 256, 512)
MODELS = ("qwen25", "mistral02")
ROOT_METHOD = "learned_root_topk"
PROOF_METHOD = "learned_and_or_control_anchor75_aux"
LLMLINGUA_METHOD = "llmlingua2_official"
LEGACY_METHODS = (
    "full",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
)
DISPLAY_METHODS = (
    "full",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
    LLMLINGUA_METHOD,
    ROOT_METHOD,
    PROOF_METHOD,
)
DISPLAY_NAMES = {
    "full": "Full",
    "streamingllm": "Stream",
    "snapkv": "Snap",
    "pyramidkv": "Pyramid",
    "chunkkv_snapkv": "Chunk",
    LLMLINGUA_METHOD: "LLM2",
    ROOT_METHOD: "Root",
    PROOF_METHOD: r"\proofkv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for model in MODELS:
        parser.add_argument(f"--legacy-{model}-dir", required=True)
        parser.add_argument(f"--fair-{model}-dir", required=True)
        parser.add_argument(f"--llmlingua-{model}-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--table-out")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(run_dir: Path) -> dict[str, str]:
    paths = {
        "results": run_dir / "results.jsonl",
        "manifest": run_dir / "run_manifest.json",
        "metrics": run_dir / "metrics.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        f"{name}_sha256": sha256_file(path)
        for name, path in paths.items()
    }


def add_rows(
    destination: dict[tuple[str, str, str, str, str], dict[str, Any]],
    model: str,
    rows: list[dict[str, Any]],
    allowed_methods: set[str],
) -> None:
    for row in rows:
        method = str(row["method"])
        if method not in allowed_methods:
            continue
        capacity = str(row["capacity"])
        if method != "full" and int(capacity) not in CAPACITIES:
            continue
        if method == "full":
            capacity = "full"
        key = (
            model,
            str(row["task"]),
            str(row["id"]),
            capacity,
            method,
        )
        if key in destination:
            raise ValueError(f"duplicate row: {key}")
        score = float(row["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid score: {key}")
        if method != "full":
            retained = float(row["actual_retained_tokens"])
            if (
                not math.isfinite(retained)
                or retained <= 0
                or retained > int(capacity) + 1e-6
            ):
                raise ValueError(f"invalid retention: {key}")
        destination[key] = row


def latex_table(rows: list[dict[str, Any]]) -> str:
    header = (
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Model & KV & Full & Stream & Snap & Pyramid & Chunk & LLM2 & Root & \proofkv \\",
        r"\midrule",
    )
    lines = list(header)
    for model_index, model in enumerate(MODELS):
        model_rows = [row for row in rows if row["model"] == model]
        for row_index, row in enumerate(model_rows):
            label = (
                "Qwen2.5--7B"
                if model == "qwen25"
                else "Mistral--7B v0.2"
            )
            if row_index:
                label = ""
            compact = {
                method: row["f1"][method]
                for method in DISPLAY_METHODS
                if method != "full"
            }
            compact_best = max(compact.values())
            values = []
            for method in DISPLAY_METHODS:
                value = row["f1"][method]
                cell = f"{value:.1f}"
                if (
                    method == PROOF_METHOD
                    and abs(value - compact_best) < 1e-9
                ):
                    cell = rf"\textbf{{{cell}}}"
                values.append(cell)
            lines.append(
                f"{label} & {row['capacity']} & "
                + " & ".join(values)
                + r" \\"
            )
        if model_index + 1 < len(MODELS):
            lines.append(r"\midrule")
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    all_rows: dict[
        tuple[str, str, str, str, str],
        dict[str, Any],
    ] = {}
    hashes: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)

    for model in MODELS:
        directories = {
            "legacy": Path(getattr(args, f"legacy_{model}_dir")),
            "fair": Path(getattr(args, f"fair_{model}_dir")),
            "llmlingua2": Path(
                getattr(args, f"llmlingua_{model}_dir")
            ),
        }
        for label, run_dir in directories.items():
            hashes[model][label] = source_files(run_dir)
        add_rows(
            all_rows,
            model,
            load_jsonl(directories["legacy"] / "results.jsonl"),
            set(LEGACY_METHODS),
        )
        add_rows(
            all_rows,
            model,
            load_jsonl(directories["fair"] / "results.jsonl"),
            {ROOT_METHOD, PROOF_METHOD},
        )
        add_rows(
            all_rows,
            model,
            load_jsonl(directories["llmlingua2"] / "results.jsonl"),
            {LLMLINGUA_METHOD},
        )

    output_rows: list[dict[str, Any]] = []
    task_cells: list[dict[str, Any]] = []
    for model in MODELS:
        for task in TASKS:
            formal_ids = {
                key[2]
                for key in all_rows
                if key[0] == model
                and key[1] == task
                and key[3] in {str(value) for value in CAPACITIES}
                and key[4] == LLMLINGUA_METHOD
            }
            if len(formal_ids) != 46:
                raise ValueError(
                    f"{model}/{task}: expected 46 formal IDs, "
                    f"found {len(formal_ids)}"
                )
            for capacity in CAPACITIES:
                cell = {
                    "model": model,
                    "task": task,
                    "capacity": capacity,
                    "n": len(formal_ids),
                    "f1": {},
                    "mean_retained_tokens": {},
                }
                for method in DISPLAY_METHODS:
                    row_capacity = "full" if method == "full" else str(capacity)
                    selected = []
                    for source_id in sorted(formal_ids):
                        key = (
                            model,
                            task,
                            source_id,
                            row_capacity,
                            method,
                        )
                        if key not in all_rows:
                            raise ValueError(f"missing matched cell: {key}")
                        selected.append(all_rows[key])
                    cell["f1"][method] = 100.0 * sum(
                        float(row["score"]) for row in selected
                    ) / len(selected)
                    if method != "full":
                        cell["mean_retained_tokens"][method] = sum(
                            float(row["actual_retained_tokens"])
                            for row in selected
                        ) / len(selected)
                task_cells.append(cell)

        for capacity in CAPACITIES:
            cells = [
                cell
                for cell in task_cells
                if cell["model"] == model
                and cell["capacity"] == capacity
            ]
            if len(cells) != len(TASKS):
                raise ValueError(
                    f"{model}/{capacity}: incomplete task macro"
                )
            output_rows.append(
                {
                    "model": model,
                    "capacity": capacity,
                    "tasks": len(TASKS),
                    "examples_per_task": 46,
                    "f1": {
                        method: sum(
                            cell["f1"][method] for cell in cells
                        )
                        / len(cells)
                        for method in DISPLAY_METHODS
                    },
                }
            )

    payload = {
        "status": "PASS",
        "protocol": "transparent_matched_longbench_baselines_v1",
        "tasks": list(TASKS),
        "capacities": list(CAPACITIES),
        "examples_per_model_task": 46,
        "display_names": DISPLAY_NAMES,
        "rows": output_rows,
        "task_cells": task_cells,
        "source_hashes": hashes,
        "selection_policy": (
            "No development-selected aggregate column and no per-cell "
            "best-of baseline: every named method is displayed separately."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.table_out:
        table_out = Path(args.table_out)
        table_out.parent.mkdir(parents=True, exist_ok=True)
        table_out.write_text(latex_table(output_rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(output_rows),
                "out": str(out),
                "table_out": args.table_out,
            }
        )
    )


if __name__ == "__main__":
    main()
