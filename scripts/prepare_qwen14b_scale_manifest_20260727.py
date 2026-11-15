#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from freeze_longbench_structured_selection import official_fields, row_id
from run_longbench_structured_upgraded_v10 import format_parts, tokenize_parts


MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
MODEL_LABEL = "qwen25_14b"
CAPACITIES = [128, 256, 512]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind Qwen2.5-14B token counts to the frozen v10 selection."
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_manifest)
    source = json.loads(source_path.read_text())
    if int(source["max_samples_per_task"]) != 47:
        raise ValueError("expected the frozen n=47 complement selection")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        local_files_only=True,
    )
    payload: dict[str, Any] = json.loads(json.dumps(source))
    equal_to_qwen7 = True
    for task in payload["tasks"]:
        task_record = payload["selections"][task]
        data_path = Path(task_record["data_file"])
        if sha256_file(data_path) != task_record["data_sha256"]:
            raise ValueError(f"source data hash mismatch for {task}")
        rows = [
            json.loads(line)
            for line in data_path.read_text().splitlines()
            if line.strip()
        ]
        for selected in task_record["selected"]:
            raw = rows[int(selected["index"])]
            if row_id(task, int(selected["index"]), raw) != selected["id"]:
                raise ValueError(f"row identity drift for {selected['id']}")
            context, question, answer_prefix = official_fields(task, raw)
            formatted_context, question_tail = format_parts(
                tokenizer,
                context,
                question,
                answer_prefix,
            )
            _, offsets, _, question_tokens = tokenize_parts(
                tokenizer,
                formatted_context,
                question_tail,
            )
            counts = {
                "prompt": len(offsets),
                "question": question_tokens,
            }
            selected["token_counts"][MODEL_LABEL] = counts
            qwen7_counts = selected["token_counts"].get("qwen25")
            if qwen7_counts != counts:
                equal_to_qwen7 = False

    payload["protocol"] = "proofkv_qwen25_14b_scale_transfer_20260727"
    payload["capacities"] = CAPACITIES
    payload["tokenizers"][MODEL_LABEL] = str(Path(args.model).resolve())
    payload["scale_transfer"] = {
        "model_label": MODEL_LABEL,
        "model_revision": MODEL_REVISION,
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": sha256_file(source_path),
        "token_counts_equal_qwen25_7b": equal_to_qwen7,
        "formal_slice": {"start": 1, "count": 46},
        "frozen_before_predictions": True,
    }

    out = Path(args.out)
    if out.exists():
        if json.loads(out.read_text()) != payload:
            raise ValueError("existing scale-transfer manifest differs")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "out": str(out),
                "sha256": sha256_file(out),
                "tasks": payload["tasks"],
                "examples_per_task": payload["max_samples_per_task"],
                "token_counts_equal_qwen25_7b": equal_to_qwen7,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
