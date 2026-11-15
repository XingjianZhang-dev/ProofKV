#!/usr/bin/env python
"""Generate and freeze a shared, outcome-blind official RULER v10 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import AutoTokenizer

from eval_official_ruler_mini import RULER_ROOT, run_generator
from freeze_longbench_structured_selection import (
    format_parts,
    row_id,
)
from official_ruler_v10_common import (
    GENERATOR_TASK,
    TASKS,
    convert_official_row,
    official_fields,
)


FROZEN_METHOD_SCHEMA = [
    "full",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
    "entity_graph_adaptive",
    "learned_topk_paths",
    "learned_and_or_aux",
]
RULER_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate official NVIDIA/RULER rows once and freeze a shared "
            "two-tokenizer ProofKV-v10 selection before evaluation."
        )
    )
    parser.add_argument("--generator-tokenizer", required=True)
    parser.add_argument("--tokenizers", required=True)
    parser.add_argument("--tokenizer-labels", required=True)
    parser.add_argument("--context-length", type=int, default=8000)
    parser.add_argument("--generated-per-task", type=int, default=64)
    parser.add_argument("--selected-per-task", type=int, default=47)
    parser.add_argument("--generator-seed", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--capacities", default="128,256,512,1024")
    parser.add_argument("--max-input-tokens", type=int, default=12000)
    parser.add_argument("--max-question-tokens", type=int, default=59)
    parser.add_argument("--split-label", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--out-manifest", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ruler_commit() -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={RULER_ROOT.resolve()}",
            "-C",
            str(RULER_ROOT),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def write_frozen_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"existing frozen data differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def main() -> None:
    args = parse_args()
    capacities = [
        int(value)
        for value in args.capacities.split(",")
        if value.strip()
    ]
    if (
        capacities != sorted(set(capacities))
        or len(capacities) < 3
        or any(value <= 4 for value in capacities)
    ):
        raise ValueError("capacities must be sorted, unique, and exceed four")
    if (
        args.generated_per_task < args.selected_per_task
        or args.selected_per_task < 1
        or args.max_question_tokens < 1
    ):
        raise ValueError("invalid generated/selected/question limits")
    tokenizer_paths = [
        value.strip() for value in args.tokenizers.split(",") if value.strip()
    ]
    tokenizer_labels = [
        value.strip()
        for value in args.tokenizer_labels.split(",")
        if value.strip()
    ]
    if (
        not tokenizer_paths
        or len(tokenizer_paths) != len(tokenizer_labels)
        or len(tokenizer_labels) != len(set(tokenizer_labels))
    ):
        raise ValueError("tokenizer paths and unique labels must align")
    observed_ruler_commit = ruler_commit()
    if observed_ruler_commit != RULER_COMMIT:
        raise ValueError(
            f"official RULER commit differs: {observed_ruler_commit}"
        )
    tokenizers = {
        label: AutoTokenizer.from_pretrained(path, use_fast=True)
        for label, path in zip(tokenizer_labels, tokenizer_paths)
    }
    out_root = Path(args.out_root)
    generated_root = out_root / "official_ruler_generated"
    converted_root = out_root / "converted"
    selections: dict[str, Any] = {}
    for task_number, task in enumerate(TASKS):
        generator_args = SimpleNamespace(
            model=args.generator_tokenizer,
            context_length=args.context_length,
            n_per_task=args.generated_per_task,
            seed=args.generator_seed + 1009 * task_number,
        )
        raw_path = run_generator(
            GENERATOR_TASK[task],
            generator_args,
            generated_root,
        )
        raw_rows = [
            json.loads(line)
            for line in raw_path.read_text().splitlines()
            if line.strip()
        ]
        converted = [
            convert_official_row(task, row)
            for row in raw_rows
        ]
        data_path = converted_root / f"{task}.jsonl"
        write_frozen_jsonl(data_path, converted)
        eligible = []
        for index, row in enumerate(converted):
            context, question, answer_prefix = official_fields(task, row)
            lengths = {}
            for label, tokenizer in tokenizers.items():
                formatted_context, question_tail = format_parts(
                    tokenizer,
                    context,
                    question,
                    answer_prefix,
                )
                context_ids = tokenizer(
                    formatted_context,
                    add_special_tokens=False,
                )["input_ids"]
                question_ids = tokenizer(
                    question_tail,
                    add_special_tokens=False,
                )["input_ids"]
                lengths[label] = {
                    "prompt": len(context_ids) + len(question_ids),
                    "question": len(question_ids),
                }
            if (
                max(record["prompt"] for record in lengths.values())
                <= args.max_input_tokens
                and max(record["question"] for record in lengths.values())
                <= args.max_question_tokens
            ):
                eligible.append(
                    {
                        "index": index,
                        "id": row_id(task, index, row),
                        "token_counts": lengths,
                    }
                )
        random.Random(
            args.selection_seed + 2003 * task_number
        ).shuffle(eligible)
        selected = eligible[: args.selected_per_task]
        if len(selected) != args.selected_per_task:
            raise ValueError(
                f"{task} has {len(eligible)} jointly eligible rows; "
                f"{args.selected_per_task} requested"
            )
        selections[task] = {
            "generator_task": GENERATOR_TASK[task],
            "generator_seed": args.generator_seed + 1009 * task_number,
            "raw_data_file": str(raw_path.resolve()),
            "raw_data_sha256": sha256_file(raw_path),
            "data_file": str(data_path.resolve()),
            "data_sha256": sha256_file(data_path),
            "total_rows": len(converted),
            "eligible_rows": len(eligible),
            "selected": selected,
        }
    payload = {
        "protocol": "proofkv_v10_official_ruler_outcome_blind_split_v1",
        "source_repo": "https://github.com/NVIDIA/RULER",
        "source_commit": observed_ruler_commit,
        "split_label": args.split_label,
        "selection_is_outcome_blind": True,
        "selection_fields": [
            "joint prompt length",
            "question-tail feasibility at the 64-token capacity",
            "task",
            "seeded order",
        ],
        "tasks": list(TASKS),
        "max_samples_per_task": args.selected_per_task,
        "max_input_tokens": args.max_input_tokens,
        "max_question_tokens": args.max_question_tokens,
        "generator_seed": args.generator_seed,
        "selection_seed": args.selection_seed,
        "tokenizers": dict(zip(tokenizer_labels, tokenizer_paths)),
        "capacities": capacities,
        "methods": FROZEN_METHOD_SCHEMA,
        "selections": selections,
    }
    out = Path(args.out_manifest)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out.exists():
        if out.read_text() != serialized:
            raise ValueError("existing official RULER split differs")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(serialized)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "split": args.split_label,
                "tasks": len(TASKS),
                "samples": len(TASKS) * args.selected_per_task,
                "manifest": str(out.resolve()),
                "manifest_sha256": sha256_file(out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
