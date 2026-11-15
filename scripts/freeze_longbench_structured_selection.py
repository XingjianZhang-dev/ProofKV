#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique")
CONTEXT_PREFIX = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question asconcisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the "
        "story asconcisely as you can, using a single phrase if possible. Do not provide any "
        "explanation.\n\n"
    ),
    "qasper": (
        'You are given a scientific article and a question. Answer the question as concisely as '
        'you can, using a single phrase or sentence if possible. If the question cannot be answered '
        'based on the information in the article, write "unanswerable". If the question is a yes/no '
        'question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
        "Article: {context}\n\n Answer the question based on the above article as concisely as you "
        'can, using a single phrase or sentence if possible. If the question cannot be answered '
        'based on the information in the article, write "unanswerable". If the question is a yes/no '
        'question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following "
        "question based on the above text, only give me the answer and do not output any other "
        "words.\n\n"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question "
        "based on the given passages. Only give me the answer and do not output any other words.\n\n"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question "
        "based on the given passages. Only give me the answer and do not output any other words.\n\n"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question "
        "based on the given passages. Only give me the answer and do not output any other words.\n\n"
    ),
}
QUESTION_TEMPLATE = {
    "narrativeqa": "Question: {input}\n\n",
    "qasper": "Question: {input}\n\n",
    "multifieldqa_en": "Question: {input}\n",
    "hotpotqa": "Question: {input}\n",
    "2wikimqa": "Question: {input}\n",
    "musique": "Question: {input}\n",
}
ANSWER_PREFIX = {
    "narrativeqa": "Answer:",
    "qasper": "Answer:",
    "multifieldqa_en": "Answer:",
    "hotpotqa": "Answer:",
    "2wikimqa": "Answer:",
    "musique": "Answer:",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a shared, non-truncated LongBench QA selection.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tokenizers", required=True, help="Comma-separated local tokenizer/model paths.")
    parser.add_argument("--tokenizer-labels", required=True, help="Comma-separated stable tokenizer labels.")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--max-samples-per-task", type=int, default=100)
    parser.add_argument("--max-input-tokens", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_id(task: str, index: int, row: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"context": row["context"], "input": row["input"], "answers": row["answers"]},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"{task}:{index}:{digest}"


def official_fields(task: str, row: dict[str, Any]) -> tuple[str, str, str]:
    context = CONTEXT_PREFIX[task].format(context=row["context"])
    question = QUESTION_TEMPLATE[task].format(input=row["input"])
    return context, question, ANSWER_PREFIX[task]


def format_parts(tokenizer: Any, context: str, question: str, answer_prefix: str) -> tuple[str, str]:
    if tokenizer.chat_template is None:
        formatted_context = str(getattr(tokenizer, "bos_token", "") or "") + context
        question_suffix = "\n"
    else:
        separator = "#" * (len(context) + 10)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        formatted_context, question_suffix = rendered.split(separator, 1)
    return formatted_context, question + question_suffix + answer_prefix


def main() -> None:
    args = parse_args()
    tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"unsupported tasks: {sorted(unknown)}")
    tokenizer_paths = [value.strip() for value in args.tokenizers.split(",") if value.strip()]
    tokenizer_labels = [value.strip() for value in args.tokenizer_labels.split(",") if value.strip()]
    if len(tokenizer_paths) != len(tokenizer_labels) or not tokenizer_paths:
        raise ValueError("tokenizer paths and labels must be non-empty and have equal length")
    tokenizers = {
        label: AutoTokenizer.from_pretrained(path, use_fast=True)
        for label, path in zip(tokenizer_labels, tokenizer_paths)
    }
    data_root = Path(args.data_root)
    selections: dict[str, Any] = {}
    for task_number, task in enumerate(tasks):
        data_path = data_root / f"{task}.jsonl"
        rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
        eligible = []
        for index, row in enumerate(rows):
            context, question, answer_prefix = official_fields(task, row)
            lengths = {
                label: {}
                for label in tokenizers
            }
            for label, tokenizer in tokenizers.items():
                formatted_context, question_tail = format_parts(tokenizer, context, question, answer_prefix)
                context_ids = tokenizer(
                    formatted_context, add_special_tokens=False
                )["input_ids"]
                question_ids = tokenizer(
                    question_tail, add_special_tokens=False
                )["input_ids"]
                lengths[label] = {
                    # KVPress pre-fills the context and then appends the
                    # uncompressed question.  Count the exact token sequence
                    # used by that interface rather than relying on joint
                    # tokenization across the context/question boundary.
                    "prompt": len(context_ids) + len(question_ids),
                    "question": len(question_ids),
                }
            if (
                max(value["prompt"] for value in lengths.values()) <= args.max_input_tokens
                and max(value["question"] for value in lengths.values()) <= 59
            ):
                eligible.append(
                    {
                        "index": index,
                        "id": row_id(task, index, row),
                        "token_counts": lengths,
                    }
                )
        rng = random.Random(args.seed + 1009 * task_number)
        rng.shuffle(eligible)
        selected = eligible[: args.max_samples_per_task]
        if len(selected) != args.max_samples_per_task:
            raise ValueError(
                f"{task} has only {len(eligible)} jointly non-truncated examples; "
                f"{args.max_samples_per_task} requested"
            )
        selections[task] = {
            "data_file": str(data_path),
            "data_sha256": sha256_file(data_path),
            "total_rows": len(rows),
            "eligible_rows": len(eligible),
            "selected": selected,
        }
    payload = {
        "protocol": "longbench_structured_qa_nontruncated_v3_paths_aux_physical",
        "selection_is_outcome_blind": True,
        "selection_fields": [
            "joint prompt length",
            "question-tail feasibility at the 64-token capacity",
            "task",
            "seeded order",
        ],
        "tasks": list(tasks),
        "max_samples_per_task": args.max_samples_per_task,
        "max_input_tokens": args.max_input_tokens,
        "seed": args.seed,
        "tokenizers": dict(zip(tokenizer_labels, tokenizer_paths)),
        "capacities": [64, 128, 256, 512],
        "methods": [
            "full",
            "streamingllm",
            "snapkv",
            "pyramidkv",
            "chunkkv_snapkv",
            "entity_graph_adaptive",
            "learned_topk_paths",
            "learned_and_or_aux",
        ],
        "compatibility_exclusions": [
            {
                "method": "adakv_snapkv",
                "reason": (
                    "prospective smoke retained the full physical DynamicCache because AdaKV "
                    "uses a head-specific attention mask rather than compacting cache tensors"
                ),
                "observed_contract": "requested 64 total-prefill tokens; measured 16589 KV-token equivalents",
            }
        ],
        "selections": selections,
    }
    out = Path(args.out)
    if out.exists():
        existing = json.loads(out.read_text())
        if existing != payload:
            raise ValueError("existing frozen selection differs from the requested protocol")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "frozen",
                "tasks": len(tasks),
                "samples": len(tasks) * args.max_samples_per_task,
                "max_input_tokens": args.max_input_tokens,
            }
        )
    )


if __name__ == "__main__":
    main()
