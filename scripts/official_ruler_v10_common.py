#!/usr/bin/env python
"""Shared, task-faithful helpers for the ProofKV-v10 official RULER runs."""

from __future__ import annotations

import re
from typing import Any, Sequence


TASKS = ("ruler_niah", "ruler_variable_tracking")
GENERATOR_TASK = {
    "ruler_niah": "niah",
    "ruler_variable_tracking": "variable_tracking",
}
MAX_NEW_TOKENS = {
    "ruler_niah": 32,
    "ruler_variable_tracking": 32,
}


def convert_official_row(task: str, row: dict[str, Any]) -> dict[str, Any]:
    """Split an official RULER prompt into compressed context and question tail.

    The question and answer prefix remain uncompressed, matching the LongBench
    physical-budget contract used by the v10 runner.
    """

    if task not in TASKS:
        raise ValueError(f"unsupported official RULER task: {task}")
    prompt = str(row["input"])
    if task == "ruler_niah":
        matches = list(re.finditer(r"\n(?=What )", prompt))
        if not matches:
            raise ValueError("official NIAH prompt has no terminal question")
        split_at = matches[-1].start()
        context = prompt[:split_at]
        question = prompt[split_at + 1 :]
    else:
        marker = "\nQuestion:"
        split_at = prompt.rfind(marker)
        if split_at < 0:
            raise ValueError(
                "official variable-tracking prompt has no terminal question"
            )
        context = prompt[:split_at]
        question = prompt[split_at + 1 :]
    answers = [str(value) for value in row.get("outputs", [])]
    if not context or not question or not answers:
        raise ValueError("official RULER row is missing context, question, or answers")
    answer_prefix = str(row.get("answer_prefix", ""))
    if not answer_prefix:
        raise ValueError("official RULER row is missing its answer prefix")
    return {
        "context": context,
        "input": question,
        "answers": answers,
        "answer_prefix": answer_prefix,
        "official_index": row.get("index"),
    }


def official_fields(
    task: str,
    row: dict[str, Any],
) -> tuple[str, str, str]:
    if task not in TASKS:
        raise ValueError(f"unsupported official RULER task: {task}")
    return (
        str(row["context"]),
        str(row["input"]),
        str(row["answer_prefix"]),
    )


def normalize_reference(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def score_prediction(prediction: str, refs: Sequence[str]) -> float:
    """Use the official mini-RULER reference-containment score."""

    normalized_prediction = normalize_reference(prediction)
    normalized_refs = [normalize_reference(value) for value in refs]
    if not normalized_refs or any(not value for value in normalized_refs):
        return 0.0
    return sum(
        1.0 if value in normalized_prediction else 0.0
        for value in normalized_refs
    ) / len(normalized_refs)
