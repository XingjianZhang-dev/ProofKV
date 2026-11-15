from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List, Sequence


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower()))).strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def contains_answer(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def first_choice_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None


def multiple_choice_accuracy(prediction: str, answer: str) -> float:
    pred = first_choice_letter(prediction)
    return float(pred == answer.strip().upper())
