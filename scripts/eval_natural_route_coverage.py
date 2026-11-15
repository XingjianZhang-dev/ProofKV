#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import load_dataset
from transformers import AutoTokenizer

from proofkv.prefill_selectors import (
    GuidepostPlan,
    _plan_from_spans,
    block_halo_guideposts,
    compass_prefill_guideposts,
)


SINK = 4
RECENT_FRACTION = 512 / 636
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "had", "has", "have", "he", "her", "his", "how", "in", "is", "it", "its", "of", "on",
    "or", "she", "that", "the", "their", "them", "they", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold-evidence coverage gate for prompt-visible route selectors.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", choices=["hotpotqa"], default="hotpotqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--budgets", default="0.04,0.08,0.16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
        if token.lower() not in STOPWORDS
    }


def token_span(offsets: Sequence[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    hits = [idx for idx, (start, end) in enumerate(offsets) if end > char_start and start < char_end and end > start]
    if not hits:
        return 0, 0
    return min(hits), max(hits) + 1


def build_hotpot_prompt(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    support = set(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]))
    pieces: list[str] = []
    sentences: list[dict[str, Any]] = []
    cursor = 0
    for title, page_sentences in zip(row["context"]["title"], row["context"]["sentences"]):
        for sent_id, sentence in enumerate(page_sentences):
            line = f"[{title}] {sentence.strip()}\n"
            start = cursor
            end = cursor + len(line)
            pieces.append(line)
            sentences.append(
                {
                    "title": title,
                    "sent_id": sent_id,
                    "text": sentence.strip(),
                    "char_span": (start, end),
                    "gold": (title, sent_id) in support,
                }
            )
            cursor = end
    query = f"\nQuestion: {row['question']}\nAnswer:"
    pieces.append(query)
    return "".join(pieces), sentences, query


def budget_parts(seq_len: int, ratio: float) -> tuple[int, int, int]:
    retained_total = max(SINK + 1, int(round(seq_len * ratio)))
    remaining = max(1, retained_total - SINK)
    recent = max(1, min(remaining, int(round(remaining * RECENT_FRACTION))))
    return retained_total, recent, max(0, remaining - recent)


def adaptive_budget_parts(seq_len: int, ratio: float, query_start: int) -> tuple[int, int, int]:
    """Preserve the actual query tail, then give the remaining budget to evidence."""
    retained_total = max(SINK + 1, int(round(seq_len * ratio)))
    remaining = max(1, retained_total - SINK)
    query_tokens = max(1, seq_len - query_start)
    recent = min(remaining, query_tokens)
    return retained_total, recent, max(0, remaining - recent)


def lexical_priority(question: str, sentences: Sequence[dict[str, Any]]) -> list[int]:
    q_terms = terms(question)
    document_frequency = Counter(term for sentence in sentences for term in terms(sentence["text"]))
    rows = []
    for idx, sentence in enumerate(sentences):
        sentence_terms = terms(sentence["text"]) | terms(sentence["title"])
        score = sum(math.log((len(sentences) + 1) / (document_frequency[term] + 1)) + 1 for term in q_terms & sentence_terms)
        rows.append((score, idx))
    return [idx for _, idx in sorted(rows, key=lambda item: (-item[0], item[1]))]


def graph_priority(question: str, sentences: Sequence[dict[str, Any]]) -> list[int]:
    q_terms = terms(question)
    sentence_terms = [terms(sentence["text"]) | terms(sentence["title"]) for sentence in sentences]
    document_frequency = Counter(term for row_terms in sentence_terms for term in row_terms)

    def idf(term: str) -> float:
        return math.log((len(sentences) + 1) / (document_frequency[term] + 1)) + 1

    selected: list[int] = []
    frontier = set(q_terms)
    remaining = set(range(len(sentences)))
    while remaining:
        scored = []
        for idx in remaining:
            overlap_q = sentence_terms[idx] & q_terms
            overlap_frontier = sentence_terms[idx] & frontier
            rare_bridge = {term for term in overlap_frontier if document_frequency[term] <= 3}
            score = (
                4.0 * sum(idf(term) for term in overlap_q)
                + 2.5 * sum(idf(term) for term in rare_bridge)
                + 0.25 * len(overlap_frontier)
            )
            scored.append((score, idx))
        score, idx = max(scored, key=lambda item: (item[0], -item[1]))
        selected.append(idx)
        remaining.remove(idx)
        frontier |= {term for term in sentence_terms[idx] if document_frequency[term] <= 3}
        if score <= 0 and len(selected) >= len(sentences):
            break
    return selected


def plan_from_priority(
    priority: Iterable[int],
    sentence_spans: Sequence[tuple[int, int]],
    seq_len: int,
    guidepost_budget: int,
    recent: int,
    halo: int = 2,
) -> GuidepostPlan:
    spans = [(sentence_spans[idx][0] - halo, sentence_spans[idx][1] + halo) for idx in priority]
    return _plan_from_spans(seq_len, spans, guidepost_budget, SINK, recent)


def coverage(plan: GuidepostPlan, gold_spans: Sequence[tuple[int, int]]) -> dict[str, float]:
    retained = set(plan.retained_indices)
    recalls = []
    full = []
    any_hit = []
    for start, end in gold_spans:
        positions = set(range(start, end))
        if not positions:
            continue
        hits = len(positions & retained)
        recalls.append(hits / len(positions))
        full.append(float(hits == len(positions)))
        any_hit.append(float(hits > 0))
    return {
        "all_gold_full": float(bool(full) and all(full)),
        "any_gold_full": float(bool(full) and any(full)),
        "fraction_gold_full": sum(full) / max(1, len(full)),
        "all_gold_any": float(bool(any_hit) and all(any_hit)),
        "fraction_gold_any": sum(any_hit) / max(1, len(any_hit)),
        "all_gold_half": float(bool(recalls) and all(value >= 0.5 for value in recalls)),
        "all_gold_eighty": float(bool(recalls) and all(value >= 0.8 for value in recalls)),
        "mean_gold_token_recall": sum(recalls) / max(1, len(recalls)),
    }


def gold_fits_budget(
    gold_spans: Sequence[tuple[int, int]], seq_len: int, sink: int, recent: int, guidepost_budget: int
) -> float:
    mandatory = set(range(min(sink, seq_len))) | set(range(max(0, seq_len - recent), seq_len))
    gold = {idx for start, end in gold_spans for idx in range(start, end)}
    return float(len(gold - mandatory) <= guidepost_budget)


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["budget_ratio"])].append(row)
    records = []
    for (method, budget), values in sorted(groups.items()):
        records.append(
            {
                "method": method,
                "budget_ratio": budget,
                "n": len(values),
                "all_gold_full": sum(v["all_gold_full"] for v in values) / len(values),
                "any_gold_full": sum(v["any_gold_full"] for v in values) / len(values),
                "fraction_gold_full": sum(v["fraction_gold_full"] for v in values) / len(values),
                "all_gold_any": sum(v["all_gold_any"] for v in values) / len(values),
                "fraction_gold_any": sum(v["fraction_gold_any"] for v in values) / len(values),
                "all_gold_half": sum(v["all_gold_half"] for v in values) / len(values),
                "all_gold_eighty": sum(v["all_gold_eighty"] for v in values) / len(values),
                "mean_gold_token_recall": sum(v["mean_gold_token_recall"] for v in values) / len(values),
                "retained_ratio": sum(v["retained_ratio"] for v in values) / len(values),
                "gold_fits_fixed_budget": sum(v["gold_fits_fixed_budget"] for v in values) / len(values),
                "gold_fits_adaptive_budget": sum(v["gold_fits_adaptive_budget"] for v in values) / len(values),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [float(value) for value in args.budgets.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    rows: list[dict[str, Any]] = []

    for dataset_idx in indices[: args.max_samples]:
        source = dataset[dataset_idx]
        prompt, sentences, query = build_hotpot_prompt(source)
        encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [(int(start), int(end)) for start, end in encoding["offset_mapping"]]
        seq_len = len(offsets)
        sentence_spans = [token_span(offsets, *sentence["char_span"]) for sentence in sentences]
        gold_spans = [span for span, sentence in zip(sentence_spans, sentences) if sentence["gold"]]
        if not gold_spans:
            continue

        query_start, _ = token_span(offsets, len(prompt) - len(query), len(prompt))

        for budget in budgets:
            _, recent, guidepost_budget = budget_parts(seq_len, budget)
            _, adaptive_recent, adaptive_guidepost_budget = adaptive_budget_parts(
                seq_len, budget, query_start
            )
            plans = {
                "uniform": block_halo_guideposts(seq_len, guidepost_budget, SINK, recent, mode="uniform"),
                "lexical": plan_from_priority(
                    lexical_priority(source["question"], sentences), sentence_spans, seq_len, guidepost_budget, recent
                ),
                "entity_graph": plan_from_priority(
                    graph_priority(source["question"], sentences), sentence_spans, seq_len, guidepost_budget, recent
                ),
                "current_compass": compass_prefill_guideposts(
                    prompt,
                    query,
                    offsets,
                    guidepost_budget,
                    SINK,
                    recent,
                    block_size=16,
                    halo=4,
                    propagation_rounds=5,
                ),
                "oracle": plan_from_priority(
                    [idx for idx, sentence in enumerate(sentences) if sentence["gold"]],
                    sentence_spans,
                    seq_len,
                    guidepost_budget,
                    recent,
                ),
                "lexical_adaptive": plan_from_priority(
                    lexical_priority(source["question"], sentences),
                    sentence_spans,
                    seq_len,
                    adaptive_guidepost_budget,
                    adaptive_recent,
                ),
                "entity_graph_adaptive": plan_from_priority(
                    graph_priority(source["question"], sentences),
                    sentence_spans,
                    seq_len,
                    adaptive_guidepost_budget,
                    adaptive_recent,
                ),
                "oracle_adaptive": plan_from_priority(
                    [idx for idx, sentence in enumerate(sentences) if sentence["gold"]],
                    sentence_spans,
                    seq_len,
                    adaptive_guidepost_budget,
                    adaptive_recent,
                ),
            }
            for method, plan in plans.items():
                row = {
                    "dataset": args.dataset,
                    "sample_id": source["id"],
                    "question_type": source["type"],
                    "level": source["level"],
                    "method": method,
                    "budget_ratio": budget,
                    "prompt_tokens": seq_len,
                    "gold_sentence_count": len(gold_spans),
                    "retained_ratio": plan.retained_ratio,
                    "gold_fits_fixed_budget": gold_fits_budget(
                        gold_spans, seq_len, SINK, recent, guidepost_budget
                    ),
                    "gold_fits_adaptive_budget": gold_fits_budget(
                        gold_spans, seq_len, SINK, adaptive_recent, adaptive_guidepost_budget
                    ),
                }
                row.update(coverage(plan, gold_spans))
                rows.append(row)

    predictions = out_dir / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "model_tokenizer": args.model,
        "budgets": budgets,
        "records": aggregate(rows),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "records": payload["records"]}, sort_keys=True))


if __name__ == "__main__":
    main()
