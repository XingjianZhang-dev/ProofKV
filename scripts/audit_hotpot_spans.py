#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from eval_natural_route_coverage import (
    SINK,
    adaptive_budget_parts,
    budget_parts,
    build_hotpot_prompt,
    graph_priority,
    plan_from_priority,
    token_span,
)
from proofkv.prefill_selectors import compass_prefill_guideposts


def recall(indices: list[int], span: tuple[int, int]) -> float:
    start, end = span
    positions = set(range(start, end))
    return len(positions & set(indices)) / max(1, len(positions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--budget", type=float, default=0.08)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    audited = []
    content_mismatch_count = 0
    shared_query_boundary_count = 0

    for dataset_idx in order[: args.samples]:
        source = dataset[dataset_idx]
        prompt, sentences, query = build_hotpot_prompt(source)
        encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [(int(start), int(end)) for start, end in encoding["offset_mapping"]]
        spans = [token_span(offsets, *sentence["char_span"]) for sentence in sentences]
        gold_ids = [idx for idx, sentence in enumerate(sentences) if sentence["gold"]]
        gold_spans = [spans[idx] for idx in gold_ids]
        query_start, _ = token_span(offsets, len(prompt) - len(query), len(prompt))

        retained_total, recent, gp_budget = budget_parts(len(offsets), args.budget)
        _, adaptive_recent, adaptive_gp_budget = adaptive_budget_parts(
            len(offsets), args.budget, query_start
        )
        plans = {
            "compass_fixed": compass_prefill_guideposts(
                prompt, query, offsets, gp_budget, SINK, recent,
                block_size=16, halo=4, propagation_rounds=5,
            ),
            "oracle_fixed": plan_from_priority(gold_ids, spans, len(offsets), gp_budget, recent),
            "graph_adaptive": plan_from_priority(
                graph_priority(source["question"], sentences), spans, len(offsets),
                adaptive_gp_budget, adaptive_recent,
            ),
            "oracle_adaptive": plan_from_priority(
                gold_ids, spans, len(offsets), adaptive_gp_budget, adaptive_recent,
            ),
        }

        details = []
        for gold_id, span in zip(gold_ids, gold_spans):
            sentence = sentences[gold_id]
            start, end = span
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            recovered = prompt[char_start:char_end]
            expected = prompt[sentence["char_span"][0]:sentence["char_span"][1]]
            content_ok = (
                start < end
                and char_start <= sentence["char_span"][0]
                and char_end >= sentence["char_span"][1]
                and expected.strip() in recovered.strip()
            )
            shared_query_boundary = end > query_start
            content_mismatch_count += int(not content_ok)
            shared_query_boundary_count += int(shared_query_boundary)
            details.append({
                "title": sentence["title"],
                "sent_id": sentence["sent_id"],
                "expected": expected,
                "recovered": recovered,
                "token_span": [start, end],
                "content_ok": content_ok,
                "shared_query_boundary_token": shared_query_boundary,
                "recall": {name: recall(plan.retained_indices, span) for name, plan in plans.items()},
            })

        for plan in plans.values():
            assert len(plan.retained_indices) <= retained_total
            assert plan.retained_indices == sorted(set(plan.retained_indices))
        audited.append({
            "id": source["id"],
            "question": source["question"],
            "prompt_tokens": len(offsets),
            "retained_total": retained_total,
            "fixed_recent": recent,
            "fixed_guidepost_budget": gp_budget,
            "adaptive_recent": adaptive_recent,
            "adaptive_guidepost_budget": adaptive_gp_budget,
            "gold": details,
        })

    payload = {
        "samples": len(audited),
        "gold_content_mismatches": content_mismatch_count,
        "shared_query_boundary_tokens": shared_query_boundary_count,
        "audited": audited,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "samples": len(audited),
        "gold_content_mismatches": content_mismatch_count,
        "shared_query_boundary_tokens": shared_query_boundary_count,
    }))
    if content_mismatch_count:
        raise SystemExit("gold/token span audit failed")


if __name__ == "__main__":
    main()
