#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import load_dataset
from tqdm import tqdm

from proofkv.eval.generation import load_model_tokenizer
from proofkv.utils.io import append_jsonl
from proofkv.physical_kv import compact_prefill_cache_stats
from audit_hotpot_spans import recall
from benchmark_physical_path import chunked_compact_prefill, full_prefill
from eval_natural_route_coverage import (
    SINK,
    adaptive_budget_parts,
    budget_parts,
    build_hotpot_prompt,
    graph_priority,
    lexical_priority,
    plan_from_priority,
    token_span,
)
from eval_proofcache_selector import select_by_priority
from run_physical_kv_replay import _greedy_decode_from_prefill_logits, _release_cuda
from proofkv.prefill_selectors import GuidepostPlan, compass_prefill_guideposts
from proofkv.prefill_selectors import _plan_from_spans
from proofkv.proofcache.index import EvidenceIndexer, extract_cues, hashed_embedding
from proofkv.proofcache.model import (
    CachedTypedRelationScorer,
    TypedRelationScorer,
    build_cached_overlay,
    build_learned_overlay,
    encode_cached_spans,
)
from proofkv.proofcache.schema import EvidenceSpan, PrecisionTier
from proofkv.proofcache.selector import AnytimeProofSelector, AndOrProofSelector, ProofSearchConfig


METHODS = {
    "full",
    "compass_fixed",
    "compass_adaptive",
    "entity_graph_adaptive",
    "oracle_adaptive",
    "learned_root_topk",
    "learned_topk_paths",
    "learned_and_or",
    "learned_and_or_aux",
    "learned_and_or_aux_cap128",
    "learned_and_or_aux_cap256",
    "learned_and_or_aux_cap384",
    "learned_and_or_diverse",
    "learned_and_or_diverse_aux",
    "learned_anytime",
    "learned_anytime_aux",
    "learned_and_or_recent_aux",
    "learned_and_or_stratified_aux",
    "learned_and_or_graph_aux",
    "learned_and_or_mixed_aux",
    "learned_and_or_control_anchor25_aux",
    "learned_and_or_control_anchor50_aux",
    "learned_and_or_control_anchor75_aux",
    "learned_and_or_lexical_anchor25_aux",
    "learned_and_or_lexical_anchor50_aux",
    "learned_and_or_lexical_anchor75_aux",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end natural multi-hop QA using physical compact prefill.")
    parser.add_argument("--dataset", choices=("hotpotqa", "musique", "2wiki"), default="hotpotqa")
    parser.add_argument("--data-file")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--unlock-locked-test", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Offset into the deterministically shuffled dataset before taking --max-samples.",
    )
    parser.add_argument("--budget", type=float, default=0.08)
    parser.add_argument(
        "--budget-tokens",
        type=int,
        default=None,
        help="Optional fixed total retained-token budget; overrides --budget per sample.",
    )
    parser.add_argument(
        "--methods",
        default="full,compass_fixed,compass_adaptive,entity_graph_adaptive,oracle_adaptive",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--full-prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--compact-prefill-chunk-size", type=int, default=512)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="HF attention backend; formal packed-context runs use flash_attention_2.",
    )
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        default=None,
        help=(
            "Deterministically pack unrelated natural paragraphs until the chat prompt is near this "
            "token length. Supporting paragraphs and the question are never truncated."
        ),
    )
    parser.add_argument("--packing-seed", type=int, default=911)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scorer-checkpoint")
    parser.add_argument("--cached-scorer-checkpoint")
    parser.add_argument("--cached-edge-non-none-floor", type=float, default=None)
    parser.add_argument("--learned-dense-core-count", type=int, default=4)
    parser.add_argument("--learned-edge-batch-size", type=int, default=256)
    parser.add_argument("--terminal-diversity-weight", type=float, default=0.05)
    parser.add_argument("--partial-proof-weight", type=float, default=0.25)
    parser.add_argument("--repair-halo", type=int, default=0)
    parser.add_argument(
        "--user-only-chat",
        action="store_true",
        help="Embed the short-answer instruction in a single user turn for legacy chat templates.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def production_full_decode(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    chunk_size: int,
) -> tuple[str, Any, list[str]]:
    warnings: list[str] = []
    prefill_out = full_prefill(model, input_ids, warnings, chunk_size)
    prompt_len = int(input_ids.shape[1])
    text, final_cache, decode_warnings = _greedy_decode_from_prefill_logits(
        model,
        tokenizer,
        prefill_out.past_key_values,
        prefill_out.logits,
        prompt_len,
        prompt_len,
        max_new_tokens,
    )
    warnings.extend(decode_warnings)
    warnings.append(f"production Full prefill chunk size: {chunk_size}")
    return text, final_cache, warnings


def production_compact_decode(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    retained_indices: Sequence[int],
    prompt_len: int,
    max_new_tokens: int,
    chunk_size: int,
) -> tuple[str, Any, dict[str, Any], list[str], dict[str, float]]:
    warnings: list[str] = []
    started = time.perf_counter()
    prefill_out, prepared = chunked_compact_prefill(
        model, input_ids, retained_indices, warnings, chunk_size
    )
    if int(prepared.retained_indices[-1].item()) != int(prompt_len - 1):
        raise ValueError("compact prefill generation requires the final prompt token")
    prefill_seconds = time.perf_counter() - started
    stats = compact_prefill_cache_stats(prompt_len, prefill_out.past_key_values)
    decode_started = time.perf_counter()
    text, final_cache, decode_warnings = _greedy_decode_from_prefill_logits(
        model,
        tokenizer,
        prefill_out.past_key_values,
        prefill_out.logits,
        prompt_len,
        int(prepared.input_ids.shape[1]),
        max_new_tokens,
    )
    decode_seconds = time.perf_counter() - decode_started
    warnings.extend(decode_warnings)
    warnings.append(f"production compact prefill chunk size: {chunk_size}")
    return text, final_cache, stats, warnings, {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": prefill_seconds + decode_seconds,
    }


def answer_from_objects(source: dict[str, Any]) -> str:
    answer = source.get("answer")
    if answer is not None:
        return str(answer)
    spans = [
        str(span)
        for answer_object in source.get("answers_objects", ())
        for span in answer_object.get("spans", ())
        if str(span).strip()
    ]
    return spans[0] if spans else ""


def normalize_ircot_source(source: dict[str, Any]) -> dict[str, Any]:
    """Map official IRCoT processed rows to the Hotpot-like prompt schema."""
    contexts = source.get("contexts")
    if contexts is None:
        return source
    supporting = [context for context in contexts if bool(context.get("is_supporting", False))]
    # Paragraph indices disambiguate repeated Wikipedia titles without exposing
    # which paragraphs are supporting evidence.
    titles = [
        f"{context.get('title', f'paragraph-{idx}')} [paragraph {context.get('idx', idx)}]"
        for idx, context in enumerate(contexts)
    ]
    return {
        "id": str(source.get("id", source.get("question_id"))),
        "question": str(source.get("question", source.get("question_text"))),
        "answer": answer_from_objects(source),
        "type": f"{len(supporting)}hop",
        "context": {
            "title": titles,
            "sentences": [[str(context["paragraph_text"])] for context in contexts],
        },
        "supporting_facts": {
            "title": [titles[idx] for idx, context in enumerate(contexts) if bool(context.get("is_supporting", False))],
            "sent_id": [0 for context in contexts if bool(context.get("is_supporting", False))],
        },
    }


def _stable_sample_seed(sample_id: str, packing_seed: int) -> int:
    digest = hashlib.sha256(f"{packing_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_natural_distractor_pool(dataset: Any) -> list[dict[str, Any]]:
    """Build a query-agnostic pool of whole natural paragraphs.

    Pool entries preserve paragraph boundaries. They are never labelled as
    distractors in the prompt, and paragraphs from the target sample are
    excluded during packing.
    """
    pool: list[dict[str, Any]] = []
    for dataset_idx in range(len(dataset)):
        source = normalize_ircot_source(dict(dataset[dataset_idx]))
        source_id = str(source["id"])
        for page_idx, (title, sentences) in enumerate(
            zip(source["context"]["title"], source["context"]["sentences"])
        ):
            clean_sentences = [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
            if clean_sentences:
                pool.append(
                    {
                        "source_id": source_id,
                        "page_idx": page_idx,
                        "title": str(title),
                        "sentences": clean_sentences,
                    }
                )
    return pool


def pack_source_to_prompt_target(
    source: dict[str, Any],
    tokenizer: Any,
    pool: list[dict[str, Any]],
    target_prompt_tokens: int,
    packing_seed: int,
    user_only_chat: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pack unrelated paragraphs without leaking labels or answer strings."""
    packed = copy.deepcopy(source)
    original_titles = [str(title) for title in packed["context"]["title"]]
    original_pages = [list(map(str, sentences)) for sentences in packed["context"]["sentences"]]
    packed["context"] = {"title": original_titles, "sentences": original_pages}
    base_prompt, _, _ = make_prompt(packed, tokenizer, user_only_chat)
    base_tokens = len(tokenizer(base_prompt, add_special_tokens=False)["input_ids"])
    if base_tokens >= target_prompt_tokens:
        return packed, {
            "target_prompt_tokens": target_prompt_tokens,
            "base_prompt_tokens": base_tokens,
            "actual_prompt_tokens": base_tokens,
            "added_pages": 0,
            "answer_leak_candidates_skipped": 0,
            "base_exceeds_target": base_tokens > target_prompt_tokens,
        }

    rng = random.Random(_stable_sample_seed(str(source["id"]), packing_seed))
    candidate_order = list(range(len(pool)))
    rng.shuffle(candidate_order)
    used_titles = set(original_titles)
    answer = normalize(str(source.get("answer", "")))
    answer_pattern = (
        re.compile(rf"(?:^|\s){re.escape(answer)}(?:\s|$)") if len(answer) >= 2 else None
    )
    answer_leak_candidates_skipped = 0
    estimated_tokens = base_tokens
    added_titles: list[str] = []

    # Leave a small margin because chat-template tokenization is not perfectly
    # additive at paragraph boundaries. The final exact pass below enforces the
    # hard target.
    for candidate_idx in candidate_order:
        candidate = pool[candidate_idx]
        title = candidate["title"]
        if candidate["source_id"] == str(source["id"]) or title in used_titles:
            continue
        paragraph = " ".join(candidate["sentences"])
        normalized_paragraph = normalize(paragraph)
        if answer_pattern is not None and answer_pattern.search(normalized_paragraph):
            answer_leak_candidates_skipped += 1
            continue
        fragment = "".join(f"[{title}] {sentence}\n" for sentence in candidate["sentences"])
        fragment_tokens = len(tokenizer(fragment, add_special_tokens=False)["input_ids"])
        if estimated_tokens + fragment_tokens > target_prompt_tokens - 16:
            continue
        packed["context"]["title"].append(title)
        packed["context"]["sentences"].append(list(candidate["sentences"]))
        added_titles.append(title)
        used_titles.add(title)
        estimated_tokens += fragment_tokens
        if estimated_tokens >= target_prompt_tokens - 64:
            break

    # Randomize paragraph position only after packing so gold evidence does not
    # stay systematically at the beginning. Title/sentence pairs remain intact.
    pages = list(zip(packed["context"]["title"], packed["context"]["sentences"]))
    rng.shuffle(pages)
    packed["context"]["title"] = [title for title, _ in pages]
    packed["context"]["sentences"] = [sentences for _, sentences in pages]

    prompt, _, _ = make_prompt(packed, tokenizer, user_only_chat)
    actual_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    # The additive estimate can overshoot by a few boundary tokens. Remove the
    # most recently admitted natural paragraphs until the exact prompt fits.
    while actual_tokens > target_prompt_tokens and added_titles:
        remove_title = added_titles.pop()
        remove_idx = packed["context"]["title"].index(remove_title)
        packed["context"]["title"].pop(remove_idx)
        packed["context"]["sentences"].pop(remove_idx)
        prompt, _, _ = make_prompt(packed, tokenizer, user_only_chat)
        actual_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    return packed, {
        "target_prompt_tokens": target_prompt_tokens,
        "base_prompt_tokens": base_tokens,
        "actual_prompt_tokens": actual_tokens,
        "added_pages": len(added_titles),
        "answer_leak_candidates_skipped": answer_leak_candidates_skipped,
        "base_exceeds_target": False,
    }


def load_sources(args: argparse.Namespace):
    if args.dataset == "hotpotqa":
        return load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    if not args.unlock_locked_test:
        raise ValueError(f"{args.dataset} is locked; pass --unlock-locked-test after protocol freeze")
    if not args.data_file:
        raise ValueError(f"{args.dataset} requires --data-file")
    suffix = Path(args.data_file).suffix.lower()
    kind = "parquet" if suffix == ".parquet" else "json"
    return load_dataset(kind, data_files={args.split: args.data_file}, split=args.split)


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction: str, answer: str) -> tuple[float, float, float]:
    pred = normalize(prediction)
    gold = normalize(answer)
    em = float(pred == gold)
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if not pred_tokens or not gold_tokens:
        f1 = float(pred_tokens == gold_tokens)
    elif overlap == 0:
        f1 = 0.0
    else:
        precision = overlap / len(pred_tokens)
        recall_value = overlap / len(gold_tokens)
        f1 = 2 * precision * recall_value / (precision + recall_value)
    contains = float(bool(gold) and gold in pred)
    return em, f1, contains


def extract_short_answer(prediction: str) -> str:
    for line in prediction.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:answer|assistant)\s*:\s*", "", line, flags=re.I)
        return line.strip()
    return prediction.strip()


def make_prompt(
    source: dict[str, Any], tokenizer: Any, user_only_chat: bool = False
) -> tuple[str, list[dict[str, Any]], str]:
    base_prompt, sentences, old_query = build_hotpot_prompt(source)
    context = base_prompt[: -len(old_query)]
    user = f"{context}\nQuestion: {source['question']}\nReturn only the shortest answer phrase from the context."
    if user_only_chat:
        user += " Do not explain."
        messages = [{"role": "user", "content": user}]
    else:
        messages = [
            {"role": "system", "content": "Answer using only a short answer phrase. Do not explain."},
            {"role": "user", "content": user},
        ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    context_start = prompt.find(context)
    if context_start < 0:
        raise ValueError("chat template did not preserve the context text")
    shifted_sentences = [
        {**sentence, "char_span": (sentence["char_span"][0] + context_start, sentence["char_span"][1] + context_start)}
        for sentence in sentences
    ]
    query = prompt[context_start + len(context):]
    return prompt, shifted_sentences, query


def _coverage_chunks(
    mode: str,
    evidence_start: int,
    evidence_end: int,
    remaining_budget: int,
    root_spans: Sequence[tuple[int, int]],
    graph_spans: Sequence[tuple[int, int]],
    lexical_spans: Sequence[tuple[int, int]],
    chunk_size: int = 16,
) -> list[list[int]]:
    """Build deterministic auxiliary token chunks for proof-core residuals."""
    if remaining_budget <= 0 or evidence_end <= evidence_start:
        return []

    def clipped_chunk(start: int, end: int) -> list[int]:
        start = max(evidence_start, min(evidence_end, start))
        end = max(start, min(evidence_end, end))
        return list(range(start, end))

    def span_chunks(spans: Sequence[tuple[int, int]]) -> list[list[int]]:
        output: list[list[int]] = []
        for start, end in spans:
            clipped = clipped_chunk(start, end)
            output.extend(
                clipped[index : index + chunk_size]
                for index in range(0, len(clipped), chunk_size)
            )
        return output

    # Root and graph spans can be much longer than one residual chunk.  Split
    # them before mixed-mode round robin so the first long span cannot consume
    # the entire coverage budget.
    root_chunks = span_chunks(root_spans)
    graph_chunks = span_chunks(graph_spans)
    lexical_chunks = span_chunks(lexical_spans)
    recent_chunks = [
        list(
            reversed(
                clipped_chunk(start, min(evidence_end, start + chunk_size))
            )
        )
        for start in range(evidence_end - chunk_size, evidence_start - chunk_size, -chunk_size)
    ]
    recent_chunks = [chunk for chunk in recent_chunks if chunk]

    window_count = max(1, (remaining_budget + chunk_size - 1) // chunk_size)
    max_start = max(evidence_start, evidence_end - chunk_size)
    if window_count == 1:
        uniform_starts = [(evidence_start + max_start) // 2]
    else:
        uniform_starts = [
            evidence_start
            + round(index * (max_start - evidence_start) / (window_count - 1))
            for index in range(window_count)
        ]
    stratified_chunks = [
        clipped_chunk(start, min(evidence_end, start + chunk_size))
        for start in uniform_starts
    ]
    stratified_chunks = [chunk for chunk in stratified_chunks if chunk]

    if mode == "recent":
        streams = [recent_chunks, stratified_chunks, root_chunks, graph_chunks]
    elif mode == "stratified":
        streams = [stratified_chunks, recent_chunks, root_chunks, graph_chunks]
    elif mode == "graph":
        streams = [graph_chunks, root_chunks, recent_chunks, stratified_chunks]
    elif mode == "lexical_anchor":
        streams = [
            lexical_chunks,
            graph_chunks,
            root_chunks,
            recent_chunks,
            stratified_chunks,
        ]
    elif mode == "mixed":
        streams = [root_chunks, graph_chunks, recent_chunks, stratified_chunks]
    else:
        raise ValueError(f"unknown coverage mode: {mode}")

    if mode != "mixed":
        return [chunk for stream in streams for chunk in stream]

    # Round-robin prevents one auxiliary signal from consuming the entire
    # residual budget before the other coverage signals receive any capacity.
    output: list[list[int]] = []
    for index in range(max((len(stream) for stream in streams), default=0)):
        for stream in streams:
            if index < len(stream):
                output.append(stream[index])
    return output


def _augment_plan_with_coverage(
    base_plan: GuidepostPlan,
    seq_len: int,
    guidepost_budget: int,
    sink_size: int,
    recent_size: int,
    evidence_end: int,
    mode: str,
    root_spans: Sequence[tuple[int, int]],
    graph_spans: Sequence[tuple[int, int]],
    lexical_spans: Sequence[tuple[int, int]],
    preserve_control: bool = False,
) -> tuple[GuidepostPlan, list[int], list[int]]:
    """Preserve the control tail, then fill unused budget with evidence.

    The official LongBench prompt repeats its output-format instruction just
    before the question.  Evidence-only compactors can remove that instruction
    even when they retain the question itself, causing a semantically correct
    answer to become a long explanation that official QA-F1 penalizes.  The
    control channel reserves up to 16 pre-question tokens before allocating the
    remaining guidepost budget between proof and lexical evidence.
    """
    remaining = max(0, guidepost_budget - len(base_plan.guidepost_indices))
    if remaining <= 0:
        return base_plan, [], []
    stream = set(range(min(sink_size, seq_len))) | set(
        range(max(0, seq_len - recent_size), seq_len)
    )
    existing = set(base_plan.guidepost_indices)
    control_end = max(sink_size, min(seq_len - recent_size, evidence_end))
    control_start = max(sink_size, control_end - 16)
    control_selected: list[int] = []
    selected: list[int] = []
    seen = set(stream) | existing
    # Nearest-to-query tokens are most likely to contain the terminal
    # "do not explain" / "only give the answer" constraint.
    if preserve_control:
        for token in range(control_end - 1, control_start - 1, -1):
            if token in seen:
                continue
            control_selected.append(token)
            seen.add(token)
            if len(control_selected) >= remaining:
                break
    remaining -= len(control_selected)
    chunks = _coverage_chunks(
        mode,
        max(0, sink_size),
        max(0, min(seq_len - recent_size, evidence_end)),
        remaining,
        root_spans,
        graph_spans,
        lexical_spans,
    )
    for chunk in chunks:
        for token in chunk:
            if token in seen:
                continue
            selected.append(token)
            seen.add(token)
            if len(selected) >= remaining:
                break
        if len(selected) >= remaining:
            break
    # A complete, deterministic fallback ensures exact capacity whenever the
    # document has enough selectable evidence tokens.
    if len(selected) < remaining:
        for token in range(max(0, min(seq_len - recent_size, evidence_end)) - 1, sink_size - 1, -1):
            if token in seen:
                continue
            selected.append(token)
            seen.add(token)
            if len(selected) >= remaining:
                break
    guideposts = sorted(
        existing | set(control_selected) | set(selected)
    )
    retained = sorted(stream | set(guideposts))
    return (
        GuidepostPlan(
            guidepost_indices=guideposts,
            retained_indices=retained,
            spans=base_plan.spans,
            retained_ratio=len(retained) / max(1, seq_len),
        ),
        sorted(selected),
        sorted(control_selected),
    )


def make_plan(
    method: str,
    source: dict[str, Any],
    prompt: str,
    query: str,
    sentences: list[dict[str, Any]],
    offsets: list[tuple[int, int]],
    budget: float,
    learned_bundle: tuple[str, Any, Any] | None = None,
    terminal_diversity_weight: float = 0.05,
    partial_proof_weight: float = 0.25,
    repair_halo: int = 0,
    cached_edge_non_none_floor: float | None = None,
    learned_dense_core_count: int = 16,
    learned_edge_batch_size: int = 64,
    learned_overlay_cache: dict[str, tuple[Any, Any]] | None = None,
    selection_metadata_out: dict[str, Any] | None = None,
    include_selection_indices: bool = False,
):
    seq_len = len(offsets)
    sentence_spans = [token_span(offsets, *sentence["char_span"]) for sentence in sentences]
    gold_ids = [idx for idx, sentence in enumerate(sentences) if sentence["gold"]]
    query_start, _ = token_span(offsets, len(prompt) - len(query), len(prompt))
    _, fixed_recent, fixed_gp = budget_parts(seq_len, budget)
    _, adaptive_recent, adaptive_gp = adaptive_budget_parts(seq_len, budget, query_start)
    if method == "compass_fixed":
        return compass_prefill_guideposts(
            prompt, query, offsets, fixed_gp, SINK, fixed_recent,
            block_size=16, halo=4, propagation_rounds=5,
        ), sentence_spans, gold_ids
    if method == "compass_adaptive":
        return compass_prefill_guideposts(
            prompt, query, offsets, adaptive_gp, SINK, adaptive_recent,
            block_size=16, halo=4, propagation_rounds=5,
        ), sentence_spans, gold_ids
    if method == "entity_graph_adaptive":
        priority = graph_priority(source["question"], sentences)
        return plan_from_priority(priority, sentence_spans, seq_len, adaptive_gp, adaptive_recent), sentence_spans, gold_ids
    if method == "oracle_adaptive":
        return plan_from_priority(gold_ids, sentence_spans, seq_len, adaptive_gp, adaptive_recent), sentence_spans, gold_ids
    if method in {
        "learned_root_topk",
        "learned_topk_paths",
        "learned_and_or",
        "learned_and_or_aux",
        "learned_and_or_aux_cap128",
        "learned_and_or_aux_cap256",
        "learned_and_or_aux_cap384",
        "learned_and_or_diverse",
        "learned_and_or_diverse_aux",
        "learned_anytime",
        "learned_anytime_aux",
        "learned_and_or_recent_aux",
        "learned_and_or_stratified_aux",
        "learned_and_or_graph_aux",
        "learned_and_or_mixed_aux",
        "learned_and_or_control_anchor25_aux",
        "learned_and_or_control_anchor50_aux",
        "learned_and_or_control_anchor75_aux",
        "learned_and_or_lexical_anchor25_aux",
        "learned_and_or_lexical_anchor50_aux",
        "learned_and_or_lexical_anchor75_aux",
    }:
        if learned_bundle is None:
            raise ValueError(f"{method} requires --scorer-checkpoint")
        scorer_kind, learned_model, learned_tokenizer = learned_bundle
        cache_key = str(source["id"])
        if learned_overlay_cache is not None and cache_key in learned_overlay_cache:
            overlay, learned_index = learned_overlay_cache[cache_key]
        else:
            evidence_spans = [
                EvidenceSpan(
                    f"s{idx}", start, end,
                    f"[{sentence['title']}] {sentence['text']}",
                    cues=extract_cues(f"{sentence['title']} {sentence['text']}"),
                    embedding=hashed_embedding(f"{sentence['title']} {sentence['text']}", 64),
                    metadata={"title": sentence["title"], "sentence_index": sentence["sent_id"]},
                )
                for idx, ((start, end), sentence) in enumerate(zip(sentence_spans, sentences))
            ]
            index = EvidenceIndexer().build(str(source["id"]), evidence_spans, seq_len)
            search_config = ProofSearchConfig(
                root_top_k=16,
                beam_size=8,
                max_hops=8,
                max_proofs=8,
                normalize_path_probability=False,
            )
            if scorer_kind == "cached":
                span_embeddings = encode_cached_spans(
                    learned_model, learned_tokenizer, [span.text for span in index.spans]
                )
                overlay, learned_index = build_cached_overlay(
                    learned_model,
                    learned_tokenizer,
                    source["question"],
                    index,
                    span_embeddings,
                    candidate_span_count=32,
                    search_config=search_config,
                    candidate_strategy="root",
                    edge_non_none_floor=cached_edge_non_none_floor,
                )
            else:
                overlay, learned_index = build_learned_overlay(
                    learned_model,
                    learned_tokenizer,
                    source["question"],
                    index,
                    candidate_span_count=32,
                    search_config=search_config,
                    candidate_strategy="root",
                    dense_core_count=learned_dense_core_count,
                    edge_batch_size=learned_edge_batch_size,
                )
            if learned_overlay_cache is not None:
                learned_overlay_cache[cache_key] = (overlay, learned_index)
        coverage_mode = None
        coverage_anchor_fraction = 0.0
        core_guidepost_budget = adaptive_gp
        coverage_auxiliary_indices: list[int] = []
        control_indices: list[int] = []
        if method == "learned_root_topk":
            selected_ids = [
                span_id
                for span_id, _ in sorted(
                    overlay.root_scores.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            selected_ids = list(
                select_by_priority(
                    selected_ids,
                    learned_index.by_id,
                    adaptive_gp,
                )
            )
            core_ids = list(selected_ids)
            auxiliary_ids = []
        elif method == "learned_topk_paths":
            selected_ids = []
            for proof in overlay.proofs:
                for span_id in proof.span_ids:
                    if span_id not in selected_ids:
                        selected_ids.append(span_id)
            core_ids = list(selected_ids)
            auxiliary_ids: list[str] = []
        elif method in {"learned_anytime", "learned_anytime_aux"}:
            selected_overlay = AnytimeProofSelector(
                fill_auxiliary=method.endswith("_aux"),
                partial_proof_weight=partial_proof_weight,
            ).select(
                overlay,
                learned_index,
                byte_budget=adaptive_gp * 2,
                values_per_token=1,
                precision=PrecisionTier.BF16,
            )
            core_ids = list(selected_overlay.selected_span_ids)
            auxiliary_ids = list(selected_overlay.auxiliary_span_ids)
            selected_ids = list(selected_overlay.selected_span_ids + selected_overlay.auxiliary_span_ids)
        elif method in {
            "learned_and_or_recent_aux",
            "learned_and_or_stratified_aux",
            "learned_and_or_graph_aux",
            "learned_and_or_mixed_aux",
            "learned_and_or_control_anchor25_aux",
            "learned_and_or_control_anchor50_aux",
            "learned_and_or_control_anchor75_aux",
            "learned_and_or_lexical_anchor25_aux",
            "learned_and_or_lexical_anchor50_aux",
            "learned_and_or_lexical_anchor75_aux",
        }:
            if "control_anchor" in method:
                coverage_mode = "lexical_anchor"
                anchor_percent = int(
                    method.split("control_anchor", 1)[1].split("_", 1)[0]
                )
                coverage_anchor_fraction = anchor_percent / 100.0
                control_reserve = min(16, adaptive_gp)
                evidence_guidepost_budget = max(
                    0,
                    adaptive_gp - control_reserve,
                )
                core_guidepost_budget = max(
                    0,
                    int(
                        round(
                            evidence_guidepost_budget
                            * (1.0 - coverage_anchor_fraction)
                        )
                    ),
                )
            elif "lexical_anchor" in method:
                coverage_mode = "lexical_anchor"
                anchor_percent = int(
                    method.split("lexical_anchor", 1)[1].split("_", 1)[0]
                )
                coverage_anchor_fraction = anchor_percent / 100.0
                core_guidepost_budget = max(
                    1,
                    int(
                        round(
                            adaptive_gp
                            * (1.0 - coverage_anchor_fraction)
                        )
                    ),
                )
            else:
                coverage_mode = (
                    method.removeprefix("learned_and_or_")
                    .removesuffix("_aux")
                )
            selected_overlay = AndOrProofSelector(fill_auxiliary=False).select(
                overlay,
                learned_index,
                byte_budget=core_guidepost_budget * 2,
                values_per_token=1,
                precision=PrecisionTier.BF16,
            )
            core_ids = list(selected_overlay.selected_span_ids)
            auxiliary_ids = []
            selected_ids = list(core_ids)
        else:
            diverse = "_diverse" in method
            fill_auxiliary = method == "learned_and_or_aux" or "_aux_cap" in method
            selected_overlay = AndOrProofSelector(
                fill_auxiliary=fill_auxiliary,
                correlation_aware=diverse,
                terminal_diversity_weight=(terminal_diversity_weight if diverse else 0.0),
            ).select(
                overlay,
                learned_index,
                byte_budget=adaptive_gp * 2,
                values_per_token=1,
                precision=PrecisionTier.BF16,
            )
            core_ids = list(selected_overlay.selected_span_ids)
            auxiliary_ids = list(selected_overlay.auxiliary_span_ids)
            if "_aux_cap" in method:
                auxiliary_cap = int(method.rsplit("_aux_cap", 1)[1])
                auxiliary_set = set(auxiliary_ids)
                priority: list[str] = []
                for proof in overlay.proofs:
                    for span_id in proof.span_ids:
                        if span_id in auxiliary_set and span_id not in priority:
                            priority.append(span_id)
                priority.extend(
                    span_id
                    for span_id, _ in sorted(
                        overlay.root_scores.items(), key=lambda item: (-item[1], item[0])
                    )
                    if span_id in auxiliary_set and span_id not in priority
                )
                priority.extend(
                    span_id for span_id in auxiliary_ids if span_id not in priority
                )
                capped_ids: list[str] = []
                capped_tokens = 0
                for span_id in priority:
                    span = learned_index.by_id[span_id]
                    span_tokens = max(0, int(span.token_end) - int(span.token_start))
                    if capped_tokens + span_tokens > auxiliary_cap:
                        continue
                    capped_ids.append(span_id)
                    capped_tokens += span_tokens
                auxiliary_ids = capped_ids
            selected_ids = list(selected_overlay.selected_span_ids + selected_overlay.auxiliary_span_ids)
            if "_aux_cap" in method:
                selected_ids = list(core_ids + auxiliary_ids)
        selected_spans = [
            (
                max(0, learned_index.by_id[span_id].token_start - repair_halo),
                min(seq_len, learned_index.by_id[span_id].token_end + repair_halo),
            )
            for span_id in selected_ids
        ]
        plan = _plan_from_spans(
            seq_len,
            selected_spans,
            core_guidepost_budget,
            SINK,
            adaptive_recent,
        )
        if coverage_mode is not None:
            root_span_ids = [
                span_id
                for span_id, _ in sorted(
                    overlay.root_scores.items(), key=lambda item: (-item[1], item[0])
                )
                if span_id not in set(core_ids)
            ]
            root_spans = [
                (
                    learned_index.by_id[span_id].token_start,
                    learned_index.by_id[span_id].token_end,
                )
                for span_id in root_span_ids
            ]
            graph_spans = [
                sentence_spans[index]
                for index in graph_priority(source["question"], sentences)
            ]
            lexical_spans = [
                sentence_spans[index]
                for index in lexical_priority(
                    source["question"],
                    sentences,
                )
            ]
            (
                plan,
                coverage_auxiliary_indices,
                control_indices,
            ) = _augment_plan_with_coverage(
                plan,
                seq_len,
                adaptive_gp,
                SINK,
                adaptive_recent,
                query_start,
                coverage_mode,
                root_spans,
                graph_spans,
                lexical_spans,
                preserve_control=("control_anchor" in method),
            )
        if selection_metadata_out is not None:
            retained = set(map(int, plan.retained_indices))
            sink = set(range(min(SINK, seq_len))) & retained
            recent = set(range(max(0, seq_len - adaptive_recent), seq_len)) & retained

            def tokens_for(span_ids: list[str]) -> set[int]:
                tokens: set[int] = set()
                for span_id in span_ids:
                    span = learned_index.by_id[span_id]
                    start = max(0, span.token_start - repair_halo)
                    end = min(seq_len, span.token_end + repair_halo)
                    tokens.update(range(start, end))
                return tokens & retained

            core_tokens = tokens_for(core_ids) - sink - recent
            auxiliary_tokens = tokens_for(auxiliary_ids) - sink - recent - core_tokens
            coverage_tokens = (
                set(coverage_auxiliary_indices)
                & retained
                - sink
                - recent
                - core_tokens
                - auxiliary_tokens
            )
            control_tokens = (
                set(control_indices)
                & retained
                - sink
                - recent
                - core_tokens
                - auxiliary_tokens
            )
            coverage_tokens -= control_tokens
            unassigned = (
                retained
                - sink
                - recent
                - core_tokens
                - auxiliary_tokens
                - coverage_tokens
                - control_tokens
            )
            proof_probabilities = sorted(
                (float(proof.probability) for proof in overlay.proofs),
                reverse=True,
            )
            root_probabilities = sorted(
                (float(value) for value in overlay.root_scores.values()),
                reverse=True,
            )
            terminal_probabilities = sorted(
                (float(value) for value in overlay.terminal_scores.values()),
                reverse=True,
            )
            selection_metadata_out.update(
                {
                    "proof_core_span_ids": core_ids,
                    "auxiliary_span_ids": auxiliary_ids,
                    "proof_core_tokens": len(core_tokens),
                    "auxiliary_tokens": len(auxiliary_tokens),
                    "coverage_auxiliary_tokens": len(coverage_tokens),
                    "coverage_auxiliary_mode": coverage_mode,
                    "coverage_anchor_fraction": (
                        coverage_anchor_fraction
                    ),
                    "control_tokens": len(control_tokens),
                    "control_reserve": (
                        min(16, adaptive_gp)
                        if "control_anchor" in method
                        else 0
                    ),
                    "sink_tokens": len(sink),
                    "recent_tokens": len(recent),
                    "unassigned_tokens": len(unassigned),
                    "auxiliary_token_cap": (
                        int(method.rsplit("_aux_cap", 1)[1]) if "_aux_cap" in method else None
                    ),
                    "overlay_confidence": float(overlay.confidence),
                    "overlay_num_proofs": len(overlay.proofs),
                    "overlay_top_proof_probability": (
                        proof_probabilities[0] if proof_probabilities else 0.0
                    ),
                    "overlay_proof_probability_margin": (
                        proof_probabilities[0] - proof_probabilities[1]
                        if len(proof_probabilities) >= 2
                        else (proof_probabilities[0] if proof_probabilities else 0.0)
                    ),
                    "overlay_top_proof_spans": (
                        len(overlay.proofs[0].span_ids) if overlay.proofs else 0
                    ),
                    "overlay_root_top_probability": (
                        root_probabilities[0] if root_probabilities else 0.0
                    ),
                    "overlay_root_probability_margin": (
                        root_probabilities[0] - root_probabilities[1]
                        if len(root_probabilities) >= 2
                        else (root_probabilities[0] if root_probabilities else 0.0)
                    ),
                    "overlay_terminal_top_probability": (
                        terminal_probabilities[0] if terminal_probabilities else 0.0
                    ),
                    "overlay_terminal_probability_margin": (
                        terminal_probabilities[0] - terminal_probabilities[1]
                        if len(terminal_probabilities) >= 2
                        else (
                            terminal_probabilities[0]
                            if terminal_probabilities
                            else 0.0
                        )
                    ),
                }
            )
            if include_selection_indices:
                def span_token_map(span_ids: list[str], excluded: set[int]) -> dict[str, list[int]]:
                    mapping: dict[str, list[int]] = {}
                    for span_id in span_ids:
                        span = learned_index.by_id[span_id]
                        start = max(0, span.token_start - repair_halo)
                        end = min(seq_len, span.token_end + repair_halo)
                        mapping[span_id] = sorted(
                            (set(range(start, end)) & retained) - sink - recent - excluded
                        )
                    return mapping

                selection_metadata_out.update(
                    {
                        "proof_core_indices": sorted(core_tokens),
                        "auxiliary_indices": sorted(auxiliary_tokens),
                        "coverage_auxiliary_indices": sorted(coverage_tokens),
                        "sink_indices": sorted(sink),
                        "recent_indices": sorted(recent),
                        "unassigned_indices": sorted(unassigned),
                        "proof_core_span_indices": span_token_map(core_ids, set()),
                        "auxiliary_span_indices": span_token_map(auxiliary_ids, core_tokens),
                    }
                )
        return plan, sentence_spans, gold_ids
    raise ValueError(method)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    return [
        {
            "method": method,
            "n": len(values),
            "answer_em": sum(v["answer_em"] for v in values) / len(values),
            "answer_f1": sum(v["answer_f1"] for v in values) / len(values),
            "answer_contains": sum(v["answer_contains"] for v in values) / len(values),
            "gold_token_recall": sum(v["gold_token_recall"] for v in values) / len(values),
            "retained_ratio": sum(v["retained_ratio"] for v in values) / len(values),
            "seconds": sum(v["seconds"] for v in values) / len(values),
            "selection_seconds": sum(v.get("selection_seconds", 0.0) for v in values) / len(values),
            "generation_seconds": sum(v.get("generation_seconds", v["seconds"]) for v in values) / len(values),
            "proof_core_tokens": sum(
                (v.get("selection_partition") or {}).get("proof_core_tokens", 0)
                for v in values
            ) / len(values),
            "auxiliary_tokens": sum(
                (v.get("selection_partition") or {}).get("auxiliary_tokens", 0)
                for v in values
            ) / len(values),
        }
        for method, values in sorted(grouped.items())
    ]


def paired_full_retention(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_method[str(row["method"])][str(row["id"])] = row
    full = by_method.get("full", {})
    records = []
    for method, values in sorted(by_method.items()):
        if method == "full":
            continue
        paired_ids = sorted(set(full) & set(values))
        if not paired_ids:
            continue
        full_f1_sum = sum(float(full[sample_id]["answer_f1"]) for sample_id in paired_ids)
        method_f1_sum = sum(float(values[sample_id]["answer_f1"]) for sample_id in paired_ids)
        full_correct_ids = [
            sample_id for sample_id in paired_ids if float(full[sample_id]["answer_em"]) >= 0.5
        ]
        records.append(
            {
                "method": method,
                "n_paired": len(paired_ids),
                "relative_f1_to_full": method_f1_sum / full_f1_sum if full_f1_sum > 0 else None,
                "n_full_em_correct": len(full_correct_ids),
                "em_retention_on_full_em_correct": (
                    sum(float(values[sample_id]["answer_em"]) for sample_id in full_correct_ids)
                    / len(full_correct_ids)
                    if full_correct_ids
                    else None
                ),
                "f1_on_full_em_correct": (
                    sum(float(values[sample_id]["answer_f1"]) for sample_id in full_correct_ids)
                    / len(full_correct_ids)
                    if full_correct_ids
                    else None
                ),
            }
        )
    return records


def read_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    keys = [(str(row["id"]), str(row["method"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate sample/method rows in {path}")
    return rows


def main() -> None:
    args = parse_args()
    if not 0.0 < args.budget <= 1.0:
        raise ValueError("budget must lie in (0, 1]")
    if args.target_prompt_tokens is not None and args.target_prompt_tokens <= 0:
        raise ValueError("target-prompt-tokens must be positive")
    if args.budget_tokens is not None and args.budget_tokens <= SINK:
        raise ValueError(f"budget-tokens must exceed the {SINK}-token sink")
    if args.scorer_checkpoint and args.cached_scorer_checkpoint:
        raise ValueError("provide at most one scorer checkpoint")
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = set(methods) - METHODS
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    metrics_path = out_dir / "metrics.json"
    run_manifest_path = out_dir / "run_manifest.json"
    if metrics_path.exists():
        raise FileExistsError(f"completed output already exists: {metrics_path}")

    dataset = load_sources(args)
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    if args.sample_offset < 0:
        raise ValueError("--sample-offset must be non-negative")
    selected_indices = order[args.sample_offset : args.sample_offset + args.max_samples]
    if len(selected_indices) != args.max_samples:
        raise ValueError(
            f"requested {args.max_samples} samples at offset {args.sample_offset}, "
            f"but only {len(selected_indices)} are available"
        )
    selected_ids = [str(normalize_ircot_source(dict(dataset[idx]))["id"]) for idx in selected_indices]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected dataset contains duplicate ids")
    run_manifest = {
        "arguments": {key: value for key, value in vars(args).items() if key != "out_dir"},
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "selected_indices": selected_indices,
        "selected_ids": selected_ids,
    }
    if run_manifest_path.exists():
        existing_manifest = json.loads(run_manifest_path.read_text())
        if existing_manifest != run_manifest:
            raise ValueError("run manifest does not match this resumed compact-QA evaluation")
    else:
        if results_path.exists():
            raise ValueError("cannot resume results without a frozen run manifest")
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    model, tokenizer = load_model_tokenizer(
        args.model,
        dtype="bfloat16",
        attn_implementation=args.attn_implementation,
    )
    source_cache: dict[int, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    if args.target_prompt_tokens is not None:
        distractor_pool = build_natural_distractor_pool(dataset)
        for dataset_idx in tqdm(selected_indices, desc="pack-natural-contexts"):
            source = normalize_ircot_source(dict(dataset[dataset_idx]))
            packed_source, packing = pack_source_to_prompt_target(
                source,
                tokenizer,
                distractor_pool,
                args.target_prompt_tokens,
                args.packing_seed,
                args.user_only_chat,
            )
            source_cache[dataset_idx] = (packed_source, packing)
    else:
        for dataset_idx in selected_indices:
            source_cache[dataset_idx] = (
                normalize_ircot_source(dict(dataset[dataset_idx])),
                None,
            )
    learned_bundle = None
    if any(method.startswith("learned_") for method in methods):
        if args.cached_scorer_checkpoint:
            scorer, scorer_tokenizer = CachedTypedRelationScorer.load_checkpoint(
                args.cached_scorer_checkpoint, device=next(model.parameters()).device
            )
            learned_bundle = ("cached", scorer, scorer_tokenizer)
        elif args.scorer_checkpoint:
            scorer, scorer_tokenizer = TypedRelationScorer.load_checkpoint(
                args.scorer_checkpoint, device=next(model.parameters()).device
            )
            learned_bundle = ("cross", scorer, scorer_tokenizer)
        else:
            raise ValueError("learned methods require a scorer checkpoint")
    rows = read_existing_results(results_path)
    completed = {(str(row["id"]), str(row["method"])) for row in rows}
    learned_overlay_cache: dict[str, tuple[Any, Any]] = {}

    for method in methods:
        for dataset_idx in tqdm(selected_indices, desc=method):
            source, packing = source_cache[dataset_idx]
            source_id = str(source["id"])
            if (source_id, method) in completed:
                continue
            prompt, sentences, query = make_prompt(source, tokenizer, args.user_only_chat)
            encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
            offsets = [(int(start), int(end)) for start, end in encoding.pop("offset_mapping")[0].tolist()]
            input_ids = encoding["input_ids"].to(next(model.parameters()).device)
            effective_budget = (
                min(1.0, args.budget_tokens / len(offsets))
                if args.budget_tokens is not None
                else args.budget
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            selection_started = time.perf_counter()
            if method == "full":
                selection_partition = None
                selection_seconds = 0.0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                generation_started = time.perf_counter()
                prediction, cache, warnings = production_full_decode(
                    model, tokenizer, input_ids, args.max_new_tokens,
                    args.full_prefill_chunk_size,
                )
                retained_ratio = 1.0
                gold_recall = 1.0
            else:
                selector_cache_hit = source_id in learned_overlay_cache if method.startswith("learned_") else None
                selection_partition: dict[str, Any] = {}
                plan, sentence_spans, gold_ids = make_plan(
                    method,
                    source,
                    prompt,
                    query,
                    sentences,
                    offsets,
                    effective_budget,
                    learned_bundle,
                    args.terminal_diversity_weight,
                    args.partial_proof_weight,
                    args.repair_halo,
                    args.cached_edge_non_none_floor,
                    args.learned_dense_core_count,
                    args.learned_edge_batch_size,
                    learned_overlay_cache,
                    selection_partition,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                selection_seconds = time.perf_counter() - selection_started
                generation_started = time.perf_counter()
                prediction, cache, _, warnings, _ = production_compact_decode(
                    model, tokenizer, input_ids, plan.retained_indices,
                    input_ids.shape[1], args.max_new_tokens,
                    args.compact_prefill_chunk_size,
                )
                retained_ratio = len(plan.retained_indices) / input_ids.shape[1]
                gold_recall = sum(
                    recall(plan.retained_indices, sentence_spans[idx]) for idx in gold_ids
                ) / max(1, len(gold_ids))
            if method == "full":
                selector_cache_hit = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_started
            seconds = selection_seconds + generation_seconds
            del cache
            _release_cuda()
            short_prediction = extract_short_answer(prediction)
            em, f1, contains = answer_scores(short_prediction, source["answer"])
            row = {
                "id": source["id"],
                "question_type": source["type"],
                "method": method,
                "budget": 1.0 if method == "full" else effective_budget,
                "budget_tokens_requested": None if method == "full" else args.budget_tokens,
                "prompt_tokens": input_ids.shape[1],
                "target_prompt_tokens": args.target_prompt_tokens,
                "packing": packing,
                "retained_ratio": retained_ratio,
                "gold_token_recall": gold_recall,
                "answer": source["answer"],
                "prediction": prediction,
                "short_prediction": short_prediction,
                "answer_em": em,
                "answer_f1": f1,
                "answer_contains": contains,
                "seconds": seconds,
                "selection_seconds": selection_seconds,
                "generation_seconds": generation_seconds,
                "selector_overlay_cache_hit": selector_cache_hit,
                "selection_partition": selection_partition,
                "warnings": warnings,
            }
            rows.append(row)
            append_jsonl(results_path, row)
            completed.add((source_id, method))

    expected_rows = len(methods) * len(selected_indices)
    if len(rows) != expected_rows:
        raise RuntimeError(f"incomplete result matrix: {len(rows)} != {expected_rows}")

    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "data_file": args.data_file,
        "max_samples": args.max_samples,
        "sample_offset": args.sample_offset,
        "budget": args.budget,
        "budget_tokens": args.budget_tokens,
        "target_prompt_tokens": args.target_prompt_tokens,
        "packing_seed": args.packing_seed if args.target_prompt_tokens is not None else None,
        "attn_implementation": args.attn_implementation,
        "full_prefill_chunk_size": args.full_prefill_chunk_size,
        "compact_prefill_chunk_size": args.compact_prefill_chunk_size,
        "methods": methods,
        "scorer_kind": None if learned_bundle is None else learned_bundle[0],
        "learned_dense_core_count": args.learned_dense_core_count,
        "learned_edge_batch_size": args.learned_edge_batch_size,
        "repair_halo": args.repair_halo,
        "records": aggregate(rows),
        "paired_full_retention": paired_full_retention(rows),
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "records": payload["records"]}))


if __name__ == "__main__":
    main()
