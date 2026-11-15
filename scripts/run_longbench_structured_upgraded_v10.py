#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import DynamicCache

from proofkv.eval.generation import load_model_tokenizer
from proofkv.proofcache.model import TypedRelationScorer
from proofkv.utils.io import append_jsonl
from freeze_longbench_structured_selection import ANSWER_PREFIX, official_fields, row_id
from run_hotpot_compact_qa_upgraded_v10 import (
    make_plan,
    production_compact_decode,
)
from run_kvpress_natural_qa import fixed_budget_targets, make_press
from run_physical_kv_replay import _release_cuda


BASELINE_METHODS = {
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "chunkkv_snapkv",
}
SELECTOR_METHODS = {
    "entity_graph_adaptive",
    "learned_root_topk",
    "learned_topk_paths",
    "learned_and_or_aux",
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
    "risk_calibrated_proof_coverage",
    "utility_calibrated_proof_coverage",
    "confirmation_stable_proofkv_v10",
}
METHODS = {"full"} | BASELINE_METHODS | SELECTOR_METHODS
RISK_FEATURES = (
    "overlay_confidence",
    "overlay_top_proof_probability",
    "overlay_proof_probability_margin",
    "overlay_root_top_probability",
    "overlay_root_probability_margin",
    "overlay_terminal_top_probability",
    "overlay_terminal_probability_margin",
    "overlay_num_proofs",
    "overlay_top_proof_spans",
    "proof_core_fraction",
    "coverage_auxiliary_fraction",
    "context_log_tokens",
    "question_budget_fraction",
)
MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common-interface non-truncated LongBench structured-QA matrix.")
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--methods", default=",".join(sorted(METHODS)))
    parser.add_argument("--capacities", default="64,128,256,512")
    parser.add_argument(
        "--tasks",
        default=None,
        help=(
            "Optional comma-separated task subset for a targeted smoke run. "
            "The default evaluates every task in the frozen selection."
        ),
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--learned-dense-core-count", type=int, default=4)
    parser.add_argument("--learned-edge-batch-size", type=int, default=256)
    parser.add_argument(
        "--risk-controller-config",
        default=None,
        help="Frozen calibration JSON required by risk_calibrated_proof_coverage.",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=None,
        help=(
            "Optional resumable pilot limit over the prefix of each frozen task selection. "
            "This scheduling limit is excluded from the immutable run manifest, so a later "
            "full run can resume the same result directory without repeating pilot rows."
        ),
    )
    parser.add_argument(
        "--sample-start-per-task",
        type=int,
        default=0,
        help=(
            "Start offset into each task's outcome-blind frozen order. Formal "
            "risk-calibrated evaluation uses the untouched suffix after the "
            "calibration prefix."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        repo_root
        / "scripts"
        / "run_hotpot_compact_qa_upgraded_v10.py",
        repo_root / "scripts" / "run_kvpress_natural_qa.py",
        repo_root / "proofkv" / "prefill_selectors.py",
        repo_root / "proofkv" / "physical_kv.py",
        repo_root / "proofkv" / "proofcache" / "model.py",
        repo_root / "proofkv" / "proofcache" / "selector.py",
        repo_root / "proofkv" / "proofcache" / "schema.py",
    ]
    spec = importlib.util.find_spec("kvpress.presses.chunkkv_press")
    if spec is None or spec.origin is None:
        raise ImportError("cannot locate the pinned ChunkKV implementation")
    paths.append(Path(spec.origin).resolve())
    output = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        output[str(path)] = sha256_file(path)
    return output


def directory_file_sha256(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"checkpoint directory is empty: {root}")
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in files
    }


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


def tokenize_parts(
    tokenizer: Any,
    formatted_context: str,
    question_tail: str,
) -> tuple[torch.Tensor, list[tuple[int, int]], int, int]:
    """Mirror KVPress's context-then-question tokenization exactly.

    Tokenizing the concatenated text can change the token at the boundary for
    some SentencePiece tokenizers even when the total length is unchanged.  The
    compact-prefill path must consume the same token IDs as the KVPress
    baselines, so tokenize both pieces independently and shift the question
    offsets into coordinates of their concatenated text.
    """
    context_encoding = tokenizer(
        formatted_context,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    question_encoding = tokenizer(
        question_tail,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    context_offsets = [
        (int(start), int(end))
        for start, end in context_encoding.pop("offset_mapping")[0].tolist()
    ]
    question_offsets = [
        (len(formatted_context) + int(start), len(formatted_context) + int(end))
        for start, end in question_encoding.pop("offset_mapping")[0].tolist()
    ]
    context_ids = context_encoding["input_ids"]
    question_ids = question_encoding["input_ids"]
    input_ids = torch.cat((context_ids, question_ids), dim=1)
    return (
        input_ids,
        context_offsets + question_offsets,
        int(context_ids.shape[1]),
        int(question_ids.shape[1]),
    )


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_qa_f1(prediction: str, answers: list[str]) -> float:
    return max((qa_f1(prediction.lstrip(), str(answer)) for answer in answers), default=0.0)


def risk_feature_value(
    feature: str,
    selection_partition: dict[str, Any],
    capacity: int,
    context_tokens: int,
    question_tokens: int,
) -> float:
    if feature == "proof_core_fraction":
        return float(selection_partition["proof_core_tokens"]) / capacity
    if feature == "coverage_auxiliary_fraction":
        return float(selection_partition["coverage_auxiliary_tokens"]) / capacity
    if feature == "context_log_tokens":
        return math.log1p(context_tokens)
    if feature == "question_budget_fraction":
        return question_tokens / capacity
    return float(selection_partition[feature])


def risk_rule_uses_proof(
    feature_value: float,
    direction: str,
    threshold: float,
) -> bool:
    if not math.isfinite(feature_value) or not math.isfinite(threshold):
        raise ValueError("risk-rule values must be finite")
    if direction == "high":
        return feature_value >= threshold
    if direction == "low":
        return feature_value <= threshold
    raise ValueError(f"unknown risk-rule direction: {direction}")


def validate_risk_controller(
    controller: dict[str, Any],
    model_label: str,
    capacities: list[int],
) -> None:
    if controller.get("status") != "PASS":
        raise ValueError("risk controller did not pass its calibration gate")
    valid_proof_methods = {
        "learned_and_or_recent_aux",
        "learned_and_or_stratified_aux",
        "learned_and_or_graph_aux",
        "learned_and_or_mixed_aux",
    }
    proof_method_map = controller.get("proof_method_by_capacity")
    if not isinstance(proof_method_map, dict):
        raise ValueError("risk controller is missing proof_method_by_capacity")
    for capacity in capacities:
        proof_method = proof_method_map.get(str(capacity))
        if proof_method not in valid_proof_methods:
            raise ValueError(
                f"invalid or missing calibrated proof method for "
                f"{capacity}: {proof_method}"
            )
    rule_map = controller.get("final_risk_rule_by_capacity")
    if not isinstance(rule_map, dict):
        raise ValueError(
            "risk controller is missing final_risk_rule_by_capacity"
        )
    for capacity in capacities:
        rule = rule_map.get(str(capacity))
        if not isinstance(rule, dict):
            raise ValueError(
                f"risk controller is missing rule for {capacity}"
            )
        feature = rule.get("feature")
        if feature not in RISK_FEATURES:
            raise ValueError(
                f"invalid calibrated risk feature for {capacity}: {feature}"
            )
        direction = str(rule.get("direction"))
        threshold = float(rule.get("threshold", float("nan")))
        risk_rule_uses_proof(0.0, direction, threshold)
    fallback_map = controller.get("fallback_by_model_capacity")
    if not isinstance(fallback_map, dict):
        raise ValueError("risk controller is missing fallback_by_model_capacity")
    for capacity in capacities:
        key = f"{model_label}:{capacity}"
        fallback_method = fallback_map.get(key)
        if fallback_method not in BASELINE_METHODS:
            raise ValueError(
                f"invalid or missing physical fallback for {key}: "
                f"{fallback_method}"
            )


def calibrated_proof_method(
    controller: dict[str, Any],
    capacity: int,
) -> str:
    return str(controller["proof_method_by_capacity"][str(capacity)])


def validate_utility_controller(
    controller: dict[str, Any],
    model_label: str,
    capacities: list[int],
) -> None:
    if controller.get("status") != "PASS":
        raise ValueError("utility controller did not pass its calibration gate")
    if controller.get("predecode_only") is not True:
        raise ValueError("utility controller is not declared predecode-only")
    feature_names = controller.get("feature_names")
    if feature_names != list(RISK_FEATURES):
        raise ValueError("utility-controller feature schema differs")
    candidate_methods = controller.get("candidate_methods")
    valid_candidates = {
        "learned_and_or_recent_aux",
        "learned_and_or_stratified_aux",
        "learned_and_or_graph_aux",
        "learned_and_or_mixed_aux",
    }
    if set(candidate_methods or []) != valid_candidates:
        raise ValueError("utility-controller candidate methods differ")
    domain_conditioned = controller.get("domain_conditioned") is True
    policy_field = (
        "final_policy_by_model_task_capacity"
        if domain_conditioned
        else "final_policy_by_model_capacity"
    )
    policies = controller.get(policy_field)
    if not isinstance(policies, dict):
        raise ValueError("utility controller is missing final policies")
    controller_tasks = controller.get("tasks")
    if domain_conditioned and (
        not isinstance(controller_tasks, list)
        or not controller_tasks
        or len(controller_tasks) != len(set(controller_tasks))
    ):
        raise ValueError("domain utility controller has invalid tasks")
    policy_keys = (
        [
            (f"{model_label}:{task}:{capacity}", str(task), capacity)
            for task in controller_tasks
            for capacity in capacities
        ]
        if domain_conditioned
        else [
            (f"{model_label}:{capacity}", None, capacity)
            for capacity in capacities
        ]
    )
    for key, _, _ in policy_keys:
        policy = policies.get(key)
        if not isinstance(policy, dict):
            raise ValueError(f"utility controller is missing policy {key}")
        if policy.get("fallback_method") not in BASELINE_METHODS:
            raise ValueError(f"invalid utility fallback for {key}")
        margin = float(policy.get("utility_margin", float("nan")))
        if not math.isfinite(margin):
            raise ValueError(f"non-finite utility margin for {key}")
        candidate_models = policy.get("candidate_models")
        if not isinstance(candidate_models, dict) or set(
            candidate_models
        ) != valid_candidates:
            raise ValueError(f"invalid utility candidate models for {key}")
        eligible = policy.get(
            "eligible_candidate_methods",
            list(valid_candidates),
        )
        if (
            not isinstance(eligible, list)
            or len(eligible) != len(set(eligible))
            or not set(eligible).issubset(valid_candidates)
        ):
            raise ValueError(f"invalid eligible utility candidates for {key}")
        if domain_conditioned:
            stability = policy.get("candidate_stability")
            stability_eligible = policy.get(
                "stability_eligible_candidate_methods"
            )
            if (
                not isinstance(stability, dict)
                or set(stability) != valid_candidates
                or not isinstance(stability_eligible, list)
                or {
                    method
                    for method, record in stability.items()
                    if isinstance(record, dict)
                    and record.get("eligible") is True
                }
                != set(stability_eligible)
                or not set(eligible).issubset(stability_eligible)
            ):
                raise ValueError(
                    f"domain candidate-stability gate differs for {key}"
                )
        for method, utility_model in candidate_models.items():
            names = utility_model.get("feature_names")
            means = utility_model.get("feature_mean")
            scales = utility_model.get("feature_scale")
            coefficients = utility_model.get("coefficients")
            if names != list(RISK_FEATURES):
                raise ValueError(f"feature schema drift for {key}/{method}")
            if not (
                isinstance(means, list)
                and isinstance(scales, list)
                and isinstance(coefficients, list)
                and len(means)
                == len(scales)
                == len(coefficients)
                == len(names)
            ):
                raise ValueError(f"invalid utility parameters for {key}/{method}")
            values = [
                *[float(value) for value in means],
                *[float(value) for value in scales],
                *[float(value) for value in coefficients],
                float(utility_model.get("intercept", float("nan"))),
            ]
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"non-finite utility parameters for {key}/{method}")
            if any(float(value) <= 0 for value in scales):
                raise ValueError(f"nonpositive utility feature scale for {key}/{method}")


def validate_upgraded_controller(
    controller: dict[str, Any],
    model_label: str,
    capacities: list[int],
) -> None:
    if (
        controller.get("status") != "PASS"
        or controller.get("predecode_only") is not True
        or controller.get("domain_conditioned") is not True
    ):
        raise ValueError("upgraded controller is not a passing predecode policy")
    if controller.get("feature_names") != list(RISK_FEATURES):
        raise ValueError("upgraded-controller feature schema differs")
    architecture = controller.get("upgraded_architecture")
    if (
        not isinstance(architecture, dict)
        or architecture.get("status") != "PASS"
        or architecture.get("predecode_only") is not True
    ):
        raise ValueError("upgraded architecture was not frozen successfully")
    tasks = controller.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("upgraded controller has no task domain")
    policies = controller.get("final_policy_by_model_task_capacity")
    if not isinstance(policies, dict):
        raise ValueError("upgraded controller has no final policies")
    valid_planners = {
        "learned_and_or_recent_aux",
        "learned_and_or_stratified_aux",
        "learned_and_or_graph_aux",
        "learned_and_or_mixed_aux",
        "learned_and_or_lexical_anchor25_aux",
        "learned_and_or_lexical_anchor50_aux",
        "learned_and_or_lexical_anchor75_aux",
        "learned_and_or_control_anchor25_aux",
        "learned_and_or_control_anchor50_aux",
        "learned_and_or_control_anchor75_aux",
    }
    for task in tasks:
        for capacity in capacities:
            key = f"{model_label}:{task}:{capacity}"
            policy = policies.get(key)
            if not isinstance(policy, dict):
                raise ValueError(f"missing upgraded policy {key}")
            if policy.get("fallback_method") not in BASELINE_METHODS:
                raise ValueError(f"invalid upgraded fallback for {key}")
            route = policy.get("upgraded_route")
            eligible = policy.get("eligible_candidate_methods")
            models = policy.get("candidate_models")
            if (
                not isinstance(eligible, list)
                or len(eligible) != len(set(eligible))
                or not set(eligible).issubset(valid_planners)
                or not isinstance(models, dict)
                or not set(eligible).issubset(models)
            ):
                raise ValueError(f"invalid upgraded candidates for {key}")
            if route is None:
                if eligible:
                    raise ValueError(f"fallback-only policy routes candidates: {key}")
            elif not isinstance(route, dict):
                raise ValueError(f"malformed upgraded route for {key}")
            elif route.get("route_type") == "static_anchor":
                if eligible != [route.get("method")]:
                    raise ValueError(f"static route schema differs for {key}")
            elif route.get("route_type") == "query_utility_router":
                if eligible != route.get("eligible_candidate_methods"):
                    raise ValueError(f"query route schema differs for {key}")
            else:
                raise ValueError(f"unknown upgraded route type for {key}")
            margin = float(policy.get("utility_margin", float("nan")))
            if not math.isfinite(margin):
                raise ValueError(f"non-finite upgraded margin for {key}")
            for method in eligible:
                utility_model = models[method]
                names = utility_model.get("feature_names")
                means = utility_model.get("feature_mean")
                scales = utility_model.get("feature_scale")
                coefficients = utility_model.get("coefficients")
                if (
                    names != list(RISK_FEATURES)
                    or not isinstance(means, list)
                    or not isinstance(scales, list)
                    or not isinstance(coefficients, list)
                    or not (
                        len(means)
                        == len(scales)
                        == len(coefficients)
                        == len(names)
                    )
                ):
                    raise ValueError(
                        f"invalid upgraded utility model for {key}/{method}"
                    )
                values = [
                    *[float(value) for value in means],
                    *[float(value) for value in scales],
                    *[float(value) for value in coefficients],
                    float(utility_model.get("intercept", float("nan"))),
                ]
                if (
                    any(not math.isfinite(value) for value in values)
                    or any(float(value) <= 0.0 for value in scales)
                ):
                    raise ValueError(
                        f"non-finite upgraded model for {key}/{method}"
                    )


def utility_prediction(
    utility_model: dict[str, Any],
    selection_partition: dict[str, Any],
    capacity: int,
    context_tokens: int,
    question_tokens: int,
) -> float:
    names = list(utility_model["feature_names"])
    means = [float(value) for value in utility_model["feature_mean"]]
    scales = [float(value) for value in utility_model["feature_scale"]]
    coefficients = [
        float(value) for value in utility_model["coefficients"]
    ]
    output = float(utility_model["intercept"])
    for name, center, scale, coefficient in zip(
        names,
        means,
        scales,
        coefficients,
    ):
        value = risk_feature_value(
            name,
            selection_partition,
            capacity,
            context_tokens,
            question_tokens,
        )
        output += ((value - center) / scale) * coefficient
    if not math.isfinite(output):
        raise ValueError("non-finite predicted proof utility")
    return output


def utility_feature_values(
    selection_partition: dict[str, Any],
    capacity: int,
    context_tokens: int,
    question_tokens: int,
) -> dict[str, float]:
    values = {
        name: risk_feature_value(
            name,
            selection_partition,
            capacity,
            context_tokens,
            question_tokens,
        )
        for name in RISK_FEATURES
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite utility feature value")
    return values


def split_context_spans(context: str, offset: int, task: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for match in re.finditer(r"[^\n.!?]+(?:[.!?]+|$)", context):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        start = match.start() + leading
        end = match.start() + trailing
        if end <= start:
            continue
        # Bound cross-encoder inputs while retaining every character as a selectable span.
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + 900)
            if chunk_end < end:
                split_at = context.rfind(" ", cursor, chunk_end)
                if split_at > cursor + 200:
                    chunk_end = split_at
            text = context[cursor:chunk_end].strip()
            if text:
                text_start = context.find(text, cursor, chunk_end + 1)
                text_end = text_start + len(text)
                span_index = len(spans)
                spans.append(
                    {
                        "title": f"{task}-span-{span_index}",
                        "sent_id": span_index,
                        "text": text,
                        "char_span": (offset + text_start, offset + text_end),
                        "gold": False,
                    }
                )
            cursor = max(chunk_end, cursor + 1)
    if not spans:
        spans.append(
            {
                "title": f"{task}-span-0",
                "sent_id": 0,
                "text": context,
                "char_span": (offset, offset + len(context)),
                "gold": False,
            }
        )
    return spans


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(row["id"], row["method"], row["capacity"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate result keys in {path}")
    return rows


def aggregate(rows: list[dict[str, Any]], tasks: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["method"], str(row["capacity"]))].append(row)
    task_records = []
    for (task, method, capacity), values in sorted(grouped.items()):
        task_records.append(
            {
                "task": task,
                "method": method,
                "capacity": capacity,
                "n": len(values),
                "score": 100 * sum(value["score"] for value in values) / len(values),
                "actual_retained_ratio": sum(value["actual_retained_ratio"] for value in values) / len(values),
                "actual_kv_bytes": sum(value["actual_kv_bytes"] for value in values) / len(values),
                "seconds": sum(value["seconds"] for value in values) / len(values),
                "peak_cuda_memory_gb": max(value["peak_cuda_memory_gb"] for value in values),
            }
        )
    task_lookup = {
        (record["task"], record["method"], record["capacity"]): record
        for record in task_records
    }
    method_capacities = sorted({(record["method"], record["capacity"]) for record in task_records})
    macro_records = []
    for method, capacity in method_capacities:
        records = [task_lookup[(task, method, capacity)] for task in tasks]
        macro_records.append(
            {
                "method": method,
                "capacity": capacity,
                "tasks": len(records),
                "n": sum(record["n"] for record in records),
                "macro_score": sum(record["score"] for record in records) / len(records),
                "actual_retained_ratio": sum(record["actual_retained_ratio"] for record in records) / len(records),
                "seconds": sum(record["seconds"] for record in records) / len(records),
            }
        )
    return task_records, macro_records


def run_pipeline_method(
    pipe: Any,
    tokenizer: Any,
    model: Any,
    context: str,
    question: str,
    answer_prefix: str,
    method: str,
    capacity: int | None,
    max_new_tokens: int,
    kv_bytes_per_token: int,
) -> tuple[str, dict[str, Any]]:
    formatted_context, question_tail = format_parts(tokenizer, context, question, answer_prefix)
    context_tokens = len(tokenizer(formatted_context, add_special_tokens=False)["input_ids"])
    question_tokens = len(tokenizer(question_tail, add_special_tokens=False)["input_ids"])
    total_tokens = context_tokens + question_tokens
    if method == "full":
        press = None
        retained_context_target = context_tokens
    else:
        _, retained_context_target, effective_budget = fixed_budget_targets(
            context_tokens,
            question_tokens,
            budget=1.0,
            budget_tokens=capacity,
        )
        press = make_press(
            method,
            effective_budget,
            exact_kept_tokens=retained_context_target,
        )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cache = DynamicCache()
    started = time.perf_counter()
    prediction = pipe(
        context,
        question=question,
        answer_prefix=answer_prefix,
        press=press,
        cache=cache,
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
    )["answer"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    actual_context_tokens = int(getattr(cache, "_proofcache_compressed_context_length", context_tokens))
    actual_context_bytes = int(
        getattr(
            cache,
            "_proofcache_compressed_context_bytes",
            actual_context_tokens * kv_bytes_per_token,
        )
    )
    actual_bytes = actual_context_bytes + question_tokens * kv_bytes_per_token
    metadata = {
        "context_tokens": context_tokens,
        "question_tokens": question_tokens,
        "prompt_tokens": total_tokens,
        "retained_context_tokens_target": retained_context_target,
        "actual_retained_tokens": actual_bytes / kv_bytes_per_token,
        "actual_retained_ratio": actual_bytes / kv_bytes_per_token / total_tokens,
        "actual_kv_bytes": actual_bytes,
        "seconds": seconds,
        "selection_seconds": 0.0,
        "generation_seconds": seconds,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    del cache
    _release_cuda()
    return prediction, metadata


def main() -> None:
    args = parse_args()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = set(methods) - METHODS
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    controller_methods = {
        method
        for method in methods
        if method
        in {
            "risk_calibrated_proof_coverage",
            "utility_calibrated_proof_coverage",
            "confirmation_stable_proofkv_v10",
        }
    }
    if len(controller_methods) > 1:
        raise ValueError("only one calibrated controller may be evaluated per run")
    uses_controller = bool(controller_methods)
    if uses_controller != bool(args.risk_controller_config):
        raise ValueError(
            "--risk-controller-config must be provided exactly when "
            "a calibrated proof controller is requested"
        )
    risk_controller = (
        json.loads(Path(args.risk_controller_config).read_text())
        if args.risk_controller_config
        else None
    )
    capacities = [int(value) for value in args.capacities.split(",") if value.strip()]
    if capacities != sorted(set(capacities)) or any(value <= 4 for value in capacities):
        raise ValueError("capacities must be unique, sorted, and exceed the four-token sink")
    if risk_controller is not None:
        if "risk_calibrated_proof_coverage" in controller_methods:
            validate_risk_controller(
                risk_controller,
                args.model_label,
                capacities,
            )
        elif "utility_calibrated_proof_coverage" in controller_methods:
            validate_utility_controller(
                risk_controller,
                args.model_label,
                capacities,
            )
        else:
            validate_upgraded_controller(
                risk_controller,
                args.model_label,
                capacities,
            )
    selection_path = Path(args.selection_manifest)
    selection = json.loads(selection_path.read_text())
    if (
        risk_controller is not None
        and risk_controller.get("domain_conditioned") is True
        and risk_controller.get("tasks") != selection["tasks"]
    ):
        raise ValueError(
            "domain utility-controller tasks differ from frozen selection"
        )
    if selection["capacities"] != capacities:
        raise ValueError("capacities differ from the frozen selection protocol")
    if selection["methods"] != [
        "full",
        "streamingllm",
        "snapkv",
        "pyramidkv",
        "chunkkv_snapkv",
        "entity_graph_adaptive",
        "learned_topk_paths",
        "learned_and_or_aux",
    ]:
        raise ValueError("frozen protocol method list is unexpected")
    tasks = list(selection["tasks"])
    if args.tasks is not None:
        requested_tasks = [
            value.strip()
            for value in str(args.tasks).split(",")
            if value.strip()
        ]
        if (
            not requested_tasks
            or len(requested_tasks) != len(set(requested_tasks))
            or not set(requested_tasks).issubset(tasks)
        ):
            raise ValueError("tasks must be a unique subset of the frozen tasks")
        tasks = requested_tasks
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    frozen_n_per_task = int(selection["max_samples_per_task"])
    execution_n_per_task = (
        frozen_n_per_task
        if args.max_samples_per_task is None
        else int(args.max_samples_per_task)
    )
    sample_start_per_task = int(args.sample_start_per_task)
    if (
        sample_start_per_task < 0
        or execution_n_per_task < 1
        or sample_start_per_task + execution_n_per_task > frozen_n_per_task
    ):
        raise ValueError(
            "sample slice must be nonempty and contained in the frozen "
            f"[0, {frozen_n_per_task}) order"
        )
    is_complete_declared_slice = (
        sample_start_per_task == 0
        and execution_n_per_task == frozen_n_per_task
    ) or (
        sample_start_per_task > 0
        and sample_start_per_task + execution_n_per_task == frozen_n_per_task
    )
    metrics_path = (
        out_dir / "metrics.json"
        if is_complete_declared_slice
        else (
            out_dir / f"pilot_metrics_n{execution_n_per_task}.json"
            if sample_start_per_task == 0
            else out_dir
            / f"pilot_metrics_s{sample_start_per_task}_n{execution_n_per_task}.json"
        )
    )
    manifest_path = out_dir / "run_manifest.json"
    if metrics_path.exists():
        raise FileExistsError(f"completed output exists: {metrics_path}")
    run_manifest = {
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"out_dir", "max_samples_per_task"}
            and not (key == "sample_start_per_task" and int(value) == 0)
            and not (key == "risk_controller_config" and value is None)
        },
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_protocol": selection["protocol"],
        "implementation_sha256": implementation_sha256(),
        "scorer_checkpoint_sha256": directory_file_sha256(
            Path(args.scorer_checkpoint)
        ),
    }
    if args.risk_controller_config:
        run_manifest["risk_controller_sha256"] = sha256_file(
            Path(args.risk_controller_config)
        )
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != run_manifest:
            raise ValueError("run manifest differs from this resumed evaluation")
    else:
        if results_path.exists():
            raise ValueError("cannot resume results without a run manifest")
        manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")

    model, tokenizer = load_model_tokenizer(
        args.model,
        dtype="bfloat16",
        attn_implementation=args.attn_implementation,
    )
    from kvpress import KVPressTextGenerationPipeline

    pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer, device=model.device)
    scorer, scorer_tokenizer = TypedRelationScorer.load_checkpoint(
        args.scorer_checkpoint,
        device=next(model.parameters()).device,
    )
    learned_bundle = ("cross", scorer, scorer_tokenizer)
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    kv_bytes_per_token = (
        model.config.num_hidden_layers * model.config.num_key_value_heads * head_dim * 2 * 2
    )
    rows = load_existing_rows(results_path)
    completed = {(row["id"], row["method"], row["capacity"]) for row in rows}
    expected_per_task = execution_n_per_task * (
        int("full" in methods) + len(capacities) * len([method for method in methods if method != "full"])
    )

    for task in tasks:
        task_selection = selection["selections"][task]
        data_path = Path(task_selection["data_file"])
        if sha256_file(data_path) != task_selection["data_sha256"]:
            raise ValueError(f"data hash mismatch for {task}")
        source_rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
        for selected in tqdm(
            task_selection["selected"][
                sample_start_per_task : sample_start_per_task
                + execution_n_per_task
            ],
            desc=f"{args.model_label}:{task}",
        ):
            raw = source_rows[int(selected["index"])]
            source_id = row_id(task, int(selected["index"]), raw)
            if source_id != selected["id"]:
                raise ValueError(f"row identity mismatch for {source_id}")
            required_keys = set()
            if "full" in methods:
                required_keys.add((source_id, "full", "full"))
            required_keys.update(
                (source_id, method, capacity)
                for capacity in capacities
                for method in methods
                if method != "full"
            )
            if required_keys.issubset(completed):
                continue
            context, question, answer_prefix = official_fields(task, raw)
            formatted_context, question_tail = format_parts(tokenizer, context, question, answer_prefix)
            prompt = formatted_context + question_tail
            input_ids, offsets, encoded_context_tokens, encoded_question_tokens = tokenize_parts(
                tokenizer,
                formatted_context,
                question_tail,
            )
            input_ids = input_ids.to(next(model.parameters()).device)
            expected_prompt_tokens = int(selected["token_counts"][args.model_label]["prompt"])
            if len(offsets) != expected_prompt_tokens:
                raise ValueError(
                    f"prompt length drift for {source_id}: {len(offsets)} != {expected_prompt_tokens}"
                )
            expected_question_tokens = int(
                selected["token_counts"][args.model_label]["question"]
            )
            if encoded_question_tokens != expected_question_tokens:
                raise ValueError(
                    f"question length drift for {source_id}: "
                    f"{encoded_question_tokens} != {expected_question_tokens}"
                )
            if encoded_context_tokens + encoded_question_tokens != len(offsets):
                raise ValueError(f"split tokenization length mismatch for {source_id}")
            context_offset = formatted_context.find(context)
            if context_offset < 0:
                raise ValueError("chat template did not preserve LongBench context")
            sentences = split_context_spans(context, context_offset, task)
            source = {
                "id": source_id,
                "question": str(raw["input"]),
                "type": task,
                "answer": str(raw["answers"][0]),
            }
            answers = [str(value) for value in raw["answers"]]
            max_new_tokens = MAX_NEW_TOKENS[task]
            if "full" in methods and (source_id, "full", "full") not in completed:
                prediction, metadata = run_pipeline_method(
                    pipe,
                    tokenizer,
                    model,
                    context,
                    question,
                    answer_prefix,
                    "full",
                    None,
                    max_new_tokens,
                    kv_bytes_per_token,
                )
                row = {
                    "id": source_id,
                    "task": task,
                    "method": "full",
                    "capacity": "full",
                    "answers": answers,
                    "prediction": prediction,
                    "score": best_qa_f1(prediction, answers),
                    **metadata,
                }
                rows.append(row)
                append_jsonl(results_path, row)
                completed.add((source_id, "full", "full"))

            learned_overlay_cache: dict[str, tuple[Any, Any]] = {}
            coverage_plan_cache: dict[
                tuple[int, str],
                tuple[Any, dict[str, Any], float],
            ] = {}
            for capacity in capacities:
                for method in methods:
                    if method == "full" or (source_id, method, capacity) in completed:
                        continue
                    if method in BASELINE_METHODS:
                        prediction, metadata = run_pipeline_method(
                            pipe,
                            tokenizer,
                            model,
                            context,
                            question,
                            answer_prefix,
                            method,
                            capacity,
                            max_new_tokens,
                            kv_bytes_per_token,
                        )
                    else:
                        effective_budget = min(1.0, capacity / len(offsets))
                        use_proof_route = True
                        fallback_method = None
                        if method in {
                            "utility_calibrated_proof_coverage",
                            "confirmation_stable_proofkv_v10",
                        }:
                            assert risk_controller is not None
                            domain_conditioned = (
                                risk_controller.get("domain_conditioned")
                                is True
                            )
                            policy_key = (
                                f"{args.model_label}:{task}:{capacity}"
                                if domain_conditioned
                                else f"{args.model_label}:{capacity}"
                            )
                            policy_field = (
                                "final_policy_by_model_task_capacity"
                                if domain_conditioned
                                else "final_policy_by_model_capacity"
                            )
                            policy = risk_controller[
                                policy_field
                            ][policy_key]
                            utility_candidates = []
                            selection_seconds = 0.0
                            eligible_candidates = policy.get(
                                "eligible_candidate_methods",
                                list(policy["candidate_models"]),
                            )
                            for candidate_method in sorted(
                                eligible_candidates
                            ):
                                cache_key = (capacity, candidate_method)
                                if cache_key not in coverage_plan_cache:
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                    candidate_started = time.perf_counter()
                                    candidate_partition: dict[str, Any] = {}
                                    candidate_plan, _, _ = make_plan(
                                        candidate_method,
                                        source,
                                        prompt,
                                        question_tail,
                                        sentences,
                                        offsets,
                                        effective_budget,
                                        learned_bundle,
                                        learned_dense_core_count=(
                                            args.learned_dense_core_count
                                        ),
                                        learned_edge_batch_size=(
                                            args.learned_edge_batch_size
                                        ),
                                        learned_overlay_cache=(
                                            learned_overlay_cache
                                        ),
                                        selection_metadata_out=(
                                            candidate_partition
                                        ),
                                    )
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                    candidate_seconds = (
                                        time.perf_counter()
                                        - candidate_started
                                    )
                                    coverage_plan_cache[cache_key] = (
                                        candidate_plan,
                                        dict(candidate_partition),
                                        candidate_seconds,
                                    )
                                (
                                    candidate_plan,
                                    candidate_partition,
                                    candidate_seconds,
                                ) = coverage_plan_cache[cache_key]
                                selection_seconds += candidate_seconds
                                predicted_utility = utility_prediction(
                                    policy["candidate_models"][
                                        candidate_method
                                    ],
                                    candidate_partition,
                                    capacity,
                                    encoded_context_tokens,
                                    encoded_question_tokens,
                                )
                                candidate_feature_values = (
                                    utility_feature_values(
                                        candidate_partition,
                                        capacity,
                                        encoded_context_tokens,
                                        encoded_question_tokens,
                                    )
                                )
                                utility_candidates.append(
                                    {
                                        "method": candidate_method,
                                        "predicted_utility": (
                                            predicted_utility
                                        ),
                                        "plan": candidate_plan,
                                        "partition": candidate_partition,
                                        "feature_values": (
                                            candidate_feature_values
                                        ),
                                    }
                                )
                            margin = float(policy["utility_margin"])
                            fallback_method = str(
                                policy["fallback_method"]
                            )
                            if utility_candidates:
                                chosen_candidate = max(
                                    utility_candidates,
                                    key=lambda value: (
                                        float(
                                            value[
                                                "predicted_utility"
                                            ]
                                        ),
                                        str(value["method"]),
                                    ),
                                )
                                planner_method = str(
                                    chosen_candidate["method"]
                                )
                                plan = chosen_candidate["plan"]
                                selection_partition = dict(
                                    chosen_candidate["partition"]
                                )
                                selected_prediction = float(
                                    chosen_candidate[
                                        "predicted_utility"
                                    ]
                                )
                                use_proof_route = (
                                    selected_prediction >= margin
                                )
                            else:
                                planner_method = None
                                selection_partition = {}
                                selected_prediction = None
                                use_proof_route = False
                            selection_partition.update(
                                {
                                    "utility_candidate_predictions": {
                                        str(value["method"]): float(
                                            value["predicted_utility"]
                                        )
                                        for value in utility_candidates
                                    },
                                    "utility_candidate_feature_values": {
                                        str(value["method"]): dict(
                                            value["feature_values"]
                                        )
                                        for value in utility_candidates
                                    },
                                    "utility_selected_method": (
                                        planner_method
                                    ),
                                    "utility_selected_prediction": (
                                        selected_prediction
                                    ),
                                    "utility_margin": margin,
                                    "utility_route": (
                                        "proof_coverage"
                                        if use_proof_route
                                        else "physical_fallback"
                                    ),
                                    "utility_fallback_method": (
                                        fallback_method
                                    ),
                                }
                            )
                        else:
                            planner_method = method
                            if method == "risk_calibrated_proof_coverage":
                                assert risk_controller is not None
                                planner_method = calibrated_proof_method(
                                    risk_controller,
                                    capacity,
                                )
                            cache_key = (capacity, planner_method)
                            if cache_key in coverage_plan_cache:
                                (
                                    plan,
                                    cached_partition,
                                    selection_seconds,
                                ) = coverage_plan_cache[cache_key]
                                selection_partition = dict(
                                    cached_partition
                                )
                            else:
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                selection_started = time.perf_counter()
                                selection_partition = {}
                                plan, _, _ = make_plan(
                                    planner_method,
                                    source,
                                    prompt,
                                    question_tail,
                                    sentences,
                                    offsets,
                                    effective_budget,
                                    (
                                        learned_bundle
                                        if planner_method.startswith(
                                            "learned_"
                                        )
                                        else None
                                    ),
                                    learned_dense_core_count=(
                                        args.learned_dense_core_count
                                    ),
                                    learned_edge_batch_size=(
                                        args.learned_edge_batch_size
                                    ),
                                    learned_overlay_cache=(
                                        learned_overlay_cache
                                    ),
                                    selection_metadata_out=(
                                        selection_partition
                                    ),
                                )
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                selection_seconds = (
                                    time.perf_counter()
                                    - selection_started
                                )
                                if (
                                    planner_method
                                    in {
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
                                ):
                                    coverage_plan_cache[cache_key] = (
                                        plan,
                                        dict(selection_partition),
                                        selection_seconds,
                                    )
                        if method == "risk_calibrated_proof_coverage":
                            assert risk_controller is not None
                            rule = risk_controller[
                                "final_risk_rule_by_capacity"
                            ][str(capacity)]
                            feature = str(rule["feature"])
                            feature_value = risk_feature_value(
                                feature,
                                selection_partition,
                                capacity,
                                encoded_context_tokens,
                                encoded_question_tokens,
                            )
                            direction = str(rule["direction"])
                            threshold = float(rule["threshold"])
                            use_proof_route = risk_rule_uses_proof(
                                feature_value,
                                direction,
                                threshold,
                            )
                            fallback_key = f"{args.model_label}:{capacity}"
                            fallback_method = str(
                                risk_controller[
                                    "fallback_by_model_capacity"
                                ][fallback_key]
                            )
                            if fallback_method not in BASELINE_METHODS:
                                raise ValueError(
                                    f"invalid physical fallback: {fallback_method}"
                                )
                            selection_partition.update(
                                {
                                    "risk_feature": feature,
                                    "risk_feature_value": feature_value,
                                    "risk_direction": direction,
                                    "risk_threshold": threshold,
                                    "risk_route": (
                                        "proof_coverage"
                                        if use_proof_route
                                        else "physical_fallback"
                                    ),
                                    "risk_fallback_method": fallback_method,
                                    "risk_proof_method": planner_method,
                                }
                            )

                        if use_proof_route:
                            reusable_source = (
                                next(
                                    (
                                        row
                                        for row in rows
                                        if row["id"] == source_id
                                        and row["task"] == task
                                        and row["method"]
                                        == planner_method
                                        and str(row["capacity"])
                                        == str(capacity)
                                    ),
                                    None,
                                )
                                if method
                                in {
                                    "utility_calibrated_proof_coverage",
                                    "confirmation_stable_proofkv_v10",
                                }
                                else None
                            )
                            if reusable_source is not None:
                                prediction = str(
                                    reusable_source["prediction"]
                                )
                                generation_seconds = float(
                                    reusable_source[
                                        "generation_seconds"
                                    ]
                                )
                                metadata = {
                                    key: value
                                    for key, value in reusable_source.items()
                                    if key
                                    not in {
                                        "id",
                                        "task",
                                        "method",
                                        "capacity",
                                        "answers",
                                        "prediction",
                                        "score",
                                        "selection_partition",
                                        "warnings",
                                    }
                                }
                                metadata.update(
                                    {
                                        "seconds": selection_seconds
                                        + generation_seconds,
                                        "selection_seconds": (
                                            selection_seconds
                                        ),
                                        "generation_seconds": (
                                            generation_seconds
                                        ),
                                        "selection_partition": (
                                            selection_partition
                                        ),
                                        "warnings": [],
                                    }
                                )
                            else:
                                torch.cuda.reset_peak_memory_stats()
                                generation_started = time.perf_counter()
                                (
                                    prediction,
                                    cache,
                                    _,
                                    warnings,
                                    _,
                                ) = production_compact_decode(
                                    model,
                                    tokenizer,
                                    input_ids,
                                    plan.retained_indices,
                                    input_ids.shape[1],
                                    max_new_tokens,
                                    512,
                                )
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                generation_seconds = (
                                    time.perf_counter()
                                    - generation_started
                                )
                                actual_tokens = len(
                                    plan.retained_indices
                                )
                                metadata = {
                                    "context_tokens": (
                                        encoded_context_tokens
                                    ),
                                    "question_tokens": (
                                        encoded_question_tokens
                                    ),
                                    "prompt_tokens": len(offsets),
                                    "retained_context_tokens_target": None,
                                    "actual_retained_tokens": actual_tokens,
                                    "actual_retained_ratio": (
                                        actual_tokens / len(offsets)
                                    ),
                                    "actual_kv_bytes": (
                                        actual_tokens
                                        * kv_bytes_per_token
                                    ),
                                    "seconds": (
                                        selection_seconds
                                        + generation_seconds
                                    ),
                                    "selection_seconds": (
                                        selection_seconds
                                    ),
                                    "generation_seconds": (
                                        generation_seconds
                                    ),
                                    "peak_cuda_memory_gb": (
                                        torch.cuda.max_memory_allocated()
                                        / 1024**3
                                    ),
                                    "selection_partition": (
                                        selection_partition
                                    ),
                                    "warnings": warnings,
                                }
                                del cache
                                _release_cuda()
                        else:
                            assert fallback_method is not None
                            reusable_fallback = next(
                                (
                                    row
                                    for row in rows
                                    if row["id"] == source_id
                                    and row["task"] == task
                                    and row["method"]
                                    == fallback_method
                                    and str(row["capacity"])
                                    == str(capacity)
                                ),
                                None,
                            )
                            if reusable_fallback is None:
                                prediction, metadata = (
                                    run_pipeline_method(
                                        pipe,
                                        tokenizer,
                                        model,
                                        context,
                                        question,
                                        answer_prefix,
                                        fallback_method,
                                        capacity,
                                        max_new_tokens,
                                        kv_bytes_per_token,
                                    )
                                )
                            else:
                                prediction = str(
                                    reusable_fallback["prediction"]
                                )
                                metadata = {
                                    key: value
                                    for key, value in reusable_fallback.items()
                                    if key
                                    not in {
                                        "id",
                                        "task",
                                        "method",
                                        "capacity",
                                        "answers",
                                        "prediction",
                                        "score",
                                        "selection_partition",
                                        "warnings",
                                    }
                                }
                            physical_selection_seconds = float(
                                metadata["selection_seconds"]
                            )
                            physical_generation_seconds = float(
                                metadata["generation_seconds"]
                            )
                            physical_seconds = float(metadata["seconds"])
                            if not math.isclose(
                                physical_seconds,
                                (
                                    physical_selection_seconds
                                    + physical_generation_seconds
                                ),
                                rel_tol=1.0e-6,
                                abs_tol=1.0e-6,
                            ):
                                raise ValueError(
                                    "physical fallback timing fields do not sum"
                                )
                            metadata.update(
                                {
                                    "seconds": selection_seconds
                                    + physical_seconds,
                                    "selection_seconds": (
                                        selection_seconds
                                        + physical_selection_seconds
                                    ),
                                    "generation_seconds": (
                                        physical_generation_seconds
                                    ),
                                    "selection_partition": selection_partition,
                                    "warnings": [],
                                }
                            )
                    if not math.isclose(
                        float(metadata["seconds"]),
                        (
                            float(metadata["selection_seconds"])
                            + float(metadata["generation_seconds"])
                        ),
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-6,
                    ):
                        raise ValueError(
                            f"{method} timing fields do not sum on {source_id}"
                        )
                    if float(metadata["actual_retained_tokens"]) > capacity + 1e-6:
                        raise ValueError(
                            f"{method} exceeded capacity {capacity} on {source_id}: "
                            f"{metadata['actual_retained_tokens']}"
                        )
                    row = {
                        "id": source_id,
                        "task": task,
                        "method": method,
                        "capacity": capacity,
                        "answers": answers,
                        "prediction": prediction,
                        "score": best_qa_f1(prediction, answers),
                        **metadata,
                    }
                    rows.append(row)
                    append_jsonl(results_path, row)
                    completed.add((source_id, method, capacity))

        task_rows = [row for row in rows if row["task"] == task]
        if len(task_rows) != expected_per_task:
            raise RuntimeError(f"incomplete task matrix for {task}: {len(task_rows)} != {expected_per_task}")

    task_records, macro_records = aggregate(rows, tasks)
    payload = {
        "protocol": selection["protocol"],
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "model": args.model_label,
        "model_path": args.model,
        "model_revision": args.model_revision,
        "tasks": tasks,
        "n_per_task": execution_n_per_task,
        "selection_start_per_task": sample_start_per_task,
        "frozen_n_per_task": frozen_n_per_task,
        "max_input_tokens": selection["max_input_tokens"],
        "truncated_inputs": 0,
        "tokenization_contract": "kvpress_split_context_question_token_ids_v1",
        "methods": methods,
        "capacities": capacities,
        "capacity_scope": "total prefill KV including the uncompressed question tail",
        "kv_bytes_per_token_bf16": kv_bytes_per_token,
        "attn_implementation": args.attn_implementation,
        "task_records": task_records,
        "macro_records": macro_records,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "macro_records": macro_records}))


if __name__ == "__main__":
    main()
