#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset

from proofkv.proofcache.datasets import from_hotpot
from proofkv.proofcache.model import CachedTypedRelationScorer, TypedRelationScorer
from proofkv.proofcache.topology import TopologyConfig, make_topology_example
try:
    from scripts.train_typed_relation_scorer import TrainItem, make_items
except ImportError:  # direct execution from scripts/
    from train_typed_relation_scorer import TrainItem, make_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out root/terminal/directed-edge diagnostics.")
    parser.add_argument("--scorer-checkpoint")
    parser.add_argument("--cached-scorer-checkpoint")
    parser.add_argument("--hotpot-samples", type=int, default=256)
    parser.add_argument("--synthetic-samples", type=int, default=256)
    parser.add_argument("--max-items-per-group", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda row: row[0])
    positive_rank_sum = 0.0
    position = 1
    idx = 0
    while idx < len(ordered):
        end = idx + 1
        while end < len(ordered) and ordered[end][0] == ordered[idx][0]:
            end += 1
        average_rank = (position + position + (end - idx) - 1) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[idx:end])
        position += end - idx
        idx = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def classification_metrics(rows: list[dict]) -> dict:
    labels = [int(row["label"] > 0) for row in rows]
    predictions = [int(row["prediction"] > 0) for row in rows]
    tp = sum(y and p for y, p in zip(labels, predictions))
    fp = sum((not y) and p for y, p in zip(labels, predictions))
    fn = sum(y and (not p) for y, p in zip(labels, predictions))
    tn = len(rows) - tp - fp - fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    payload = {
        "n": len(rows),
        "positive_rate": sum(labels) / max(1, len(labels)),
        "accuracy": (tp + tn) / max(1, len(rows)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "auroc": auc(labels, [float(row["score"]) for row in rows]),
        "false_positive_rate": fp / max(1, fp + tn),
    }
    positive_rows = [row for row in rows if row["label"] > 0]
    if positive_rows and "typed_prediction" in positive_rows[0]:
        payload["typed_positive_accuracy"] = sum(
            row["typed_prediction"] == row["label"] for row in positive_rows
        ) / len(positive_rows)
    return payload


def heldout_items(args: argparse.Namespace) -> dict[str, list[TrainItem]]:
    rng = random.Random(args.seed)
    hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    hotpot_indices = list(range(len(hotpot)))
    rng.shuffle(hotpot_indices)
    natural = []
    for position, idx in enumerate(hotpot_indices[: args.hotpot_samples]):
        natural.extend(make_items(from_hotpot(hotpot[idx], "validation"), 6, random.Random(args.seed + position)))
    synthetic = []
    for idx in range(args.synthetic_samples):
        variant = idx % 9
        example = make_topology_example(
            TopologyConfig(
                hops=(1, 2, 3, 4, 5, 6, 8)[idx % 7],
                branch_factor=(1, 2, 4)[(idx // 7) % 3],
                alternative_proofs=(1, 2, 4)[(idx // 21) % 3],
                distractor_chains=(0, 4, 16)[(idx // 63) % 3],
                evidence_position=("start", "middle", "end", "mixed")[(idx // 113) % 4],
                entity_collisions=variant == 0,
                fact_reorder=variant == 1,
                paraphrase=variant == 2,
                marker_removal=variant == 3,
                contradiction=variant == 4,
                cycle=variant == 5,
                broken_middle=variant in {6, 7, 8},
            ),
            seed=args.seed,
            sample_index=100_000 + idx,
        )
        synthetic.extend(make_items(example, 6, random.Random(args.seed * 1009 + idx)))
    return {"hotpot_validation": natural, "neutral_topology_ood": synthetic}


@torch.inference_mode()
def score_items(
    model, tokenizer, items: list[TrainItem], args: argparse.Namespace, cached_mode: bool
) -> list[dict]:
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        task = batch[0].task
        if cached_mode:
            def encode(texts, max_length):
                encoded = tokenizer(
                    texts, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                )
                return model.encode({key: value.to(device) for key, value in encoded.items()})

            query = encode([item.question for item in batch], 96)
            source = encode([item.source for item in batch], 192)
            if task == "root":
                logits = model.root_logits_from_embeddings(query, source)
            elif task == "terminal":
                logits = model.terminal_logits_from_embeddings(query, source)
            else:
                target = encode([item.target for item in batch], 192)
                logits = model.relation_logits_from_embeddings(query, source, target)
        else:
            encoded = tokenizer(
                [item.text for item in batch], padding=True, truncation=True,
                max_length=320, return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if task == "root":
                logits = model.root_logits(encoded)
            elif task == "terminal":
                logits = model.terminal_logits(encoded)
            else:
                logits = model.relation_logits(encoded)
        if task in {"root", "terminal"}:
            probabilities = torch.sigmoid(logits).cpu().tolist()
            typed = [int(value >= 0.5) for value in probabilities]
        else:
            distribution = torch.softmax(logits, dim=-1).cpu()
            probabilities = (1.0 - distribution[:, 0]).tolist()
            typed = distribution.argmax(-1).tolist()
        rows.extend(
            {
                "task": item.task,
                "negative_type": item.negative_type,
                "label": item.label,
                "score": float(score),
                "prediction": int(prediction > 0),
                "typed_prediction": int(prediction),
            }
            for item, score, prediction in zip(batch, probabilities, typed)
        )
    return rows


def main() -> None:
    args = parse_args()
    if bool(args.scorer_checkpoint) == bool(args.cached_scorer_checkpoint):
        raise ValueError("provide exactly one scorer checkpoint")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "metrics.json").exists():
        raise FileExistsError(out_dir)
    if args.cached_scorer_checkpoint:
        model, tokenizer = CachedTypedRelationScorer.load_checkpoint(
            args.cached_scorer_checkpoint, device=args.device
        )
        cached_mode = True
    else:
        model, tokenizer = TypedRelationScorer.load_checkpoint(
            args.scorer_checkpoint, device=args.device
        )
        cached_mode = False
    all_metrics = {}
    rng = random.Random(args.seed + 7)
    for dataset_name, items in heldout_items(args).items():
        grouped_items: dict[tuple[str, str], list[TrainItem]] = defaultdict(list)
        for item in items:
            grouped_items[(item.task, item.negative_type)].append(item)
        sampled = []
        for values in grouped_items.values():
            rng.shuffle(values)
            sampled.extend(values[: args.max_items_per_group])
        # Homogeneous task batches avoid invoking the wrong model head.
        sampled.sort(key=lambda item: item.task)
        rows = []
        for task in ("root", "terminal", "edge"):
            rows.extend(
                score_items(
                    model,
                    tokenizer,
                    [item for item in sampled if item.task == task],
                    args,
                    cached_mode,
                )
            )
        metrics = {task: classification_metrics([row for row in rows if row["task"] == task]) for task in ("root", "terminal", "edge")}
        for negative_type in sorted({row["negative_type"] for row in rows if row["label"] == 0}):
            negatives = [row for row in rows if row["negative_type"] == negative_type]
            metrics[f"negative:{negative_type}"] = {
                "n": len(negatives),
                "false_positive_rate": sum(row["prediction"] > 0 for row in negatives) / max(1, len(negatives)),
                "mean_non_none_score": sum(row["score"] for row in negatives) / max(1, len(negatives)),
            }
        all_metrics[dataset_name] = metrics
    payload = {
        "checkpoint": args.cached_scorer_checkpoint or args.scorer_checkpoint,
        "cached_mode": cached_mode,
        "seed": args.seed,
        "datasets": all_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "datasets": list(all_metrics)}))


if __name__ == "__main__":
    main()
