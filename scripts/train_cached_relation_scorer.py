#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from train_typed_relation_scorer import ItemDataset, collate, load_training_examples, make_items
except ImportError:
    from scripts.train_typed_relation_scorer import ItemDataset, collate, load_training_examples, make_items

from proofkv.proofcache.model import CachedTypedRelationScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train query-agnostic cached-span typed relation scorer.")
    parser.add_argument("--encoder", default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument("--hotpot-samples", type=int, default=12000)
    parser.add_argument("--two-wiki-samples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hard-negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lambda-proof", type=float, default=0.5)
    parser.add_argument("--lambda-budget", type=float, default=0.05)
    parser.add_argument("--lambda-calibration", type=float, default=0.2)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def encode(model, tokenizer, texts, device, max_length):
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return model.encode({key: value.to(device) for key, value in encoded.items()})


def batch_loss(model, tokenizer, batch, device, args):
    root_items = [item for item in batch if item.task == "root"]
    edge_items = [item for item in batch if item.task == "edge"]
    losses = []
    if root_items:
        query = encode(model, tokenizer, [item.question for item in root_items], device, 96)
        span = encode(model, tokenizer, [item.source for item in root_items], device, 192)
        logits = model.root_logits_from_embeddings(query, span)
        labels = torch.tensor([item.label for item in root_items], dtype=torch.float32, device=device)
        probabilities = torch.sigmoid(logits)
        root_loss = F.binary_cross_entropy_with_logits(logits, labels)
        positive = labels > 0.5
        proof_loss = -torch.log(probabilities[positive].clamp_min(1e-6)).mean() if positive.any() else logits.sum() * 0
        negative = ~positive
        costs = torch.tensor([item.token_cost for item in root_items], dtype=torch.float32, device=device)
        budget_loss = (probabilities[negative] * costs[negative]).mean() if negative.any() else logits.sum() * 0
        calibration_loss = F.mse_loss(probabilities, labels)
        losses.append(
            root_loss + args.lambda_proof * proof_loss
            + args.lambda_budget * budget_loss / 128.0
            + args.lambda_calibration * calibration_loss
        )
    if edge_items:
        query = encode(model, tokenizer, [item.question for item in edge_items], device, 96)
        source = encode(model, tokenizer, [item.source for item in edge_items], device, 192)
        target = encode(model, tokenizer, [item.target for item in edge_items], device, 192)
        logits = model.relation_logits_from_embeddings(query, source, target)
        labels = torch.tensor([item.label for item in edge_items], dtype=torch.long, device=device)
        probabilities = torch.softmax(logits, dim=-1)
        edge_loss = F.cross_entropy(logits, labels)
        positive = labels > 0
        proof_loss = -torch.log(probabilities[positive, labels[positive]].clamp_min(1e-6)).mean() if positive.any() else logits.sum() * 0
        calibration_loss = F.mse_loss(1.0 - probabilities[:, 0], positive.float())
        losses.append(edge_loss + args.lambda_proof * proof_loss + args.lambda_calibration * calibration_loss)
    return sum(losses) / max(1, len(losses))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "proofcache_cached_scorer.json").exists():
        raise FileExistsError(output_dir)
    examples = load_training_examples(args)
    rng = random.Random(args.seed + 3)
    items = [item for example in examples for item in make_items(example, args.hard_negatives, rng)]
    rng.shuffle(items)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    model = CachedTypedRelationScorer(args.encoder, projection_dim=args.projection_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loader = DataLoader(ItemDataset(items), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, tokenizer, batch, device, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        record = {"epoch": epoch + 1, "loss": total / max(1, step), "steps": step}
        history.append(record)
        print(json.dumps(record), flush=True)
    elapsed = time.perf_counter() - started
    counts = Counter((item.task, item.negative_type) for item in items)
    metadata = {
        "architecture": "query-agnostic cached span bi-encoder plus typed relation MLP",
        "encoder": args.encoder,
        "projection_dim": args.projection_dim,
        "hotpot_samples": args.hotpot_samples,
        "two_wiki_samples": args.two_wiki_samples,
        "epochs": args.epochs,
        "seed": args.seed,
        "training_items": len(items),
        "negative_counts": {f"{task}:{kind}": value for (task, kind), value in sorted(counts.items())},
        "history": history,
        "elapsed_seconds": elapsed,
    }
    model.save_checkpoint(output_dir, tokenizer, metadata)
    (output_dir / "train_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "items": len(items), "seconds": elapsed}))


if __name__ == "__main__":
    main()
