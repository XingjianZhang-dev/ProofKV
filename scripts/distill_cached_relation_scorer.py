#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from train_typed_relation_scorer import (
        ItemDataset,
        collate,
        load_training_examples,
        make_items_job,
    )
except ImportError:
    from scripts.train_typed_relation_scorer import (
        ItemDataset,
        collate,
        load_training_examples,
        make_items_job,
    )

from proofkv.proofcache.model import (
    CachedTypedRelationScorer,
    LateInteractionTypedRelationScorer,
    TypedRelationScorer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill cross-encoder proof scorer into cached bi-encoder.")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-encoder", default="prajjwal1/bert-tiny")
    parser.add_argument(
        "--student-architecture",
        choices=("pooled", "late_interaction"),
        default="pooled",
    )
    parser.add_argument("--num-vectors", type=int, default=8)
    parser.add_argument("--init-student-checkpoint", default=None)
    parser.add_argument(
        "--student-tokenizer",
        default=None,
        help="Optional compatible fast tokenizer (useful for legacy checkpoints with vocab.txt only).",
    )
    parser.add_argument("--hotpot-samples", type=int, default=4000)
    parser.add_argument("--two-wiki-samples", type=int, default=4000)
    parser.add_argument("--synthetic-samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--hard-negatives", type=int, default=4)
    parser.add_argument("--prep-workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distill-weight", type=float, default=0.7)
    parser.add_argument("--root-positive-weight", type=float, default=1.0)
    parser.add_argument("--terminal-positive-weight", type=float, default=1.0)
    parser.add_argument("--relation-positive-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def encode_student(model, tokenizer, texts, device, max_length):
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return model.encode({key: value.to(device) for key, value in encoded.items()})


def encode_teacher(tokenizer, texts, device, max_length=320):
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return {key: value.to(device) for key, value in encoded.items()}


def distillation_loss(teacher, teacher_tokenizer, student, student_tokenizer, batch, device, args):
    losses = []
    temperature = args.temperature
    for task in ("root", "terminal", "edge"):
        items = [item for item in batch if item.task == task]
        if not items:
            continue
        with torch.inference_mode():
            teacher_encoded = encode_teacher(teacher_tokenizer, [item.text for item in items], device)
            if task == "root":
                teacher_logits = teacher.root_logits(teacher_encoded)
            elif task == "terminal":
                teacher_logits = teacher.terminal_logits(teacher_encoded)
            else:
                teacher_logits = teacher.relation_logits(teacher_encoded)

        query = encode_student(student, student_tokenizer, [item.question for item in items], device, 96)
        source = encode_student(student, student_tokenizer, [item.source for item in items], device, 192)
        if task == "root":
            student_logits = student.root_logits_from_embeddings(query, source)
            labels = torch.tensor([item.label for item in items], dtype=torch.float32, device=device)
            supervised = F.binary_cross_entropy_with_logits(
                student_logits,
                labels,
                pos_weight=torch.tensor(args.root_positive_weight, device=device),
            )
            soft_targets = torch.sigmoid(teacher_logits / temperature)
            distilled = F.binary_cross_entropy_with_logits(student_logits / temperature, soft_targets) * temperature**2
        elif task == "terminal":
            student_logits = student.terminal_logits_from_embeddings(query, source)
            labels = torch.tensor([item.label for item in items], dtype=torch.float32, device=device)
            supervised = F.binary_cross_entropy_with_logits(
                student_logits,
                labels,
                pos_weight=torch.tensor(args.terminal_positive_weight, device=device),
            )
            soft_targets = torch.sigmoid(teacher_logits / temperature)
            distilled = F.binary_cross_entropy_with_logits(student_logits / temperature, soft_targets) * temperature**2
        else:
            target = encode_student(student, student_tokenizer, [item.target for item in items], device, 192)
            student_logits = student.relation_logits_from_embeddings(query, source, target)
            labels = torch.tensor([item.label for item in items], dtype=torch.long, device=device)
            class_weights = torch.ones(student_logits.shape[-1], device=device)
            class_weights[1:] = args.relation_positive_weight
            supervised = F.cross_entropy(student_logits, labels, weight=class_weights)
            distilled = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * temperature**2
        losses.append((1.0 - args.distill_weight) * supervised + args.distill_weight * distilled)
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
    jobs = [
        (example, args.hard_negatives, args.seed * 1_000_003 + idx)
        for idx, example in enumerate(examples)
    ]
    with Pool(processes=args.prep_workers) as pool:
        groups = pool.map(make_items_job, jobs, chunksize=max(1, len(jobs) // (args.prep_workers * 8)))
    items = [item for group in groups for item in group]
    random.Random(args.seed + 3).shuffle(items)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher, teacher_tokenizer = TypedRelationScorer.load_checkpoint(args.teacher_checkpoint, device=device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if args.init_student_checkpoint:
        student, student_tokenizer = CachedTypedRelationScorer.load_checkpoint(
            args.init_student_checkpoint, device=device
        )
        if student.projection_dim != args.projection_dim:
            raise ValueError(
                f"projection_dim mismatch: checkpoint={student.projection_dim} args={args.projection_dim}"
            )
        if args.student_architecture == "late_interaction":
            if not isinstance(student, LateInteractionTypedRelationScorer):
                raise ValueError("late_interaction continuation requires a late-interaction checkpoint")
            if student.num_vectors != args.num_vectors:
                raise ValueError(
                    f"num_vectors mismatch: checkpoint={student.num_vectors} args={args.num_vectors}"
                )
        elif isinstance(student, LateInteractionTypedRelationScorer):
            raise ValueError("pooled continuation requires a pooled checkpoint")
        student_tokenizer_source = args.init_student_checkpoint
    else:
        student_tokenizer_source = args.student_tokenizer or args.student_encoder
        student_tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_source, use_fast=True)
        scorer_class = (
            LateInteractionTypedRelationScorer
            if args.student_architecture == "late_interaction"
            else CachedTypedRelationScorer
        )
        scorer_kwargs = {"projection_dim": args.projection_dim}
        if args.student_architecture == "late_interaction":
            scorer_kwargs["num_vectors"] = args.num_vectors
        student = scorer_class(args.student_encoder, **scorer_kwargs).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loader = DataLoader(ItemDataset(items), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    history = []
    started = time.perf_counter()
    student.train()
    for epoch in range(args.epochs):
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = distillation_loss(
                teacher, teacher_tokenizer, student, student_tokenizer, batch, device, args
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            if step % 200 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "running_loss": running / step}), flush=True)
        record = {"epoch": epoch + 1, "loss": running / max(1, step), "steps": step}
        history.append(record)
        print(json.dumps(record), flush=True)
    elapsed = time.perf_counter() - started
    metadata = {
        "architecture": args.student_architecture,
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_encoder": args.student_encoder,
        "student_tokenizer": student_tokenizer_source,
        "projection_dim": args.projection_dim,
        "num_vectors": args.num_vectors,
        "hotpot_samples": args.hotpot_samples,
        "two_wiki_samples": args.two_wiki_samples,
        "synthetic_samples": args.synthetic_samples,
        "training_items": len(items),
        "temperature": args.temperature,
        "distill_weight": args.distill_weight,
        "root_positive_weight": args.root_positive_weight,
        "terminal_positive_weight": args.terminal_positive_weight,
        "relation_positive_weight": args.relation_positive_weight,
        "init_student_checkpoint": args.init_student_checkpoint,
        "history": history,
        "elapsed_seconds": elapsed,
    }
    student.save_checkpoint(output_dir, student_tokenizer, metadata)
    (output_dir / "train_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "items": len(items), "seconds": elapsed}))


if __name__ == "__main__":
    main()
