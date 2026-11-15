#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from multiprocessing import Pool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer

from proofkv.proofcache.datasets import EvidenceExample, from_2wiki, from_hotpot, from_musique
from proofkv.proofcache.index import EvidenceIndexer, extract_cues, hashed_embedding
from proofkv.proofcache.model import (
    RELATION_TO_ID,
    TypedRelationScorer,
    edge_text,
    root_text,
    terminal_text,
    tokenize_edge_batch,
)
from proofkv.proofcache.schema import EvidenceSpan, RelationType
from proofkv.proofcache.topology import TopologyConfig, make_topology_example
try:
    from scripts.eval_proofcache_selector import TWO_WIKI_TRAIN, bm25_priority
except ImportError:  # direct execution from scripts/
    from eval_proofcache_selector import TWO_WIKI_TRAIN, bm25_priority


@dataclass(frozen=True)
class TrainItem:
    task: str
    text: str
    label: int
    negative_type: str
    token_cost: int
    question: str = ""
    source: str = ""
    target: str = ""


class ItemDataset(Dataset):
    def __init__(self, items: list[TrainItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> TrainItem:
        return self.items[index]


class TaskLengthBatchSampler(Sampler[list[int]]):
    """Homogeneous task batches with coarse length bucketing.

    The three scorer heads require separate forwards.  Mixing them inside one
    nominal batch creates several small kernels and pads short span examples to
    long paragraph edges.  This sampler preserves global randomness at the
    batch level while reducing both sources of wasted GPU time.
    """

    def __init__(self, items: list[TrainItem], batch_size: int, seed: int):
        self.items = items
        self.batch_size = batch_size
        self.seed = seed

    def __len__(self) -> int:
        counts = Counter(item.task for item in self.items)
        return sum(math.ceil(count / self.batch_size) for count in counts.values())

    def __iter__(self):
        rng = random.Random(self.seed)
        batches: list[list[int]] = []
        for task in ("root", "terminal", "edge"):
            indices = [index for index, item in enumerate(self.items) if item.task == task]
            indices.sort(key=lambda index: self.items[index].token_cost)
            bucket_size = self.batch_size * 16
            for bucket_start in range(0, len(indices), bucket_size):
                bucket = indices[bucket_start : bucket_start + bucket_size]
                rng.shuffle(bucket)
                batches.extend(
                    bucket[start : start + self.batch_size]
                    for start in range(0, len(bucket), self.batch_size)
                )
        rng.shuffle(batches)
        yield from batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the frozen-LLM ProofCache typed relation scorer.")
    parser.add_argument("--encoder", default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--hotpot-samples", type=int, default=12000)
    parser.add_argument("--two-wiki-samples", type=int, default=12000)
    parser.add_argument("--musique-data-file", default=None)
    parser.add_argument("--musique-samples", type=int, default=0)
    parser.add_argument("--synthetic-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--edge-encoding", choices=("single", "balanced_pair"), default="single")
    parser.add_argument(
        "--homogeneous-length-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hard-negatives", type=int, default=4)
    parser.add_argument("--prep-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lambda-proof", type=float, default=0.5)
    parser.add_argument("--lambda-budget", type=float, default=0.05)
    parser.add_argument("--lambda-calibration", type=float, default=0.2)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def evidence_spans(example: EvidenceExample) -> list[EvidenceSpan]:
    spans = []
    cursor = 0
    for source in example.spans:
        token_count = max(1, len(source.text.split()))
        text = f"[{source.title}] {source.text}"
        spans.append(
            EvidenceSpan(
                source.span_id,
                cursor,
                cursor + token_count,
                text,
                source.kind,
                extract_cues(text),
                hashed_embedding(text, 64),
                {**source.metadata, "title": source.title},
            )
        )
        cursor += token_count
    return spans


def positive_relation_type(index, source_id: str, target_id: str) -> RelationType:
    candidates = [
        relation for relation in index.adjacency.get(source_id, ()) if relation.target == target_id
    ]
    if not candidates:
        return RelationType.LEXICAL_BRIDGE
    return max(candidates, key=lambda relation: relation.probability).relation_type


def make_items(example: EvidenceExample, hard_negative_count: int, rng: random.Random) -> list[TrainItem]:
    spans = evidence_spans(example)
    by_id = {span.span_id: span for span in spans}
    gold = set(example.gold_span_ids)
    if not gold or len(gold) == len(spans):
        return []
    index = EvidenceIndexer().build(example.sample_id, spans, max(span.token_end for span in spans))
    hard_priority = [span_id for span_id in bm25_priority(example.question, spans) if span_id not in gold]
    hard = hard_priority[:hard_negative_count]
    random_pool = [span.span_id for span in spans if span.span_id not in gold and span.span_id not in hard]
    rng.shuffle(random_pool)
    random_negatives = random_pool[:2]
    items: list[TrainItem] = []
    normalized_answer = " ".join(example.answer.lower().split())
    gold_terminal_ids = {
        span_id for span_id in gold if bool(by_id[span_id].metadata.get("is_terminal", False))
    } or {
        span_id for span_id in gold if normalized_answer and normalized_answer in by_id[span_id].text.lower()
    }
    if not gold_terminal_ids:
        gold_terminal_ids = {
            max(proof, key=lambda span_id: by_id[span_id].token_start)
            for proof in example.gold_proofs
            if proof
        }
    # Terminal means "can end a proof", not "is secretly the gold answer".
    # Synthetic wrong-answer chains therefore provide positive terminal-form
    # examples; correctness must come from the root and directed relations.
    terminal_ids = gold_terminal_ids | {
        span.span_id
        for span in spans
        if bool(span.metadata.get("is_terminal_candidate", False))
    }
    bm25_order = bm25_priority(example.question, spans)
    bm25_rank = {span_id: rank for rank, span_id in enumerate(bm25_order)}
    root_ids = set()
    for proof in example.gold_proofs:
        if not proof:
            continue
        annotated = [span_id for span_id in proof if bool(by_id[span_id].metadata.get("is_root", False))]
        root_ids.add(
            annotated[0]
            if annotated
            else min(proof, key=lambda span_id: (bm25_rank.get(span_id, len(spans)), by_id[span_id].token_start))
        )

    for span_id in sorted(root_ids, key=lambda value: by_id[value].token_start):
        span = by_id[span_id]
        items.append(TrainItem("root", root_text(example.question, span.text), 1, "positive", span.token_count, example.question, span.text))
    gold_nonroots = sorted(
        gold - root_ids,
        key=lambda value: (bm25_rank.get(value, len(spans)), by_id[value].token_start),
    )[:hard_negative_count]
    root_negative_ids = list(dict.fromkeys(hard + random_negatives + gold_nonroots))
    for span_id in root_negative_ids:
        span = by_id[span_id]
        if span_id in gold_nonroots:
            negative_type = "gold_nonroot"
        elif span_id in hard:
            negative_type = "high_lexical_overlap"
        else:
            negative_type = "random_distractor"
        items.append(TrainItem("root", root_text(example.question, span.text), 0, negative_type, span.token_count, example.question, span.text))

    nonterminal_gold = sorted(
        gold - terminal_ids,
        key=lambda value: (bm25_rank.get(value, len(spans)), by_id[value].token_start),
    )[:hard_negative_count]
    terminal_candidates = list(dict.fromkeys(sorted(terminal_ids) + nonterminal_gold + hard[:2]))
    for span_id in terminal_candidates:
        span = by_id[span_id]
        label = int(span_id in terminal_ids)
        items.append(
            TrainItem(
                "terminal", terminal_text(example.question, span.text), label,
                "positive" if label else "nonterminal_or_distractor",
                span.token_count, example.question, span.text,
            )
        )

    positive_pairs: list[tuple[str, str]] = []
    explicitly_directed_pairs: list[tuple[str, str]] = []
    ordered_proofs = []
    for proof in example.gold_proofs:
        if not proof:
            continue
        annotated_roots = [span_id for span_id in proof if bool(by_id[span_id].metadata.get("is_root", False))]
        root_id = (
            annotated_roots[0]
            if annotated_roots
            else min(proof, key=lambda span_id: (bm25_rank.get(span_id, len(spans)), by_id[span_id].token_start))
        )
        proof_terminals = set(proof) & gold_terminal_ids
        terminal_id = max(
            proof_terminals or set(proof), key=lambda span_id: by_id[span_id].token_start
        )
        def proof_order(span_id: str) -> tuple[int, int]:
            hop = by_id[span_id].metadata.get("hop")
            # Synthetic/topology data supplies the actual logical hop.  Document
            # order is only a fallback for natural datasets without dependency
            # annotations; fact-reorder augmentation must not reverse labels.
            return (
                int(hop) if hop is not None else 1_000_000,
                by_id[span_id].token_start,
            )

        middle = sorted(set(proof) - {root_id, terminal_id}, key=proof_order)
        ordered_proof = [root_id] + middle + ([] if terminal_id == root_id else [terminal_id])
        ordered_proofs.append(ordered_proof)
        has_explicit_direction = any(
            by_id[span_id].metadata.get("hop") is not None for span_id in proof
        )
        if len(ordered_proof) == 1:
            positive_pairs.append((ordered_proof[0], ordered_proof[0]))
        else:
            for left, right in zip(ordered_proof, ordered_proof[1:]):
                positive_pairs.append((left, right))
                if has_explicit_direction:
                    explicitly_directed_pairs.append((left, right))
    for source_id, target_id in positive_pairs:
        source, target = by_id[source_id], by_id[target_id]
        relation_type = positive_relation_type(index, source_id, target_id) if source_id != target_id else RelationType.LEXICAL_BRIDGE
        items.append(
            TrainItem(
                "edge",
                edge_text(example.question, source.text, target.text),
                RELATION_TO_ID[relation_type.value],
                "positive",
                source.token_count + target.token_count,
                example.question,
                source.text,
                target.text,
            )
        )

    positive_pair_set = set(positive_pairs)
    negative_pair_set: set[tuple[str, str]] = set()

    def add_edge_negative(source_id: str, target_id: str, negative_type: str) -> None:
        pair = (source_id, target_id)
        if (
            source_id == target_id
            or pair in positive_pair_set
            or pair in negative_pair_set
        ):
            return
        negative_pair_set.add(pair)
        source, target = by_id[source_id], by_id[target_id]
        items.append(
            TrainItem(
                "edge", edge_text(example.question, source.text, target.text), 0,
                negative_type, source.token_count + target.token_count,
                example.question, source.text, target.text,
            )
        )

    # Directed proof search needs explicit reverse-edge supervision.  Merely
    # omitting reverse edges from positives does not teach a softmax classifier
    # that the reverse direction is invalid.
    for source_id, target_id in explicitly_directed_pairs:
        add_edge_negative(target_id, source_id, "reversed_gold_edge")

    # Alternative proofs are OR branches, not permission to splice arbitrary
    # starts and terminals into shortcut edges.
    for left_idx, left_proof in enumerate(ordered_proofs):
        for right_proof in ordered_proofs[left_idx + 1 :]:
            add_edge_negative(left_proof[0], right_proof[-1], "cross_proof_shortcut")
            add_edge_negative(right_proof[0], left_proof[-1], "cross_proof_shortcut")

    # Hard edge negatives: correct start with wrong bridge; wrong start with correct end;
    # and correct endpoints connected through a distractor rather than the gold middle.
    if hard:
        endpoints = list(dict.fromkeys(
            sorted(root_ids, key=lambda value: by_id[value].token_start)
            + sorted(gold_terminal_ids, key=lambda value: by_id[value].token_start)
        ))
        for negative_id in hard[:hard_negative_count]:
            for source_id in endpoints:
                add_edge_negative(source_id, negative_id, "missing_or_wrong_middle")
            for terminal_id in gold_terminal_ids:
                add_edge_negative(negative_id, terminal_id, "correct_endpoint_wrong_source")
    return items


def load_training_examples(args: argparse.Namespace) -> list[EvidenceExample]:
    examples: list[EvidenceExample] = []
    if args.hotpot_samples:
        hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
        hotpot_order = list(range(len(hotpot)))
        random.Random(args.seed).shuffle(hotpot_order)
        examples.extend(from_hotpot(hotpot[idx], "train") for idx in hotpot_order[: args.hotpot_samples])
    if args.two_wiki_samples:
        wiki = load_dataset("parquet", data_files={"train": TWO_WIKI_TRAIN}, split="train")
        wiki_order = list(range(len(wiki)))
        random.Random(args.seed + 1).shuffle(wiki_order)
        examples.extend(from_2wiki(wiki[idx], "train") for idx in wiki_order[: args.two_wiki_samples])
    if args.musique_samples:
        if not args.musique_data_file:
            raise ValueError("--musique-samples requires --musique-data-file")
        musique = load_dataset(
            "json", data_files={"train": args.musique_data_file}, split="train"
        )
        musique_order = list(range(len(musique)))
        random.Random(args.seed + 2).shuffle(musique_order)
        examples.extend(
            from_musique(musique[idx], "train")
            for idx in musique_order[: args.musique_samples]
        )
    synthetic_samples = int(getattr(args, "synthetic_samples", 0))
    for idx in range(synthetic_samples):
        variant = idx % 9
        examples.append(
            make_topology_example(
                TopologyConfig(
                    hops=(1, 2, 3, 4, 5, 6, 8)[idx % 7],
                    branch_factor=(1, 2, 4)[(idx // 7) % 3],
                    alternative_proofs=(1, 2, 4)[(idx // 21) % 3],
                    distractor_chains=(0, 4, 16)[(idx // 63) % 3],
                    evidence_position=("start", "middle", "end", "mixed")[(idx // 189) % 4],
                    entity_collisions=variant == 0,
                    fact_reorder=variant == 1,
                    paraphrase=variant == 2,
                    marker_removal=variant == 3,
                    contradiction=variant == 4,
                    cycle=variant == 5,
                    broken_middle=variant in {6, 7, 8},
                ),
                seed=args.seed,
                sample_index=idx,
            )
        )
    random.Random(args.seed + 3).shuffle(examples)
    return examples


def collate(items: list[TrainItem]) -> list[TrainItem]:
    return items


def make_items_job(payload: tuple[EvidenceExample, int, int]) -> list[TrainItem]:
    example, hard_negatives, seed = payload
    return make_items(example, hard_negatives, random.Random(seed))


def batch_loss(
    model: TypedRelationScorer,
    tokenizer: Any,
    batch: list[TrainItem],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    root_items = [item for item in batch if item.task == "root"]
    terminal_items = [item for item in batch if item.task == "terminal"]
    edge_items = [item for item in batch if item.task == "edge"]
    losses: list[torch.Tensor] = []
    stats: dict[str, float] = {}
    if root_items:
        encoded = tokenizer(
            [item.text for item in root_items], padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = torch.tensor([item.label for item in root_items], dtype=torch.float32, device=device)
        logits = model.root_logits(encoded)
        probabilities = torch.sigmoid(logits)
        root_loss = F.binary_cross_entropy_with_logits(logits, labels)
        positive = labels > 0.5
        proof_loss = -torch.log(probabilities[positive].clamp_min(1e-6)).mean() if positive.any() else logits.sum() * 0
        negative = ~positive
        costs = torch.tensor([item.token_cost for item in root_items], dtype=torch.float32, device=device)
        budget_loss = (probabilities[negative] * costs[negative]).mean() if negative.any() else logits.sum() * 0
        calibration_loss = F.mse_loss(probabilities, labels)
        losses.append(
            root_loss
            + args.lambda_proof * proof_loss
            + args.lambda_budget * budget_loss / 128.0
            + args.lambda_calibration * calibration_loss
        )
        stats.update(root_loss=float(root_loss.detach()), proof_loss=float(proof_loss.detach()), calibration_loss=float(calibration_loss.detach()))
    if terminal_items:
        encoded = tokenizer(
            [item.text for item in terminal_items], padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = torch.tensor([item.label for item in terminal_items], dtype=torch.float32, device=device)
        logits = model.terminal_logits(encoded)
        probabilities = torch.sigmoid(logits)
        terminal_loss = F.binary_cross_entropy_with_logits(logits, labels)
        positive = labels > 0.5
        proof_loss = -torch.log(probabilities[positive].clamp_min(1e-6)).mean() if positive.any() else logits.sum() * 0
        calibration_loss = F.mse_loss(probabilities, labels)
        losses.append(terminal_loss + args.lambda_proof * proof_loss + args.lambda_calibration * calibration_loss)
        stats.update(terminal_loss=float(terminal_loss.detach()))
    if edge_items:
        encoded = tokenize_edge_batch(
            tokenizer,
            [item.question for item in edge_items],
            [item.source for item in edge_items],
            [item.target for item in edge_items],
            args.max_length,
            edge_encoding=args.edge_encoding,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = torch.tensor([item.label for item in edge_items], dtype=torch.long, device=device)
        logits = model.relation_logits(encoded)
        edge_loss = F.cross_entropy(logits, labels)
        probabilities = torch.softmax(logits, dim=-1)
        positive = labels > 0
        proof_edge_loss = -torch.log(
            probabilities[positive, labels[positive]].clamp_min(1e-6)
        ).mean() if positive.any() else logits.sum() * 0
        calibration_target = positive.float()
        non_none_probability = 1.0 - probabilities[:, 0]
        calibration_loss = F.mse_loss(non_none_probability, calibration_target)
        losses.append(edge_loss + args.lambda_proof * proof_edge_loss + args.lambda_calibration * calibration_loss)
        stats.update(edge_loss=float(edge_loss.detach()), proof_edge_loss=float(proof_edge_loss.detach()))
    return sum(losses) / max(1, len(losses)), stats


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "proofcache_scorer.json").exists():
        raise FileExistsError(output_dir)
    examples = load_training_examples(args)
    jobs = [
        (example, args.hard_negatives, args.seed * 1_000_003 + idx)
        for idx, example in enumerate(examples)
    ]
    if args.prep_workers > 1:
        with Pool(processes=args.prep_workers) as pool:
            item_groups = pool.map(make_items_job, jobs, chunksize=max(1, len(jobs) // (args.prep_workers * 8)))
    else:
        item_groups = [make_items_job(job) for job in jobs]
    items = [item for group in item_groups for item in group]
    rng = random.Random(args.seed + 3)
    rng.shuffle(items)
    counts = Counter((item.task, item.negative_type) for item in items)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.init_checkpoint:
        model, tokenizer = TypedRelationScorer.load_checkpoint(args.init_checkpoint, device=device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
        model = TypedRelationScorer(args.encoder)
        model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    dataset = ItemDataset(items)
    if args.homogeneous_length_batches:
        loader = DataLoader(
            dataset,
            batch_sampler=TaskLengthBatchSampler(items, args.batch_size, args.seed + 4),
            collate_fn=collate,
        )
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    print(
        json.dumps(
            {
                "training_items": len(items),
                "steps_per_epoch": len(loader),
                "batch_size": args.batch_size,
            }
        ),
        flush=True,
    )
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        steps = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = batch_loss(model, tokenizer, batch, device, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            steps += 1
            if steps % 200 == 0:
                print(
                    json.dumps(
                        {"epoch": epoch + 1, "step": steps, "running_loss": running / steps}
                    ),
                    flush=True,
                )
        history.append({"epoch": epoch + 1, "loss": running / max(1, steps), "steps": steps})
        print(json.dumps(history[-1]), flush=True)
    elapsed = time.perf_counter() - started
    metadata = {
        "encoder": args.encoder,
        "init_checkpoint": args.init_checkpoint,
        "hotpot_samples": args.hotpot_samples,
        "two_wiki_samples": args.two_wiki_samples,
        "musique_data_file": args.musique_data_file,
        "musique_samples": args.musique_samples,
        "synthetic_samples": args.synthetic_samples,
        "epochs": args.epochs,
        "seed": args.seed,
        "edge_encoding": args.edge_encoding,
        "homogeneous_length_batches": args.homogeneous_length_batches,
        "loss_weights": {
            "proof": args.lambda_proof,
            "budget": args.lambda_budget,
            "calibration": args.lambda_calibration,
        },
        "negative_counts": {f"{task}:{kind}": value for (task, kind), value in sorted(counts.items())},
        "training_items": len(items),
        "history": history,
        "elapsed_seconds": elapsed,
    }
    model.edge_encoding = args.edge_encoding
    model.save_checkpoint(output_dir, tokenizer, metadata)
    (output_dir / "train_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "items": len(items), "seconds": elapsed}))


if __name__ == "__main__":
    main()
