from __future__ import annotations

import random
import hashlib
from dataclasses import dataclass

from .datasets import EvidenceExample, EvidenceExampleSpan
from .schema import SpanKind


@dataclass(frozen=True)
class TopologyConfig:
    hops: int = 2
    branch_factor: int = 1
    alternative_proofs: int = 1
    distractor_chains: int = 0
    evidence_position: str = "mixed"
    entity_collisions: bool = False
    fact_reorder: bool = False
    paraphrase: bool = False
    marker_removal: bool = False
    contradiction: bool = False
    cycle: bool = False
    broken_middle: bool = False

    def __post_init__(self) -> None:
        if self.hops < 1:
            raise ValueError("hops must be positive")
        if self.branch_factor < 1 or self.alternative_proofs < 1:
            raise ValueError("branch_factor and alternative_proofs must be positive")
        if self.evidence_position not in {"start", "middle", "end", "mixed"}:
            raise ValueError("unknown evidence_position")


def _edge_text(source: str, target: str, config: TopologyConfig, variant: int) -> str:
    if config.marker_removal:
        templates = [
            "Travelers leaving {source} invariably arrive at {target}.",
            "The successor associated with {source} is {target}.",
            "Archival notes pair {source} with the next station, {target}.",
        ]
    elif config.paraphrase:
        templates = [
            "Following {source} takes the route onward to {target}.",
            "From {source}, the recorded continuation is {target}.",
            "The trail after {source} resumes at {target}.",
        ]
    else:
        templates = ["{source} points to {target}."]
    return templates[variant % len(templates)].format(source=source, target=target)


def make_topology_example(config: TopologyConfig, seed: int, sample_index: int = 0) -> EvidenceExample:
    rng = random.Random((seed + 1) * 1_000_003 + sample_index)

    def opaque(kind: str, *parts: object) -> str:
        payload = ":".join(map(str, (seed, sample_index, kind, *parts))).encode()
        digest = hashlib.sha256(payload).hexdigest()[:10].upper()
        return f"{kind}_{digest}"

    root = opaque("NODE", "root")
    answer = opaque("CODE", "answer")
    spans: list[EvidenceExampleSpan] = []
    gold_proofs: list[frozenset[str]] = []

    def add(
        span_id: str,
        text: str,
        role: str,
        gold_proof: int | None = None,
        **extra_metadata,
    ) -> None:
        spans.append(
            EvidenceExampleSpan(
                span_id,
                "Evidence",
                text,
                SpanKind.SENTENCE,
                {"role": role, "gold_proof": gold_proof, **extra_metadata},
            )
        )

    for proof_idx in range(config.alternative_proofs):
        nodes = [root] + [opaque("NODE", "proof", proof_idx, hop) for hop in range(1, config.hops + 1)]
        proof_ids: set[str] = set()
        for hop, (source, target) in enumerate(zip(nodes, nodes[1:])):
            span_id = f"gold_p{proof_idx}_e{hop}"
            add(
                span_id,
                _edge_text(source, target, config, proof_idx + hop),
                "gold",
                proof_idx,
                hop=hop,
                is_root=hop == 0,
            )
            proof_ids.add(span_id)
            for branch in range(1, config.branch_factor):
                wrong = opaque("NODE", "branch", proof_idx, hop, branch)
                add(
                    f"branch_p{proof_idx}_e{hop}_b{branch}",
                    _edge_text(source, wrong, config, branch),
                    "branch_distractor",
                )
        terminal_id = f"gold_p{proof_idx}_answer"
        terminal_text = (
            f"The value kept at {nodes[-1]} is {answer}."
            if config.marker_removal
            else f"{nodes[-1]} stores answer {answer}."
        )
        add(
            terminal_id, terminal_text, "gold", proof_idx,
            is_terminal=True, is_terminal_candidate=True,
        )
        proof_ids.add(terminal_id)
        gold_proofs.append(frozenset(proof_ids))
        if config.cycle:
            add(f"cycle_p{proof_idx}", _edge_text(nodes[-1], root, config, proof_idx), "cycle")
        if config.contradiction:
            add(
                f"contradiction_p{proof_idx}",
                f"Despite a misleading report, {nodes[0]} does not point to {nodes[1]}.",
                "contradiction",
            )

    for chain_idx in range(config.distractor_chains):
        nodes = [opaque("NODE", "distractor", chain_idx, hop) for hop in range(config.hops + 1)]
        for hop, (source, target) in enumerate(zip(nodes, nodes[1:])):
            add(
                f"distractor_c{chain_idx}_e{hop}",
                _edge_text(source, target, config, chain_idx + hop),
                "distractor_chain",
            )
        add(
            f"distractor_c{chain_idx}_answer",
            f"{nodes[-1]} stores answer {opaque('CODE', 'distractor', chain_idx)}.",
            "distractor_chain",
            is_terminal_candidate=True,
        )

    if config.broken_middle:
        left = opaque("NODE", "broken", "left")
        right = opaque("NODE", "broken", "right")
        add("broken_start", _edge_text(root, left, config, 0), "broken_middle")
        add("broken_end", f"{right} stores answer {answer}.", "broken_middle")

    filler_count = max(8, 4 * (config.distractor_chains + config.branch_factor))
    for idx in range(filler_count):
        shared = root if config.entity_collisions and idx % 2 == 0 else opaque("NODE", "filler", idx)
        add(
            f"filler_{idx}",
            f"{shared} appears in an unrelated archival note numbered {rng.randrange(10_000, 99_999)}.",
            "filler",
        )

    gold_spans = [span for span in spans if span.metadata.get("role") == "gold"]
    nongold_spans = [span for span in spans if span.metadata.get("role") != "gold"]
    rng.shuffle(nongold_spans)
    if config.fact_reorder:
        rng.shuffle(gold_spans)
    if config.evidence_position == "start":
        ordered = gold_spans + nongold_spans
    elif config.evidence_position == "end":
        ordered = nongold_spans + gold_spans
    elif config.evidence_position == "middle":
        midpoint = len(nongold_spans) // 2
        ordered = nongold_spans[:midpoint] + gold_spans + nongold_spans[midpoint:]
    else:
        ordered = spans[:]
        rng.shuffle(ordered)
    return EvidenceExample(
        dataset="proof_topology",
        split="generated",
        sample_id=f"topology-{seed}-{sample_index}",
        question=f"Starting from {root}, what stored answer can be reached?",
        answer=answer,
        spans=tuple(ordered),
        gold_proofs=tuple(gold_proofs),
        question_type=f"{config.hops}hop_b{config.branch_factor}_p{config.alternative_proofs}",
    )
