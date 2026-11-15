from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class SpanKind(str, Enum):
    SENTENCE = "sentence"
    TABLE_ROW = "table_row"
    CODE_BLOCK = "code_block"
    DIALOGUE_TURN = "dialogue_turn"
    SECTION_HEADER = "section_header"


class RelationType(str, Enum):
    COREFERENCE = "coreference"
    STATE_TRANSITION = "state_transition"
    CAUSAL_SUPPORT = "causal_support"
    TABLE_JOIN = "table_join"
    DEFINITION_USE = "definition_use"
    TEMPORAL = "temporal"
    CONTRADICTION = "contradiction"
    LEXICAL_BRIDGE = "lexical_bridge"


class PrecisionTier(str, Enum):
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def bytes_per_value(self) -> float:
        return {self.BF16: 2.0, self.INT8: 1.0, self.INT4: 0.5}[self]


@dataclass(frozen=True)
class EvidenceSpan:
    span_id: str
    token_start: int
    token_end: int
    text: str
    kind: SpanKind = SpanKind.SENTENCE
    cues: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    embedding: tuple[float, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError(f"invalid token span for {self.span_id}")

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    def kv_bytes(self, values_per_token: int, precision: PrecisionTier = PrecisionTier.BF16) -> int:
        return int(round(self.token_count * values_per_token * precision.bytes_per_value))


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    relation_type: RelationType
    probability: float
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("relation probability must lie in [0, 1]")
        if self.source == self.target:
            raise ValueError("self relations are not allowed")

    @property
    def relation_id(self) -> str:
        return f"{self.source}|{self.relation_type.value}|{self.target}"


@dataclass(frozen=True)
class EvidenceIndex:
    document_id: str
    spans: tuple[EvidenceSpan, ...]
    relations: tuple[Relation, ...]
    token_count: int
    index_version: str = "proofcache-v1"

    def __post_init__(self) -> None:
        ids = [span.span_id for span in self.spans]
        if len(ids) != len(set(ids)):
            raise ValueError("span ids must be unique")
        known = set(ids)
        if any(rel.source not in known or rel.target not in known for rel in self.relations):
            raise ValueError("relation endpoint missing from index")
        if any(span.token_end > self.token_count for span in self.spans):
            raise ValueError("span extends beyond document token count")

    @property
    def by_id(self) -> dict[str, EvidenceSpan]:
        return {span.span_id: span for span in self.spans}

    @property
    def adjacency(self) -> dict[str, tuple[Relation, ...]]:
        rows: dict[str, list[Relation]] = {span.span_id: [] for span in self.spans}
        for relation in self.relations:
            rows[relation.source].append(relation)
        return {key: tuple(values) for key, values in rows.items()}


@dataclass(frozen=True)
class Proof:
    proof_id: str
    span_ids: tuple[str, ...]
    relations: tuple[Relation, ...]
    probability: float
    root_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.span_ids:
            raise ValueError("proof must contain at least one span")
        if len(self.span_ids) != len(set(self.span_ids)):
            raise ValueError("proof cannot repeat a span")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("proof probability must lie in [0, 1]")


@dataclass(frozen=True)
class ProofOverlay:
    query: str
    proofs: tuple[Proof, ...]
    root_scores: Mapping[str, float]
    confidence: float
    terminal_scores: Mapping[str, float] = field(default_factory=dict)
    selected_proof_ids: tuple[str, ...] = ()
    selected_span_ids: tuple[str, ...] = ()
    auxiliary_span_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("overlay confidence must lie in [0, 1]")


@dataclass(frozen=True)
class CompiledMemoryPlan:
    proof_core_span_ids: tuple[str, ...]
    auxiliary_span_ids_by_layer: Mapping[int, tuple[str, ...]]
    retained_token_indices: tuple[int, ...]
    precision_by_span: Mapping[str, PrecisionTier]
    byte_cost: int
    byte_budget: int
    confidence: float
    fallback_to_full: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.byte_cost > self.byte_budget and not self.fallback_to_full:
            raise ValueError("compiled plan exceeds byte budget")
        if tuple(sorted(set(self.retained_token_indices))) != self.retained_token_indices:
            raise ValueError("retained token indices must be sorted and unique")


def union_span_tokens(spans: Sequence[EvidenceSpan]) -> tuple[int, ...]:
    return tuple(sorted({token for span in spans for token in range(span.token_start, span.token_end)}))
