from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .schema import EvidenceIndex, EvidenceSpan, Relation, RelationType, SpanKind


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
ENTITY_RE = re.compile(r"\b(?:[A-Z][\w'-]*)(?:\s+[A-Z][\w'-]*){0,4}\b")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
VARIABLE_RE = re.compile(r"\b(?:NODE_[A-Z0-9_]+|[A-Z][A-Z0-9_]*\d+|[A-Za-z_]\w*\s*=)\b")
DEFINITION_RE = re.compile(r"\b([A-Za-z][\w -]{1,40})\s+(?:is|are|means|refers to|is called)\b", re.I)
NEGATION_RE = re.compile(r"\b(?:not|never|neither|no longer|failed|false|incorrect)\b", re.I)
STATE_RE = re.compile(r"\b(?:becomes?|changes?|updates?|points? to|resolves? to|assigned|set to|moves? to)\b", re.I)
CAUSE_RE = re.compile(r"\b(?:because|therefore|thus|causes?|leads? to|results? in|due to)\b", re.I)
STOPWORDS = {
    "about", "after", "again", "also", "among", "answer", "before", "being", "between",
    "could", "does", "from", "have", "into", "itself", "more", "most", "other", "question",
    "should", "some", "such", "than", "that", "their", "there", "these", "they", "this",
    "those", "through", "under", "using", "very", "were", "what", "when", "where", "which",
    "while", "with", "would",
}


@dataclass(frozen=True)
class IndexerConfig:
    embedding_dim: int = 64
    min_bridge_chars: int = 3
    max_relation_distance: int | None = None
    lexical_relation_max_df: int = 3
    relation_floor: float = 0.20


def lexical_terms(text: str) -> tuple[str, ...]:
    return tuple(
        token.lower()
        for token in WORD_RE.findall(text)
        if len(token) >= 3 and token.lower() not in STOPWORDS
    )


def hashed_embedding(text: str, dim: int) -> tuple[float, ...]:
    values = [0.0] * dim
    for token, count in Counter(lexical_terms(text)).items():
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        index = raw % dim
        sign = 1.0 if (raw >> 8) & 1 else -1.0
        values[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return tuple(value / norm for value in values)


def extract_cues(text: str) -> Mapping[str, tuple[str, ...]]:
    entities = tuple(dict.fromkeys(match.strip().lower() for match in ENTITY_RE.findall(text)))
    years = tuple(dict.fromkeys(YEAR_RE.findall(text)))
    variables = tuple(dict.fromkeys(match.rstrip(" =").lower() for match in VARIABLE_RE.findall(text)))
    definitions = tuple(dict.fromkeys(match.strip().lower() for match in DEFINITION_RE.findall(text)))
    return {
        "entities": entities,
        "times": years,
        "variables": variables,
        "definitions": definitions,
        "terms": tuple(dict.fromkeys(lexical_terms(text))),
        "negation": ("true",) if NEGATION_RE.search(text) else (),
        "state": ("true",) if STATE_RE.search(text) else (),
        "causal": ("true",) if CAUSE_RE.search(text) else (),
    }


class EvidenceIndexer:
    """Build a query-agnostic semantic span index and typed candidate relations."""

    def __init__(self, config: IndexerConfig | None = None):
        self.config = config or IndexerConfig()

    def build(
        self,
        document_id: str,
        spans: Sequence[tuple[int, int, str] | EvidenceSpan],
        token_count: int,
        kinds: Sequence[SpanKind] | None = None,
        metadata: Sequence[Mapping[str, object]] | None = None,
    ) -> EvidenceIndex:
        evidence: list[EvidenceSpan] = []
        for idx, value in enumerate(spans):
            if isinstance(value, EvidenceSpan):
                evidence.append(value)
                continue
            start, end, text = value
            evidence.append(
                EvidenceSpan(
                    span_id=f"s{idx}",
                    token_start=start,
                    token_end=end,
                    text=text,
                    kind=SpanKind.SENTENCE if kinds is None else kinds[idx],
                    cues=extract_cues(text),
                    embedding=hashed_embedding(text, self.config.embedding_dim),
                    metadata={} if metadata is None else metadata[idx],
                )
            )
        relations = self._candidate_relations(evidence)
        return EvidenceIndex(document_id, tuple(evidence), tuple(relations), token_count)

    def _candidate_relations(self, spans: Sequence[EvidenceSpan]) -> list[Relation]:
        df = Counter(term for span in spans for term in set(span.cues.get("terms", ())))
        relations: list[Relation] = []
        for left_idx, left in enumerate(spans):
            for right_idx, right in enumerate(spans):
                if left_idx == right_idx:
                    continue
                if self.config.max_relation_distance is not None:
                    if abs(left_idx - right_idx) > self.config.max_relation_distance:
                        continue
                relation = self._score_relation(left, right, df)
                if relation is not None and relation.probability >= self.config.relation_floor:
                    relations.append(relation)
        return relations

    def _score_relation(
        self, left: EvidenceSpan, right: EvidenceSpan, document_frequency: Counter[str]
    ) -> Relation | None:
        left_entities = set(left.cues.get("entities", ()))
        right_entities = set(right.cues.get("entities", ()))
        left_variables = set(left.cues.get("variables", ()))
        right_variables = set(right.cues.get("variables", ()))
        shared_entities = left_entities & right_entities
        shared_variables = left_variables & right_variables
        shared_times = set(left.cues.get("times", ())) & set(right.cues.get("times", ()))
        shared_terms = set(left.cues.get("terms", ())) & set(right.cues.get("terms", ()))
        rare_terms = {term for term in shared_terms if document_frequency[term] <= self.config.lexical_relation_max_df}

        relation_type: RelationType | None = None
        probability = 0.0
        evidence: Iterable[str] = ()
        if shared_variables and (left.cues.get("state") or right.cues.get("state")):
            relation_type = RelationType.STATE_TRANSITION
            probability = min(0.95, 0.62 + 0.10 * len(shared_variables))
            evidence = shared_variables
        elif shared_entities and left.cues.get("negation") != right.cues.get("negation"):
            relation_type = RelationType.CONTRADICTION
            probability = min(0.95, 0.58 + 0.08 * len(shared_entities))
            evidence = shared_entities
        elif shared_entities and right.cues.get("causal"):
            relation_type = RelationType.CAUSAL_SUPPORT
            probability = min(0.92, 0.57 + 0.07 * len(shared_entities))
            evidence = shared_entities
        elif shared_entities:
            relation_type = RelationType.COREFERENCE
            probability = min(0.92, 0.55 + 0.08 * len(shared_entities))
            evidence = shared_entities
        elif shared_times:
            relation_type = RelationType.TEMPORAL
            probability = min(0.86, 0.48 + 0.08 * len(shared_times))
            evidence = shared_times
        else:
            definitions = set(left.cues.get("definitions", ()))
            definition_hits = {term for term in definitions if term and term in right.text.lower()}
            if definition_hits:
                relation_type = RelationType.DEFINITION_USE
                probability = min(0.90, 0.60 + 0.08 * len(definition_hits))
                evidence = definition_hits
            elif rare_terms:
                relation_type = RelationType.LEXICAL_BRIDGE
                probability = min(0.78, 0.30 + 0.06 * len(rare_terms))
                evidence = rare_terms
        if relation_type is None:
            return None
        return Relation(
            source=left.span_id,
            target=right.span_id,
            relation_type=relation_type,
            probability=probability,
            evidence=tuple(sorted(evidence)),
        )
