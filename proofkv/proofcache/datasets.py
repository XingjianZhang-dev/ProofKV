from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schema import SpanKind


@dataclass(frozen=True)
class EvidenceExampleSpan:
    span_id: str
    title: str
    text: str
    kind: SpanKind
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class EvidenceExample:
    dataset: str
    split: str
    sample_id: str
    question: str
    answer: str
    spans: tuple[EvidenceExampleSpan, ...]
    gold_proofs: tuple[frozenset[str], ...]
    question_type: str = "unknown"

    @property
    def gold_span_ids(self) -> frozenset[str]:
        return frozenset().union(*self.gold_proofs) if self.gold_proofs else frozenset()

    def render_context(self) -> tuple[str, tuple[tuple[int, int], ...]]:
        pieces: list[str] = []
        char_spans: list[tuple[int, int]] = []
        cursor = 0
        for span in self.spans:
            line = f"[{span.title}] {span.text.strip()}\n"
            pieces.append(line)
            char_spans.append((cursor, cursor + len(line)))
            cursor += len(line)
        return "".join(pieces), tuple(char_spans)


def _as_pairs(value: Any, left: str, right: str) -> list[tuple[Any, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return list(zip(value[left], value[right]))
    return [
        (item[left], item[right]) if isinstance(item, dict) else (item[0], item[1])
        for item in value
    ]


def from_hotpot(row: Mapping[str, Any], split: str) -> EvidenceExample:
    contexts = _as_pairs(row["context"], "title", "sentences")
    support = set(_as_pairs(row["supporting_facts"], "title", "sent_id"))
    spans: list[EvidenceExampleSpan] = []
    gold: set[str] = set()
    for doc_idx, (title, sentences) in enumerate(contexts):
        for sent_idx, sentence in enumerate(sentences):
            span_id = f"d{doc_idx}s{sent_idx}"
            spans.append(
                EvidenceExampleSpan(
                    span_id,
                    str(title),
                    str(sentence),
                    SpanKind.SENTENCE,
                    {"document_index": doc_idx, "sentence_index": sent_idx},
                )
            )
            if (title, sent_idx) in support:
                gold.add(span_id)
    return EvidenceExample(
        dataset="hotpotqa",
        split=split,
        sample_id=str(row.get("id", row.get("_id"))),
        question=str(row["question"]),
        answer=str(row["answer"]),
        spans=tuple(spans),
        gold_proofs=(frozenset(gold),),
        question_type=str(row.get("type", "unknown")),
    )


def from_2wiki(row: Mapping[str, Any], split: str) -> EvidenceExample:
    contexts = _as_pairs(row["context"], "title", "content")
    support = set(_as_pairs(row["supporting_facts"], "title", "sent_id"))
    spans: list[EvidenceExampleSpan] = []
    gold: set[str] = set()
    for doc_idx, (title, sentences) in enumerate(contexts):
        for sent_idx, sentence in enumerate(sentences):
            span_id = f"d{doc_idx}s{sent_idx}"
            spans.append(
                EvidenceExampleSpan(
                    span_id,
                    str(title),
                    str(sentence),
                    SpanKind.SENTENCE,
                    {"document_index": doc_idx, "sentence_index": sent_idx},
                )
            )
            if (title, sent_idx) in support:
                gold.add(span_id)
    return EvidenceExample(
        dataset="2wikimultihopqa",
        split=split,
        sample_id=str(row["_id"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        spans=tuple(spans),
        gold_proofs=(frozenset(gold),),
        question_type=str(row.get("type", "unknown")),
    )


def from_musique(row: Mapping[str, Any], split: str) -> EvidenceExample:
    # Support both the original MuSiQue release schema and the official IRCoT
    # processed schema used by the locked evaluation protocol.
    paragraphs: Sequence[Mapping[str, Any]] = row.get("paragraphs", row.get("contexts", ()))
    if not paragraphs:
        raise ValueError("MuSiQue row contains neither paragraphs nor contexts")
    spans: list[EvidenceExampleSpan] = []
    gold: set[str] = set()
    for position, paragraph in enumerate(paragraphs):
        paragraph_idx = int(paragraph.get("idx", position))
        span_id = f"p{paragraph_idx}"
        spans.append(
            EvidenceExampleSpan(
                span_id,
                str(paragraph.get("title", f"paragraph-{paragraph_idx}")),
                str(paragraph["paragraph_text"]),
                SpanKind.SENTENCE,
                {"paragraph_index": paragraph_idx},
            )
        )
        if bool(paragraph.get("is_supporting", False)):
            gold.add(span_id)
    answer = row.get("answer")
    if answer is None:
        answer_objects = row.get("answers_objects", ())
        answer_spans = [
            str(span)
            for answer_object in answer_objects
            for span in answer_object.get("spans", ())
            if str(span).strip()
        ]
        answer = answer_spans[0] if answer_spans else ""
    return EvidenceExample(
        dataset="musique",
        split=split,
        sample_id=str(row.get("id", row.get("question_id"))),
        question=str(row.get("question", row.get("question_text"))),
        answer=str(answer),
        spans=tuple(spans),
        gold_proofs=(frozenset(gold),),
        question_type=f"{len(gold)}hop",
    )
