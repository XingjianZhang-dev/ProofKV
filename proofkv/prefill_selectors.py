from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


STOPWORDS = {
    "answer",
    "code",
    "final",
    "follow",
    "from",
    "give",
    "only",
    "question",
    "start",
    "starting",
    "the",
    "value",
    "what",
}

NODE_RE = re.compile(r"\bNODE_[A-Z0-9]+_[0-9]+\b", re.I)
VAR_ASSIGN_RE = re.compile(r"\bVAR\s+([A-Z]{2,8})\s*=\s*(VAR\s+)?([A-Z]{2,8}|\d{3,})\b")
QUERY_VALUE_RE = re.compile(r"\bvalue\s+(\d{3,})\b", re.I)
NATURAL_QA_STOPWORDS = STOPWORDS | {
    "are",
    "can",
    "did",
    "does",
    "find",
    "how",
    "is",
    "mentioned",
    "most",
    "recent",
    "where",
    "which",
    "who",
}


@dataclass(frozen=True)
class GuidepostPlan:
    guidepost_indices: List[int]
    retained_indices: List[int]
    spans: List[Tuple[int, int]]
    retained_ratio: float


def _stream_indices(seq_len: int, sink_size: int, recent_size: int) -> set[int]:
    sink = set(range(0, min(seq_len, sink_size)))
    recent = set(range(max(0, seq_len - recent_size), seq_len))
    return sink | recent


def _cap_to_budget(indices: Iterable[int], budget: int) -> List[int]:
    if budget <= 0:
        return []
    out: List[int] = []
    seen: set[int] = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
        if len(out) >= budget:
            break
    return sorted(out)


def _plan_from_spans(
    seq_len: int,
    spans: Sequence[Tuple[int, int]],
    guidepost_budget: int,
    sink_size: int,
    recent_size: int,
) -> GuidepostPlan:
    stream = _stream_indices(seq_len, sink_size, recent_size)
    ordered: List[int] = []
    clipped_spans: List[Tuple[int, int]] = []
    for start, end in spans:
        s = max(sink_size, min(seq_len, start))
        e = max(s, min(max(0, seq_len - recent_size), end))
        if e <= s:
            continue
        clipped_spans.append((s, e))
        ordered.extend(i for i in range(s, e) if i not in stream)
    guideposts = _cap_to_budget(ordered, guidepost_budget)
    retained = sorted(stream | set(guideposts))
    return GuidepostPlan(guideposts, retained, clipped_spans, len(retained) / max(1, seq_len))


def block_halo_guideposts(
    seq_len: int,
    guidepost_budget: int,
    sink_size: int,
    recent_size: int,
    block_size: int = 16,
    halo: int = 4,
    mode: str = "uniform",
    seed: int = 0,
) -> GuidepostPlan:
    """Select contiguous block+halo spans for random/uniform baselines."""

    start = min(max(0, sink_size), seq_len)
    end = max(start, seq_len - max(1, recent_size))
    if guidepost_budget <= 0 or end <= start:
        return _plan_from_spans(seq_len, [], 0, sink_size, recent_size)

    blocks = [(s, min(end, s + block_size)) for s in range(start, end, block_size)]
    span_width = block_size + 2 * halo
    n_spans = max(1, math.ceil(guidepost_budget / max(1, span_width)))
    n_spans = min(n_spans, len(blocks))
    if mode == "random":
        rng = random.Random(seed)
        picked = sorted(rng.sample(range(len(blocks)), n_spans))
    elif mode == "uniform":
        if n_spans == 1:
            picked = [len(blocks) // 2]
        else:
            picked = sorted({round(i * (len(blocks) - 1) / (n_spans - 1)) for i in range(n_spans)})
    else:
        raise ValueError("mode must be random or uniform")
    spans = [(blocks[i][0] - halo, blocks[i][1] + halo) for i in picked]
    return _plan_from_spans(seq_len, spans, guidepost_budget, sink_size, recent_size)


def _terms(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z]+[-_]?\d+|[A-Z]\d{2,}|K\d+|N\d+|[A-Za-z0-9_]{3,}", text)
    return {t.lower() for t in raw if t.lower() not in STOPWORDS}


def _node_ids(text: str) -> List[str]:
    return [m.group(0) for m in NODE_RE.finditer(text)]


def _entities(text: str) -> set[str]:
    raw = _node_ids(text) + re.findall(r"\b(?:K\d+|N\d+|[A-Z][A-Za-z]*[-_]?\d+|[A-Z]{2,}\d*)\b", text)
    return {t.lower() for t in raw if t.lower() not in STOPWORDS}


def _fact_like(text: str) -> float:
    score = 0.0
    if re.search(r"(?:->|=>|links? to|points? to|maps? to|stored|stores|reading|value|code|node)", text, re.I):
        score += 1.0
    if re.search(r"\b(?:K\d+|N\d+)\b", text):
        score += 1.0
    if "\n" in text:
        score += 0.25
    return score


def _block_text(prompt: str, offsets: Sequence[Tuple[int, int]], start: int, end: int) -> str:
    valid = [(s, e) for s, e in offsets[start:end] if e > s]
    if not valid:
        return ""
    return prompt[min(s for s, _ in valid) : max(e for _, e in valid)]


def _line_expanded_token_span(prompt: str, offsets: Sequence[Tuple[int, int]], start: int, end: int) -> Tuple[int, int]:
    valid = [(s, e) for s, e in offsets[start:end] if e > s]
    if not valid:
        return start, end
    char_start = min(s for s, _ in valid)
    char_end = max(e for _, e in valid)
    line_start = prompt.rfind("\n", 0, char_start) + 1
    line_end = prompt.find("\n", char_end)
    if line_end < 0:
        line_end = len(prompt)
    hits = [i for i, (s, e) in enumerate(offsets) if e > line_start and s < line_end]
    if not hits:
        return start, end
    return min(hits), max(hits) + 1


def _line_token_spans(
    prompt: str,
    offsets: Sequence[Tuple[int, int]],
    start: int,
    end: int,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    line_start = 0
    for line in prompt.splitlines(keepends=True):
        line_end = line_start + len(line)
        hits = [
            i
            for i, (s, e) in enumerate(offsets)
            if start <= i < end and e > line_start and s < line_end and e > s
        ]
        if hits:
            spans.append((min(hits), max(hits) + 1))
        line_start = line_end
    return spans


def _node_chain_spans(
    prompt: str,
    offsets: Sequence[Tuple[int, int]],
    query_text: str,
    start: int,
    end: int,
    halo: int,
    max_hops: int,
) -> List[Tuple[int, int]]:
    start_nodes = [n.lower() for n in _node_ids(query_text)]
    if not start_nodes:
        return []

    edges: dict[str, Tuple[str, Tuple[int, int]]] = {}
    resolving: dict[str, Tuple[int, int]] = {}
    for s, e in _line_token_spans(prompt, offsets, start, end):
        text = _block_text(prompt, offsets, s, e)
        if not re.match(r"\s*Fact:", text):
            continue
        nodes = [n.lower() for n in _node_ids(text)]
        if len(nodes) >= 2 and re.search(r"\bpoints?\s+to\b", text, re.I):
            edges.setdefault(nodes[0], (nodes[1], (s, e)))
        elif nodes and re.search(r"\bresolves?\s+to\s+number\s+\d+\b", text, re.I):
            resolving.setdefault(nodes[0], (s, e))

    spans: List[Tuple[int, int]] = []
    seen_spans: set[Tuple[int, int]] = set()
    node = start_nodes[0]
    visited: set[str] = set()
    for _ in range(max(1, min(max_hops, 5))):
        if node in visited:
            break
        visited.add(node)
        if node in edges:
            node, span = edges[node]
        elif node in resolving:
            span = resolving[node]
            if span not in seen_spans:
                seen_spans.add(span)
                spans.append((span[0] - halo, span[1] + halo))
            break
        else:
            break
        if span not in seen_spans:
            seen_spans.add(span)
            spans.append((span[0] - halo, span[1] + halo))
    return sorted(spans, key=lambda span: span[0])


def _var_assignment_chain_spans(
    prompt: str,
    offsets: Sequence[Tuple[int, int]],
    query_text: str,
    start: int,
    end: int,
    halo: int,
    max_hops: int,
) -> List[Tuple[int, int]]:
    """Recover official RULER variable-tracking chains from visible assignment lines."""

    value_match = QUERY_VALUE_RE.search(query_text)
    if value_match is None:
        return []
    query_value = value_match.group(1)

    roots: List[Tuple[str, Tuple[int, int]]] = []
    children: dict[str, List[Tuple[str, Tuple[int, int]]]] = {}
    for s, e in _line_token_spans(prompt, offsets, start, end):
        text = _block_text(prompt, offsets, s, e)
        for match in VAR_ASSIGN_RE.finditer(text):
            lhs = match.group(1).lower()
            rhs = match.group(3).lower()
            span = (s, e)
            if match.group(2):
                children.setdefault(rhs, []).append((lhs, span))
            elif rhs == query_value:
                roots.append((lhs, span))

    spans: List[Tuple[int, int]] = []
    seen_vars: set[str] = set()
    frontier = sorted(roots, key=lambda item: item[1][0])
    for _ in range(max(1, min(max_hops, 8))):
        if not frontier:
            break
        next_frontier: List[Tuple[str, Tuple[int, int]]] = []
        for var, span in frontier:
            if var in seen_vars:
                continue
            seen_vars.add(var)
            spans.append((span[0] - halo, span[1] + halo))
            next_frontier.extend(children.get(var, []))
        frontier = sorted(next_frontier, key=lambda item: item[1][0])
    return sorted(spans, key=lambda span: span[0])


def _natural_qa_terms(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_-]*|[A-Z0-9]{2,}", text)
    return {tok.lower() for tok in raw if tok.lower() not in NATURAL_QA_STOPWORDS and len(tok) >= 3}


def _natural_qa_main_entities(text: str) -> list[str]:
    entities: list[str] = []
    where = re.search(r"\bWhere\s+is\s+([A-Z][A-Za-z0-9_-]*)", text)
    if where:
        entities.append(where.group(1).lower())
    entities.extend(t.lower() for t in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", text))
    out: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        if entity in seen or entity in NATURAL_QA_STOPWORDS:
            continue
        seen.add(entity)
        out.append(entity)
    return out


def _sentence_token_spans(
    prompt: str,
    offsets: Sequence[Tuple[int, int]],
    start: int,
    end: int,
    char_limit: int,
) -> List[Tuple[int, int, str]]:
    rows: List[Tuple[int, int, str]] = []
    for match in re.finditer(r"[^.!?\n]+[.!?]?", prompt[:char_limit]):
        text = match.group(0).strip()
        if len(text) < 3:
            continue
        hits = [
            i
            for i, (s, e) in enumerate(offsets)
            if start <= i < end and e > match.start() and s < match.end() and e > s
        ]
        if hits:
            rows.append((min(hits), max(hits) + 1, text))
    return rows


def _natural_qa_sentence_spans(
    prompt: str,
    offsets: Sequence[Tuple[int, int]],
    query_text: str,
    start: int,
    end: int,
    guidepost_budget: int,
    halo: int,
) -> List[Tuple[int, int]]:
    """Select prompt-visible sentence spans for natural-language QA."""

    if NODE_RE.search(query_text) or VAR_ASSIGN_RE.search(prompt) or re.search(r"\bVAR\b|variable assignment", query_text, re.I):
        return []
    if not re.search(r"\b(?:Question:|Where|What|Who|Which|Answer:)\b", query_text, re.I):
        return []

    q_terms = _natural_qa_terms(query_text)
    main_entities = _natural_qa_main_entities(query_text)
    if not q_terms and not main_entities:
        return []

    q_start = prompt.rfind("\n\nQuestion:")
    if q_start < 0:
        q_start = prompt.rfind("Question:")
    char_limit = q_start if q_start >= 0 else len(prompt)
    sentences = _sentence_token_spans(prompt, offsets, start, end, char_limit)
    if not sentences:
        return []

    scored = []
    for idx, (s, e, text) in enumerate(sentences):
        terms = _natural_qa_terms(text)
        entities = {entity for entity in main_entities if re.search(rf"\b{re.escape(entity)}\b", text, re.I)}
        overlap = terms & q_terms
        score = 4.0 * len(entities) + 2.0 * len(overlap)
        if entities:
            score += 1.5 * (idx / max(1, len(sentences) - 1))
        if re.search(r"\b(?:went|moved|travelled|journeyed|is|was|located|picked|dropped)\b", text, re.I):
            score += 0.75
        if score > 0:
            scored.append({"idx": idx, "span": (s, e), "score": score, "entities": entities})
    if not scored:
        return []

    selected: list[dict[str, object]] = []
    selected_idx: set[int] = set()
    for entity in main_entities:
        entity_rows = [row for row in scored if entity in row["entities"]]
        if entity_rows:
            row = max(entity_rows, key=lambda r: (int(r["span"][0]), float(r["score"])))
            selected.append(row)
            selected_idx.add(int(row["idx"]))

    small_halo = min(2, halo)
    used = sum(int(row["span"][1]) - int(row["span"][0]) + 2 * small_halo for row in selected)
    for row in sorted(scored, key=lambda r: (-float(r["score"]), int(r["idx"]))):
        if int(row["idx"]) in selected_idx:
            continue
        width = int(row["span"][1]) - int(row["span"][0]) + 2 * small_halo
        if selected and used + width > guidepost_budget:
            continue
        selected.append(row)
        selected_idx.add(int(row["idx"]))
        used += width
        if used >= guidepost_budget:
            break

    spans = [(int(row["span"][0]) - small_halo, int(row["span"][1]) + small_halo) for row in selected]
    return sorted(spans, key=lambda span: span[0])


def compass_prefill_guideposts(
    prompt: str,
    query_text: str,
    token_offsets: Sequence[Tuple[int, int]],
    guidepost_budget: int,
    sink_size: int,
    recent_size: int,
    block_size: int = 16,
    halo: int = 4,
    propagation_rounds: int = 5,
) -> GuidepostPlan:
    """Deployable prompt-text selector for contiguous guidepost spans.

    It uses only prompt/query text available at prefill time: query overlap,
    rare/entity overlap, fact-like line structure, and a small entity
    propagation pass through blocks that introduce new rare entities.
    """

    seq_len = len(token_offsets)
    start = min(max(0, sink_size), seq_len)
    end = max(start, seq_len - max(1, recent_size))
    blocks = [(s, min(end, s + block_size)) for s in range(start, end, block_size)]
    if guidepost_budget <= 0 or not blocks:
        return _plan_from_spans(seq_len, [], 0, sink_size, recent_size)

    q_terms = _terms(query_text)
    q_entities = _entities(query_text)
    chain_spans = _node_chain_spans(
        prompt,
        token_offsets,
        query_text,
        start,
        end,
        halo=halo,
        max_hops=propagation_rounds,
    )
    var_chain_spans = _var_assignment_chain_spans(
        prompt,
        token_offsets,
        query_text,
        start,
        end,
        halo=halo,
        max_hops=propagation_rounds,
    )
    natural_qa_spans = _natural_qa_sentence_spans(
        prompt,
        token_offsets,
        query_text,
        start,
        end,
        guidepost_budget,
        halo=halo,
    )

    line_rows = []
    for idx, (s, e) in enumerate(_line_token_spans(prompt, token_offsets, start, end)):
        text = _block_text(prompt, token_offsets, s, e)
        terms = _terms(text)
        entities = _entities(text)
        overlap = len(terms & q_terms)
        entity_overlap = len(entities & (q_entities | q_terms))
        score = 8.0 * overlap + 10.0 * entity_overlap + 1.5 * _fact_like(text)
        line_rows.append({"idx": idx, "span": (s, e), "text": text, "terms": terms, "entities": entities, "score": score})

    graph_spans: List[Tuple[int, int]] = []
    selected_lines: set[int] = set()
    frontier = set(q_entities) or {t for t in q_terms if re.search(r"\d", t)}
    for _ in range(min(max(1, propagation_rounds), 4)):
        candidates = [
            row
            for row in line_rows
            if row["idx"] not in selected_lines and row["entities"] & frontier and _fact_like(row["text"]) > 0
        ]
        if not candidates:
            break
        added_entities: set[str] = set()
        for row in sorted(candidates, key=lambda r: (-len(r["entities"] & frontier), -r["score"], r["span"][0])):
            selected_lines.add(row["idx"])
            added_entities |= {e for e in row["entities"] if len(e) >= 2}
            s, e = row["span"]
            graph_spans.append((s - halo, e + halo))
        new_entities = added_entities - frontier
        if not new_entities:
            break
        frontier |= new_entities

    block_rows = []
    for idx, (s, e) in enumerate(blocks):
        text = _block_text(prompt, token_offsets, s, e)
        terms = _terms(text)
        entities = _entities(text)
        overlap = len(terms & q_terms)
        entity_overlap = len(entities & (q_entities | q_terms))
        score = 8.0 * overlap + 10.0 * entity_overlap + 1.5 * _fact_like(text)
        block_rows.append({"idx": idx, "span": (s, e), "text": text, "terms": terms, "entities": entities, "score": score})

    selected: set[int] = {r["idx"] for r in block_rows if r["score"] > 0 and (r["terms"] & q_terms or r["entities"] & (q_entities | q_terms))}
    frontier = set(q_entities | q_terms)
    for _ in range(propagation_rounds):
        added = False
        for row in sorted(block_rows, key=lambda r: (-r["score"], r["idx"])):
            shared = (row["entities"] | row["terms"]) & frontier
            if not shared:
                continue
            if _fact_like(row["text"]) <= 0 and row["idx"] not in selected:
                continue
            if row["idx"] not in selected:
                selected.add(row["idx"])
                added = True
            new_terms = {t for t in (row["entities"] | row["terms"]) if len(t) >= 3}
            if not new_terms.issubset(frontier):
                frontier |= new_terms
                added = True
        if not added:
            break

    ranked = sorted(block_rows, key=lambda r: (r["idx"] not in selected, -r["score"], r["idx"]))
    span_budget = max(1, math.ceil(guidepost_budget / max(1, block_size + 2 * halo)))
    top = [r for r in ranked if r["idx"] in selected][: span_budget * 2]
    if len(top) < span_budget:
        top.extend(r for r in ranked if r["idx"] not in {x["idx"] for x in top})
    spans = list(chain_spans) + list(var_chain_spans) + list(natural_qa_spans) + list(graph_spans)
    for row in top:
        s, e = _line_expanded_token_span(prompt, token_offsets, row["span"][0], row["span"][1])
        spans.append((s - halo, e + halo))
    return _plan_from_spans(seq_len, spans, guidepost_budget, sink_size, recent_size)
