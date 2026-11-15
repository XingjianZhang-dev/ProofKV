from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .schema import EvidenceIndex, Relation, RelationType
from .selector import ProofSearchConfig, TypedProofSearcher


RELATION_LABELS = ("none",) + tuple(value.value for value in RelationType)
RELATION_TO_ID = {label: idx for idx, label in enumerate(RELATION_LABELS)}


def root_text(question: str, span_text: str) -> str:
    return f"[ROOT] Question: {question} [EVIDENCE] {span_text}"


def edge_text(question: str, source_text: str, target_text: str) -> str:
    return f"[EDGE] Question: {question} [SOURCE] {source_text} [TARGET] {target_text}"


def tokenize_edge_batch(
    tokenizer: Any,
    questions: Sequence[str],
    sources: Sequence[str],
    targets: Sequence[str],
    max_length: int,
    edge_encoding: str = "single",
) -> dict[str, torch.Tensor]:
    if edge_encoding == "single":
        return tokenizer(
            [edge_text(question, source, target) for question, source, target in zip(questions, sources, targets)],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    if edge_encoding == "balanced_pair":
        # Pair encoding lets longest-first truncation preserve information from
        # both endpoints when paragraphs are long, instead of silently removing
        # the target from a right-truncated concatenation.
        left = [
            f"[EDGE] Question: {question} [SOURCE] {source}"
            for question, source in zip(questions, sources)
        ]
        right = [f"[TARGET] {target}" for target in targets]
        return tokenizer(
            left,
            right,
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
    raise ValueError(f"unknown edge encoding: {edge_encoding}")


def terminal_text(question: str, span_text: str) -> str:
    return f"[TERMINAL] Question: {question} [EVIDENCE] {span_text}"


class TypedRelationScorer(nn.Module):
    """Small cross-encoder with separate root and typed-edge heads."""

    def __init__(self, encoder_name_or_path: str):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.root_head = nn.Linear(hidden, 1)
        self.terminal_head = nn.Linear(hidden, 1)
        self.relation_head = nn.Linear(hidden, len(RELATION_LABELS))
        self.encoder_name_or_path = encoder_name_or_path
        self.edge_encoding = "single"

    def _pooled(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(**encoded)
        if getattr(output, "pooler_output", None) is not None:
            return self.dropout(output.pooler_output)
        mask = encoded.get("attention_mask")
        hidden = output.last_hidden_state
        if mask is None:
            return self.dropout(hidden[:, 0])
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return self.dropout((hidden * weights).sum(1) / weights.sum(1).clamp_min(1))

    def root_logits(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.root_head(self._pooled(encoded)).squeeze(-1)

    def relation_logits(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.relation_head(self._pooled(encoded))

    def terminal_logits(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.terminal_head(self._pooled(encoded)).squeeze(-1)

    def save_checkpoint(self, output_dir: str | Path, tokenizer: Any, metadata: dict[str, Any]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(output_dir / "encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        torch.save(
            {
                "root_head": self.root_head.state_dict(),
                "terminal_head": self.terminal_head.state_dict(),
                "relation_head": self.relation_head.state_dict(),
            },
            output_dir / "heads.pt",
        )
        payload = {"relation_labels": RELATION_LABELS, **metadata}
        (output_dir / "proofcache_scorer.json").write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load_checkpoint(cls, output_dir: str | Path, device: str | torch.device = "cpu"):
        from transformers import AutoTokenizer

        output_dir = Path(output_dir)
        model = cls(str(output_dir / "encoder"))
        metadata_path = output_dir / "proofcache_scorer.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            model.edge_encoding = str(metadata.get("edge_encoding", "single"))
        heads = torch.load(output_dir / "heads.pt", map_location="cpu", weights_only=True)
        model.root_head.load_state_dict(heads["root_head"])
        if "terminal_head" in heads:
            model.terminal_head.load_state_dict(heads["terminal_head"])
        model.relation_head.load_state_dict(heads["relation_head"])
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(output_dir / "tokenizer", use_fast=True)
        return model, tokenizer


class CachedTypedRelationScorer(nn.Module):
    """Bi-encoder scorer whose document span embeddings are query-agnostic and reusable."""

    def __init__(self, encoder_name_or_path: str, projection_dim: int = 128):
        super().__init__()
        from transformers import AutoModel

        try:
            self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        except ValueError as error:
            # Some older BERT checkpoints (notably prajjwal1/bert-tiny) predate
            # `model_type` in config.json.  The architecture is unambiguous, but
            # Transformers 5.x rejects it through AutoModel.  Loading through the
            # concrete class upgrades the saved config without changing weights.
            if "Unrecognized model" not in str(error):
                raise
            from transformers import BertModel

            self.encoder = BertModel.from_pretrained(encoder_name_or_path)
        hidden = int(self.encoder.config.hidden_size)
        self.projection = nn.Linear(hidden, projection_dim)
        self.root_head = nn.Sequential(
            nn.Linear(projection_dim * 4, projection_dim), nn.GELU(), nn.Linear(projection_dim, 1)
        )
        self.terminal_head = nn.Sequential(
            nn.Linear(projection_dim * 4, projection_dim), nn.GELU(), nn.Linear(projection_dim, 1)
        )
        self.relation_head = nn.Sequential(
            nn.Linear(projection_dim * 8, projection_dim * 2),
            nn.GELU(),
            nn.Linear(projection_dim * 2, len(RELATION_LABELS)),
        )
        self.projection_dim = projection_dim
        self.encoder_name_or_path = encoder_name_or_path

    def encode(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(**encoded)
        hidden = output.last_hidden_state
        mask = encoded.get("attention_mask")
        if mask is None:
            pooled = hidden[:, 0]
        else:
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        return nn.functional.normalize(self.projection(pooled), dim=-1)

    @staticmethod
    def _root_features(query: torch.Tensor, span: torch.Tensor) -> torch.Tensor:
        return torch.cat([query, span, query * span, torch.abs(query - span)], dim=-1)

    @staticmethod
    def _edge_features(query: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                query, source, target,
                query * source, query * target, source * target,
                torch.abs(query - source), torch.abs(source - target),
            ],
            dim=-1,
        )

    def root_logits_from_embeddings(self, query: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        if query.shape[0] == 1 and spans.shape[0] != 1:
            query = query.expand(spans.shape[0], -1)
        return self.root_head(self._root_features(query, spans)).squeeze(-1)

    def relation_logits_from_embeddings(
        self, query: torch.Tensor, sources: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if query.shape[0] == 1 and sources.shape[0] != 1:
            query = query.expand(sources.shape[0], -1)
        return self.relation_head(self._edge_features(query, sources, targets))

    def terminal_logits_from_embeddings(self, query: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        if query.shape[0] == 1 and spans.shape[0] != 1:
            query = query.expand(spans.shape[0], -1)
        return self.terminal_head(self._root_features(query, spans)).squeeze(-1)

    def save_checkpoint(self, output_dir: str | Path, tokenizer: Any, metadata: dict[str, Any]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(output_dir / "encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        torch.save(
            {
                "projection": self.projection.state_dict(),
                "root_head": self.root_head.state_dict(),
                "terminal_head": self.terminal_head.state_dict(),
                "relation_head": self.relation_head.state_dict(),
                "projection_dim": self.projection_dim,
            },
            output_dir / "heads.pt",
        )
        (output_dir / "proofcache_cached_scorer.json").write_text(
            json.dumps({"relation_labels": RELATION_LABELS, **metadata}, indent=2) + "\n"
        )

    @classmethod
    def load_checkpoint(cls, output_dir: str | Path, device: str | torch.device = "cpu"):
        from transformers import AutoTokenizer

        output_dir = Path(output_dir)
        heads = torch.load(output_dir / "heads.pt", map_location="cpu", weights_only=True)
        if heads.get("scorer_type") == "late_interaction":
            return LateInteractionTypedRelationScorer.load_checkpoint(output_dir, device=device)
        model = cls(str(output_dir / "encoder"), projection_dim=int(heads["projection_dim"]))
        model.projection.load_state_dict(heads["projection"])
        model.root_head.load_state_dict(heads["root_head"])
        if "terminal_head" in heads:
            model.terminal_head.load_state_dict(heads["terminal_head"])
        model.relation_head.load_state_dict(heads["relation_head"])
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(output_dir / "tokenizer", use_fast=True)
        return model, tokenizer


class LateInteractionTypedRelationScorer(nn.Module):
    """Cached multi-vector scorer with cheap query-time late interaction.

    Each span is encoded once into a small set of normalized vectors.  Query-span
    and span-span MaxSim statistics preserve token-level alignments that are lost
    by a single pooled bi-encoder vector, while all document-side encoder work
    remains reusable across queries.
    """

    def __init__(
        self,
        encoder_name_or_path: str,
        projection_dim: int = 64,
        num_vectors: int = 8,
    ):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        hidden = int(self.encoder.config.hidden_size)
        self.projection = nn.Linear(hidden, projection_dim)
        pair_dim = projection_dim * 4 + 4
        edge_dim = projection_dim * 8 + 12
        self.root_head = nn.Sequential(
            nn.Linear(pair_dim, projection_dim * 2),
            nn.GELU(),
            nn.Linear(projection_dim * 2, 1),
        )
        self.terminal_head = nn.Sequential(
            nn.Linear(pair_dim, projection_dim * 2),
            nn.GELU(),
            nn.Linear(projection_dim * 2, 1),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(edge_dim, projection_dim * 3),
            nn.GELU(),
            nn.Linear(projection_dim * 3, len(RELATION_LABELS)),
        )
        self.projection_dim = projection_dim
        self.num_vectors = num_vectors
        self.encoder_name_or_path = encoder_name_or_path

    def encode(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.projection(self.encoder(**encoded).last_hidden_state)
        hidden = nn.functional.normalize(hidden, dim=-1)
        mask = encoded.get("attention_mask")
        if mask is None:
            mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        lengths = mask.sum(-1).clamp_min(1)
        vector_ids = torch.arange(self.num_vectors, device=hidden.device).unsqueeze(0)
        starts = torch.div(
            vector_ids * lengths.unsqueeze(1), self.num_vectors, rounding_mode="floor"
        )
        ends = torch.maximum(
            starts + 1,
            torch.div(
                (vector_ids + 1) * lengths.unsqueeze(1),
                self.num_vectors,
                rounding_mode="floor",
            ),
        ).minimum(lengths.unsqueeze(1))
        positions = torch.arange(hidden.shape[1], device=hidden.device).view(1, 1, -1)
        assignments = (
            (positions >= starts.unsqueeze(-1))
            & (positions < ends.unsqueeze(-1))
            & mask.unsqueeze(1).bool()
        ).to(hidden.dtype)
        pooled = torch.einsum("bml,bld->bmd", assignments, hidden)
        counts = assignments.sum(-1).unsqueeze(-1).clamp_min(1)
        return nn.functional.normalize(pooled / counts, dim=-1)

    @staticmethod
    def _maxsim(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        similarities = torch.einsum("bqd,bkd->bqk", left, right)
        left_max = similarities.max(dim=-1).values
        right_max = similarities.max(dim=-2).values
        return torch.stack(
            [left_max.mean(-1), left_max.max(-1).values,
             right_max.mean(-1), right_max.max(-1).values],
            dim=-1,
        )

    @staticmethod
    def _expand_query(query: torch.Tensor, count: int) -> torch.Tensor:
        if query.shape[0] == 1 and count != 1:
            return query.expand(count, -1, -1)
        return query

    def _pair_features(self, query: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        query = self._expand_query(query, spans.shape[0])
        query_mean = query.mean(1)
        span_mean = spans.mean(1)
        return torch.cat(
            [query_mean, span_mean, query_mean * span_mean,
             torch.abs(query_mean - span_mean), self._maxsim(query, spans)],
            dim=-1,
        )

    def root_logits_from_embeddings(self, query: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        return self.root_head(self._pair_features(query, spans)).squeeze(-1)

    def terminal_logits_from_embeddings(self, query: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        return self.terminal_head(self._pair_features(query, spans)).squeeze(-1)

    def relation_logits_from_embeddings(
        self, query: torch.Tensor, sources: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        query = self._expand_query(query, sources.shape[0])
        q = query.mean(1)
        s = sources.mean(1)
        t = targets.mean(1)
        pooled = torch.cat(
            [q, s, t, q * s, q * t, s * t, torch.abs(q - s), torch.abs(s - t)],
            dim=-1,
        )
        interactions = torch.cat(
            [self._maxsim(query, sources), self._maxsim(query, targets),
             self._maxsim(sources, targets)],
            dim=-1,
        )
        return self.relation_head(torch.cat([pooled, interactions], dim=-1))

    def save_checkpoint(self, output_dir: str | Path, tokenizer: Any, metadata: dict[str, Any]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(output_dir / "encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        torch.save(
            {
                "scorer_type": "late_interaction",
                "projection": self.projection.state_dict(),
                "root_head": self.root_head.state_dict(),
                "terminal_head": self.terminal_head.state_dict(),
                "relation_head": self.relation_head.state_dict(),
                "projection_dim": self.projection_dim,
                "num_vectors": self.num_vectors,
            },
            output_dir / "heads.pt",
        )
        (output_dir / "proofcache_cached_scorer.json").write_text(
            json.dumps({"relation_labels": RELATION_LABELS, **metadata}, indent=2) + "\n"
        )

    @classmethod
    def load_checkpoint(cls, output_dir: str | Path, device: str | torch.device = "cpu"):
        from transformers import AutoTokenizer

        output_dir = Path(output_dir)
        heads = torch.load(output_dir / "heads.pt", map_location="cpu", weights_only=True)
        model = cls(
            str(output_dir / "encoder"),
            projection_dim=int(heads["projection_dim"]),
            num_vectors=int(heads["num_vectors"]),
        )
        model.projection.load_state_dict(heads["projection"])
        model.root_head.load_state_dict(heads["root_head"])
        model.terminal_head.load_state_dict(heads["terminal_head"])
        model.relation_head.load_state_dict(heads["relation_head"])
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(output_dir / "tokenizer", use_fast=True)
        return model, tokenizer


@torch.inference_mode()
def score_roots(
    model: TypedRelationScorer,
    tokenizer: Any,
    question: str,
    span_texts: Sequence[str],
    batch_size: int = 64,
    max_length: int = 512,
) -> list[float]:
    device = next(model.parameters()).device
    probabilities: list[float] = []
    texts = [root_text(question, span_text) for span_text in span_texts]
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        probabilities.extend(torch.sigmoid(model.root_logits(encoded)).cpu().tolist())
    return probabilities


@torch.inference_mode()
def score_terminals(
    model: TypedRelationScorer,
    tokenizer: Any,
    question: str,
    span_texts: Sequence[str],
    batch_size: int = 64,
    max_length: int = 512,
) -> list[float]:
    device = next(model.parameters()).device
    probabilities: list[float] = []
    texts = [terminal_text(question, span_text) for span_text in span_texts]
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        probabilities.extend(torch.sigmoid(model.terminal_logits(encoded)).cpu().tolist())
    return probabilities


@torch.inference_mode()
def score_edges(
    model: TypedRelationScorer,
    tokenizer: Any,
    question: str,
    pairs: Sequence[tuple[str, str]],
    batch_size: int = 64,
    max_length: int = 512,
) -> list[tuple[RelationType | None, float]]:
    device = next(model.parameters()).device
    outputs: list[tuple[RelationType | None, float]] = []
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        encoded = tokenize_edge_batch(
            tokenizer,
            [question] * len(batch_pairs),
            [source for source, _ in batch_pairs],
            [target for _, target in batch_pairs],
            max_length,
            edge_encoding=getattr(model, "edge_encoding", "single"),
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        probabilities = torch.softmax(model.relation_logits(encoded), dim=-1).cpu()
        values, labels = probabilities.max(dim=-1)
        for label, probability in zip(labels.tolist(), values.tolist()):
            outputs.append((None if label == 0 else RelationType(RELATION_LABELS[label]), float(probability)))
    return outputs


def build_learned_overlay(
    model: TypedRelationScorer,
    tokenizer: Any,
    question: str,
    index: EvidenceIndex,
    candidate_span_count: int = 32,
    edge_probability_floor: float = 0.35,
    search_config: ProofSearchConfig | None = None,
    candidate_strategy: str = "frontier",
    dense_core_count: int = 16,
    edge_batch_size: int = 64,
):
    if dense_core_count < 0:
        raise ValueError("dense_core_count must be non-negative")
    if edge_batch_size <= 0:
        raise ValueError("edge_batch_size must be positive")
    spans = list(index.spans)
    root_probabilities = score_roots(model, tokenizer, question, [span.text for span in spans])
    root_scores = {span.span_id: probability for span, probability in zip(spans, root_probabilities)}
    terminal_probabilities = score_terminals(model, tokenizer, question, [span.text for span in spans])
    terminal_scores = {
        span.span_id: probability for span, probability in zip(spans, terminal_probabilities)
    }
    if candidate_strategy == "frontier":
        candidate_ids = _frontier_candidate_ids(
            index, root_scores, terminal_scores, candidate_span_count
        )
    elif candidate_strategy == "root":
        candidate_ids = sorted(
            root_scores, key=lambda span_id: (-root_scores[span_id], span_id)
        )[:candidate_span_count]
    else:
        raise ValueError(f"unknown candidate strategy: {candidate_strategy}")
    candidate_set = set(candidate_ids)
    by_id = index.by_id
    pair_ids: list[tuple[str, str]] = []
    # Always score candidate relations from the query-agnostic index.
    for relation in index.relations:
        if relation.source in candidate_set and relation.target in candidate_set:
            pair_ids.append((relation.source, relation.target))
    # Add a fully-connected high-root core so a missing heuristic edge cannot block learning.
    dense_core = candidate_ids[: min(dense_core_count, len(candidate_ids))]
    pair_ids.extend((source, target) for source in dense_core for target in dense_core if source != target)
    pair_ids = list(dict.fromkeys(pair_ids))
    predictions = score_edges(
        model,
        tokenizer,
        question,
        [(by_id[source].text, by_id[target].text) for source, target in pair_ids],
        batch_size=edge_batch_size,
    )
    relations = [
        Relation(source, target, relation_type, probability)
        for (source, target), (relation_type, probability) in zip(pair_ids, predictions)
        if relation_type is not None and probability >= edge_probability_floor
    ]
    learned_index = EvidenceIndex(
        index.document_id,
        index.spans,
        tuple(relations),
        index.token_count,
        index_version=f"{index.index_version}+learned-relations",
    )
    overlay = TypedProofSearcher(search_config).search(
        question, learned_index, root_scores=root_scores, terminal_scores=terminal_scores
    )
    return overlay, learned_index


def _frontier_candidate_ids(
    index: EvidenceIndex,
    root_scores: dict[str, float],
    terminal_scores: dict[str, float],
    candidate_span_count: int,
) -> list[str]:
    """Keep roots, terminals, and graph bridges before expensive edge scoring.

    Root-only truncation is unsafe for compositional retrieval: intermediate and
    answer-bearing spans are often deliberately dissimilar to the question.  We
    seed both ends and expand over the reusable query-agnostic evidence graph.
    """
    if candidate_span_count <= 0:
        return []
    root_order = sorted(root_scores, key=lambda key: (-root_scores[key], key))
    terminal_order = sorted(terminal_scores, key=lambda key: (-terminal_scores[key], key))
    seed_count = max(1, min(8, candidate_span_count // 4))
    selected: list[str] = []

    def add(span_id: str) -> bool:
        if span_id in selected or len(selected) >= candidate_span_count:
            return False
        selected.append(span_id)
        return True

    for span_id in root_order[:seed_count]:
        add(span_id)
    for span_id in terminal_order[:seed_count]:
        add(span_id)

    # Treat the static graph as undirected only for candidate discovery.  The
    # learned scorer still predicts directed typed edges used by proof search.
    graph_edges = sorted(
        index.relations,
        key=lambda relation: (-relation.probability, relation.source, relation.target),
    )
    frontier = set(selected)
    while len(selected) < candidate_span_count:
        added_this_round: list[str] = []
        for relation in graph_edges:
            if relation.source in frontier:
                if add(relation.target):
                    added_this_round.append(relation.target)
            elif relation.target in frontier:
                if add(relation.source):
                    added_this_round.append(relation.source)
            if len(selected) >= candidate_span_count:
                break
        if not added_this_round:
            break
        frontier.update(added_this_round)

    # Disconnected evidence remains reachable through the learned dense core.
    for rank in range(max(len(root_order), len(terminal_order))):
        if len(selected) >= candidate_span_count:
            break
        if rank < len(root_order):
            add(root_order[rank])
        if rank < len(terminal_order):
            add(terminal_order[rank])
    return selected


@torch.inference_mode()
def encode_cached_spans(
    model: CachedTypedRelationScorer,
    tokenizer: Any,
    span_texts: Sequence[str],
    batch_size: int = 128,
    max_length: int = 192,
) -> torch.Tensor:
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(span_texts), batch_size):
        encoded = tokenizer(
            list(span_texts[start : start + batch_size]), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs.append(model.encode(encoded))
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, model.projection_dim), device=device)


@torch.inference_mode()
def build_cached_overlay(
    model: CachedTypedRelationScorer,
    tokenizer: Any,
    question: str,
    index: EvidenceIndex,
    span_embeddings: torch.Tensor,
    candidate_span_count: int = 32,
    edge_probability_floor: float = 0.35,
    search_config: ProofSearchConfig | None = None,
    candidate_strategy: str = "frontier",
    relation_prior_weight: float = 0.0,
    edge_non_none_floor: float | None = None,
):
    if not 0.0 <= relation_prior_weight <= 1.0:
        raise ValueError("relation_prior_weight must lie in [0, 1]")
    if edge_non_none_floor is not None and not 0.0 <= edge_non_none_floor <= 1.0:
        raise ValueError("edge_non_none_floor must lie in [0, 1]")
    device = next(model.parameters()).device
    query_encoded = tokenizer(
        [question], padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    query_encoded = {key: value.to(device) for key, value in query_encoded.items()}
    query_embedding = model.encode(query_encoded)
    root_probabilities = torch.sigmoid(
        model.root_logits_from_embeddings(query_embedding, span_embeddings)
    )
    spans = list(index.spans)
    root_scores = {
        span.span_id: float(probability)
        for span, probability in zip(spans, root_probabilities.tolist())
    }
    terminal_probabilities = torch.sigmoid(
        model.terminal_logits_from_embeddings(query_embedding, span_embeddings)
    )
    terminal_scores = {
        span.span_id: float(probability)
        for span, probability in zip(spans, terminal_probabilities.tolist())
    }
    position = {span.span_id: idx for idx, span in enumerate(spans)}
    if candidate_strategy == "frontier":
        candidate_ids = _frontier_candidate_ids(
            index, root_scores, terminal_scores, candidate_span_count
        )
    elif candidate_strategy == "root":
        candidate_ids = sorted(
            root_scores, key=lambda span_id: (-root_scores[span_id], span_id)
        )[:candidate_span_count]
    else:
        raise ValueError(f"unknown candidate strategy: {candidate_strategy}")
    candidate_set = set(candidate_ids)
    pair_ids = [
        (relation.source, relation.target)
        for relation in index.relations
        if relation.source in candidate_set and relation.target in candidate_set
    ]
    dense_core = candidate_ids[: min(16, len(candidate_ids))]
    pair_ids.extend((source, target) for source in dense_core for target in dense_core if source != target)
    pair_ids = list(dict.fromkeys(pair_ids))
    static_by_pair: dict[tuple[str, str], Relation] = {}
    for relation in index.relations:
        pair = (relation.source, relation.target)
        if pair not in static_by_pair or relation.probability > static_by_pair[pair].probability:
            static_by_pair[pair] = relation
    if pair_ids:
        source_embeddings = torch.stack([span_embeddings[position[source]] for source, _ in pair_ids])
        target_embeddings = torch.stack([span_embeddings[position[target]] for _, target in pair_ids])
        probabilities = torch.softmax(
            model.relation_logits_from_embeddings(query_embedding, source_embeddings, target_embeddings),
            dim=-1,
        )
        values, labels = probabilities.max(dim=-1)
        relations = []
        for pair, label, probability, distribution in zip(
            pair_ids, labels.tolist(), values.tolist(), probabilities.tolist()
        ):
            source, target = pair
            static = static_by_pair.get(pair)
            non_none_probability = 1.0 - float(distribution[0])
            if static is not None and relation_prior_weight > 0:
                blended_probability = (
                    relation_prior_weight * static.probability
                    + (1.0 - relation_prior_weight) * non_none_probability
                )
                # The static relation is a query-agnostic connectivity prior;
                # the cached student reweights it but cannot erase the edge
                # solely by collapsing to the NONE class.
                if blended_probability >= min(edge_probability_floor, 0.20):
                    relations.append(
                        Relation(
                            source,
                            target,
                            static.relation_type,
                            min(1.0, blended_probability),
                            static.evidence,
                        )
                    )
            elif edge_non_none_floor is not None:
                # NONE may remain the argmax under severe edge imbalance even
                # when aggregate relation mass carries useful signal.  Calibrate
                # the binary relation decision separately from relation typing.
                positive_label = max(
                    range(1, len(distribution)), key=lambda idx: distribution[idx]
                )
                if non_none_probability >= edge_non_none_floor:
                    relations.append(
                        Relation(
                            source,
                            target,
                            RelationType(RELATION_LABELS[positive_label]),
                            non_none_probability,
                        )
                    )
            elif label != 0 and probability >= edge_probability_floor:
                relations.append(
                    Relation(
                        source,
                        target,
                        RelationType(RELATION_LABELS[label]),
                        float(probability),
                    )
                )
    else:
        relations = []
    learned_index = EvidenceIndex(
        index.document_id,
        index.spans,
        tuple(relations),
        index.token_count,
        index_version=(
            f"{index.index_version}+cached-prior-{relation_prior_weight:.2f}"
            if relation_prior_weight > 0
            else (
                f"{index.index_version}+cached-non-none-{edge_non_none_floor:.2f}"
                if edge_non_none_floor is not None
                else f"{index.index_version}+cached-learned-relations"
            )
        ),
    )
    overlay = TypedProofSearcher(search_config).search(
        question, learned_index, root_scores=root_scores, terminal_scores=terminal_scores
    )
    return overlay, learned_index
