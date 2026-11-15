from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from .index import STOPWORDS, lexical_terms
from .schema import EvidenceIndex, PrecisionTier, Proof, ProofOverlay, Relation, RelationType


@dataclass(frozen=True)
class ProofSearchConfig:
    root_top_k: int = 16
    beam_size: int = 8
    # Number of relation edges, not number of evidence spans.  A H-hop proof
    # contains H+1 spans once its terminal evidence is included.
    max_hops: int = 8
    max_proofs: int = 8
    min_spans_per_proof: int = 2
    min_relation_probability: float = 0.20
    length_penalty: float = 0.04
    contradiction_penalty: float = 0.55
    terminal_probability_floor: float = 0.20
    normalize_path_probability: bool = True
    normalization_min_spans: int = 5


def _query_terms(query: str) -> set[str]:
    return {term for term in lexical_terms(query) if term not in STOPWORDS}


def _root_scores(query: str, index: EvidenceIndex) -> dict[str, float]:
    query_terms = _query_terms(query)
    document_frequency: dict[str, int] = {}
    for span in index.spans:
        for term in set(span.cues.get("terms", ())):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    raw: dict[str, float] = {}
    for span in index.spans:
        span_terms = set(span.cues.get("terms", ()))
        overlap = query_terms & span_terms
        lexical = sum(
            math.log((len(index.spans) + 1) / (document_frequency.get(term, 0) + 1)) + 1.0
            for term in overlap
        )
        entity_bonus = 1.5 * len(query_terms & set(span.cues.get("entities", ())))
        raw[span.span_id] = lexical + entity_bonus
    maximum = max(raw.values(), default=0.0)
    if maximum <= 0:
        return {key: 1.0 / max(1, len(raw)) for key in raw}
    return {key: min(1.0, value / maximum) for key, value in raw.items()}


class TypedProofSearcher:
    """Generate typed candidate proof paths without changing the evidence index."""

    def __init__(self, config: ProofSearchConfig | None = None):
        self.config = config or ProofSearchConfig()

    def search(
        self,
        query: str,
        index: EvidenceIndex,
        root_scores: dict[str, float] | None = None,
        terminal_scores: dict[str, float] | None = None,
    ) -> ProofOverlay:
        root_scores = _root_scores(query, index) if root_scores is None else root_scores
        roots = sorted(root_scores, key=lambda span_id: (-root_scores[span_id], span_id))[
            : self.config.root_top_k
        ]
        adjacency = index.adjacency
        # Beam rows: probability, span ids, relation path, root score.
        beam: list[tuple[float, tuple[str, ...], tuple[Relation, ...], float]] = [
            (max(1e-6, root_scores[root]), (root,), (), root_scores[root]) for root in roots
        ]
        candidates = []

        def candidate_probability(
            path_probability: float,
            span_ids: tuple[str, ...],
        ) -> float:
            terminal = 1.0 if terminal_scores is None else terminal_scores.get(span_ids[-1], 0.0)
            joint = max(1e-12, path_probability * terminal)
            if (
                not self.config.normalize_path_probability
                or len(span_ids) < self.config.normalization_min_spans
            ):
                return joint
            # Root, every edge, and the terminal are separate decisions.  Their
            # geometric mean avoids systematically ranking a correct long proof
            # below a shorter false terminal solely because it has more factors.
            component_count = len(span_ids) + 1
            return joint ** (1.0 / component_count)
        if self.config.min_spans_per_proof <= 1:
            candidates = [
                (
                    candidate_probability(probability, span_ids),
                    span_ids,
                    relations,
                    root_score,
                )
                for probability, span_ids, relations, root_score in beam
                if terminal_scores is None
                or terminal_scores.get(span_ids[-1], 0.0) >= self.config.terminal_probability_floor
            ]
        query_has_negation = bool(re.search(r"\b(?:not|never|false|incorrect|except)\b", query, re.I))
        for _ in range(self.config.max_hops):
            expanded: list[tuple[float, tuple[str, ...], tuple[Relation, ...], float]] = []
            for probability, span_ids, relations, root_score in beam:
                for relation in adjacency.get(span_ids[-1], ()):
                    if relation.probability < self.config.min_relation_probability:
                        continue
                    if relation.target in span_ids:
                        continue
                    edge_probability = relation.probability
                    if relation.relation_type == RelationType.CONTRADICTION and not query_has_negation:
                        edge_probability *= self.config.contradiction_penalty
                    # Apply a linear prior over path length.  The previous code
                    # multiplied by exp(-p*j) at step j, accidentally creating a
                    # quadratic penalty p*H*(H-1)/2 against long proofs.
                    length_factor = math.exp(-self.config.length_penalty)
                    expanded.append(
                        (
                            probability * edge_probability * length_factor,
                            span_ids + (relation.target,),
                            relations + (relation,),
                            root_score,
                        )
                    )
            if not expanded:
                break
            expanded.sort(key=lambda row: (-row[0], row[1]))
            beam = expanded[: self.config.beam_size]
            if len(beam[0][1]) >= self.config.min_spans_per_proof:
                candidates.extend(
                    (
                        candidate_probability(probability, span_ids),
                        span_ids,
                        relations,
                        root_score,
                    )
                    for probability, span_ids, relations, root_score in beam
                    if terminal_scores is None
                    or terminal_scores.get(span_ids[-1], 0.0) >= self.config.terminal_probability_floor
                )

        if not candidates:
            candidates = [
                (
                    max(1e-6, root_scores[root])
                    * (1.0 if terminal_scores is None else max(1e-6, terminal_scores.get(root, 0.0))),
                    (root,),
                    (),
                    root_scores[root],
                )
                for root in roots
            ]

        # Prefer compositional proofs when probabilities are close, but keep a one-span alternative.
        candidates.sort(key=lambda row: (-(row[0] * (1.0 + 0.05 * (len(row[1]) - 1))), row[1]))
        unique: list[Proof] = []
        seen: set[tuple[str, ...]] = set()
        for probability, span_ids, relations, root_score in candidates:
            if span_ids in seen:
                continue
            seen.add(span_ids)
            unique.append(
                Proof(
                    proof_id=f"p{len(unique)}",
                    span_ids=span_ids,
                    relations=relations,
                    probability=min(1.0, probability),
                    root_score=root_score,
                )
            )
            if len(unique) >= self.config.max_proofs:
                break
        confidence = 1.0 - math.prod(1.0 - proof.probability for proof in unique)
        return ProofOverlay(
            query,
            tuple(unique),
            root_scores,
            confidence,
            terminal_scores={} if terminal_scores is None else terminal_scores,
        )


def proof_objective(
    proofs: Iterable[Proof],
    selected_span_ids: set[str],
    group_correlated_terminals: bool = False,
) -> float:
    complete = [proof for proof in proofs if set(proof.span_ids) <= selected_span_ids]
    if group_correlated_terminals:
        # Multiple near-duplicate paths ending at the same answer candidate are
        # correlated evidence, not independent OR events.  Counting each path as
        # independent overstates confidence and can crowd out a diverse terminal.
        by_terminal: dict[str, float] = {}
        for proof in complete:
            terminal = proof.span_ids[-1]
            by_terminal[terminal] = max(by_terminal.get(terminal, 0.0), proof.probability)
        probabilities = by_terminal.values()
    else:
        probabilities = (proof.probability for proof in complete)
    failure_probability = 1.0
    for probability in probabilities:
        failure_probability *= 1.0 - probability
    return 1.0 - failure_probability


def proof_progress_objective(
    proofs: Iterable[Proof], selected_span_ids: set[str], power: float = 2.0
) -> float:
    """Probability-weighted progress through complete and incomplete proofs.

    Squaring the retained fraction favors finishing a nearly complete proof while
    still assigning value to useful partial paths when another full proof cannot
    fit in the remaining budget.
    """
    weighted_progress = 0.0
    total_probability = 0.0
    for proof in proofs:
        total_probability += proof.probability
        fraction = len(set(proof.span_ids) & selected_span_ids) / len(proof.span_ids)
        weighted_progress += proof.probability * fraction**power
    return weighted_progress / max(1e-12, total_probability)


class AndOrProofSelector:
    """Budgeted beam search over complete AND proofs under an OR objective."""

    def __init__(
        self,
        max_selected_proofs: int = 8,
        beam_width: int = 128,
        fill_auxiliary: bool = False,
        correlation_aware: bool = False,
        terminal_diversity_weight: float = 0.0,
    ):
        self.max_selected_proofs = max_selected_proofs
        self.beam_width = beam_width
        self.fill_auxiliary = fill_auxiliary
        self.correlation_aware = correlation_aware
        self.terminal_diversity_weight = terminal_diversity_weight

    def select(
        self,
        overlay: ProofOverlay,
        index: EvidenceIndex,
        byte_budget: int,
        values_per_token: int,
        precision: PrecisionTier = PrecisionTier.BF16,
    ) -> ProofOverlay:
        by_id = index.by_id
        # State is (proof ids, selected spans, byte cost). The proof count is small
        # (normally K=4/8), so a bounded subset beam avoids greedy blocking failures.
        states: list[tuple[tuple[str, ...], frozenset[str], int]] = [((), frozenset(), 0)]
        for proof in overlay.proofs[: self.max_selected_proofs]:
            expanded = list(states)
            for proof_ids, selected_spans, _ in states:
                candidate_spans = selected_spans | frozenset(proof.span_ids)
                cost = sum(
                    by_id[span_id].kv_bytes(values_per_token, precision)
                    for span_id in candidate_spans
                )
                if cost <= byte_budget:
                    expanded.append((proof_ids + (proof.proof_id,), candidate_spans, cost))

            def rank(state: tuple[tuple[str, ...], frozenset[str], int]):
                proof_ids, selected_spans, cost = state
                objective = proof_objective(
                    overlay.proofs, set(selected_spans), self.correlation_aware
                )
                efficiency = objective / max(1, cost)
                completed = [
                    proof for proof in overlay.proofs
                    if set(proof.span_ids) <= set(selected_spans)
                ]
                terminal_diversity = len({proof.span_ids[-1] for proof in completed}) / max(
                    1, len({proof.span_ids[-1] for proof in overlay.proofs})
                )
                # Primary objective is probability of at least one complete proof;
                # efficiency and lower bytes break ties.
                return (
                    objective + self.terminal_diversity_weight * terminal_diversity,
                    objective,
                    efficiency,
                    -cost,
                    proof_ids,
                )

            deduplicated = {
                (state[0], state[1]): state for state in expanded
            }
            states = sorted(deduplicated.values(), key=rank, reverse=True)[: self.beam_width]

        selected_proofs, frozen_spans, _ = max(
            states,
            key=lambda state: (
                proof_objective(overlay.proofs, set(state[1]), self.correlation_aware)
                + self.terminal_diversity_weight
                * len({
                    proof.span_ids[-1] for proof in overlay.proofs
                    if set(proof.span_ids) <= set(state[1])
                })
                / max(1, len({proof.span_ids[-1] for proof in overlay.proofs})),
                proof_objective(overlay.proofs, set(state[1]), self.correlation_aware),
                proof_objective(overlay.proofs, set(state[1]), self.correlation_aware)
                / max(1, state[2]),
                -state[2],
            ),
        )
        selected_spans = set(frozen_spans)
        current_objective = proof_objective(
            overlay.proofs, selected_spans, self.correlation_aware
        )
        auxiliary_spans: set[str] = set()
        if self.fill_auxiliary:
            current_cost = sum(
                by_id[span_id].kv_bytes(values_per_token, precision) for span_id in selected_spans
            )
            priority: list[str] = []
            for proof in overlay.proofs:
                for span_id in proof.span_ids:
                    if span_id not in priority:
                        priority.append(span_id)
            priority.extend(
                span_id
                for span_id, _ in sorted(
                    overlay.root_scores.items(), key=lambda item: (-item[1], item[0])
                )
                if span_id not in priority
            )
            for span_id in priority:
                if span_id in selected_spans:
                    continue
                added_cost = by_id[span_id].kv_bytes(values_per_token, precision)
                if current_cost + added_cost > byte_budget:
                    continue
                auxiliary_spans.add(span_id)
                current_cost += added_cost

        return ProofOverlay(
            query=overlay.query,
            proofs=overlay.proofs,
            root_scores=overlay.root_scores,
            confidence=current_objective,
            terminal_scores=overlay.terminal_scores,
            selected_proof_ids=tuple(selected_proofs),
            selected_span_ids=tuple(sorted(selected_spans)),
            auxiliary_span_ids=tuple(sorted(auxiliary_spans)),
        )


class AnytimeProofSelector:
    """Span-level proof knapsack supporting useful partial proof progress.

    Unlike :class:`AndOrProofSelector`, this selector does not require a whole
    candidate proof to fit atomically.  A bounded beam chooses individual spans
    using complete-proof probability plus a soft progress term.  This makes the
    route useful at very tight budgets while retaining the AND/OR completion
    objective and a calibrated complete-proof confidence.
    """

    def __init__(
        self,
        max_candidate_proofs: int = 8,
        max_candidate_spans: int = 32,
        beam_width: int = 128,
        partial_proof_weight: float = 0.25,
        progress_power: float = 2.0,
        fill_auxiliary: bool = False,
        correlation_aware: bool = False,
    ):
        if partial_proof_weight < 0:
            raise ValueError("partial_proof_weight must be non-negative")
        if progress_power <= 0:
            raise ValueError("progress_power must be positive")
        self.max_candidate_proofs = max_candidate_proofs
        self.max_candidate_spans = max_candidate_spans
        self.beam_width = beam_width
        self.partial_proof_weight = partial_proof_weight
        self.progress_power = progress_power
        self.fill_auxiliary = fill_auxiliary
        self.correlation_aware = correlation_aware

    def select(
        self,
        overlay: ProofOverlay,
        index: EvidenceIndex,
        byte_budget: int,
        values_per_token: int,
        precision: PrecisionTier = PrecisionTier.BF16,
    ) -> ProofOverlay:
        by_id = index.by_id
        candidate_ids: list[str] = []
        for proof in overlay.proofs[: self.max_candidate_proofs]:
            for span_id in proof.span_ids:
                if span_id not in candidate_ids:
                    candidate_ids.append(span_id)
        candidate_ids = candidate_ids[: self.max_candidate_spans]

        def rank(state: tuple[frozenset[str], int]):
            selected, cost = state
            selected_set = set(selected)
            complete = proof_objective(
                overlay.proofs, selected_set, self.correlation_aware
            )
            progress = proof_progress_objective(
                overlay.proofs, selected_set, self.progress_power
            )
            utility = complete + self.partial_proof_weight * progress
            return utility, complete, progress, utility / max(1, cost), -cost

        states: list[tuple[frozenset[str], int]] = [(frozenset(), 0)]
        for span_id in candidate_ids:
            added_cost = by_id[span_id].kv_bytes(values_per_token, precision)
            expanded = list(states)
            for selected, cost in states:
                if cost + added_cost <= byte_budget:
                    expanded.append((selected | frozenset((span_id,)), cost + added_cost))
            deduplicated = {state[0]: state for state in expanded}
            states = sorted(deduplicated.values(), key=rank, reverse=True)[: self.beam_width]

        frozen_spans, current_cost = max(states, key=rank)
        selected_spans = set(frozen_spans)
        complete_proofs = tuple(
            proof.proof_id
            for proof in overlay.proofs
            if set(proof.span_ids) <= selected_spans
        )
        current_objective = proof_objective(
            overlay.proofs, selected_spans, self.correlation_aware
        )

        auxiliary_spans: set[str] = set()
        if self.fill_auxiliary:
            priority = candidate_ids + [
                span_id
                for span_id, _ in sorted(
                    overlay.root_scores.items(), key=lambda item: (-item[1], item[0])
                )
                if span_id not in candidate_ids
            ]
            for span_id in priority:
                if span_id in selected_spans:
                    continue
                added_cost = by_id[span_id].kv_bytes(values_per_token, precision)
                if current_cost + added_cost > byte_budget:
                    continue
                auxiliary_spans.add(span_id)
                current_cost += added_cost

        return ProofOverlay(
            query=overlay.query,
            proofs=overlay.proofs,
            root_scores=overlay.root_scores,
            confidence=current_objective,
            terminal_scores=overlay.terminal_scores,
            selected_proof_ids=complete_proofs,
            selected_span_ids=tuple(sorted(selected_spans)),
            auxiliary_span_ids=tuple(sorted(auxiliary_spans)),
        )
