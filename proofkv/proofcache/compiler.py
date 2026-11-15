from __future__ import annotations

from dataclasses import dataclass

from .schema import CompiledMemoryPlan, EvidenceIndex, PrecisionTier, ProofOverlay


@dataclass(frozen=True)
class CompilerConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    sink_tokens: int = 4
    recent_tokens: int = 32
    halo_tokens: int = 8
    core_precision: PrecisionTier = PrecisionTier.BF16
    halo_precision: PrecisionTier = PrecisionTier.INT8
    auxiliary_precision: PrecisionTier = PrecisionTier.INT4
    max_auxiliary_spans_per_layer: int = 2

    @property
    def values_per_token_all_layers(self) -> int:
        return self.num_layers * 2 * self.num_kv_heads * self.head_dim

    @property
    def values_per_token_one_layer(self) -> int:
        return 2 * self.num_kv_heads * self.head_dim


class ByteBudgetCompiler:
    """Compile one proof overlay into an all-layer core plus per-layer residual plan."""

    def __init__(self, config: CompilerConfig):
        self.config = config

    def compile(
        self,
        overlay: ProofOverlay,
        index: EvidenceIndex,
        byte_budget: int,
        fallback_to_full: bool = False,
    ) -> CompiledMemoryPlan:
        cfg = self.config
        by_id = index.by_id
        full_tokens = set(range(index.token_count))
        mandatory = set(range(min(cfg.sink_tokens, index.token_count)))
        mandatory |= set(range(max(0, index.token_count - cfg.recent_tokens), index.token_count))
        core_ids = tuple(span_id for span_id in overlay.selected_span_ids if span_id in by_id)
        core_tokens = {token for span_id in core_ids for token in range(by_id[span_id].token_start, by_id[span_id].token_end)}
        bf16_tokens = mandatory | core_tokens

        if fallback_to_full:
            byte_cost = self._all_layer_cost(full_tokens, PrecisionTier.BF16)
            return CompiledMemoryPlan(
                proof_core_span_ids=core_ids,
                auxiliary_span_ids_by_layer={},
                retained_token_indices=tuple(sorted(full_tokens)),
                precision_by_span={span_id: PrecisionTier.BF16 for span_id in core_ids},
                byte_cost=byte_cost,
                byte_budget=byte_budget,
                confidence=overlay.confidence,
                fallback_to_full=True,
                metadata={"reason": "risk_controller_fallback"},
            )

        halo_tokens: set[int] = set()
        for span_id in core_ids:
            span = by_id[span_id]
            halo_tokens.update(
                range(
                    max(0, span.token_start - cfg.halo_tokens),
                    min(index.token_count, span.token_end + cfg.halo_tokens),
                )
            )
        halo_tokens -= bf16_tokens
        base_cost = self._all_layer_cost(bf16_tokens, cfg.core_precision)
        base_cost += self._all_layer_cost(halo_tokens, cfg.halo_precision)
        if base_cost > byte_budget:
            return CompiledMemoryPlan(
                proof_core_span_ids=core_ids,
                auxiliary_span_ids_by_layer={},
                retained_token_indices=tuple(sorted(full_tokens)),
                precision_by_span={span_id: cfg.core_precision for span_id in core_ids},
                byte_cost=self._all_layer_cost(full_tokens, PrecisionTier.BF16),
                byte_budget=byte_budget,
                confidence=overlay.confidence,
                fallback_to_full=True,
                metadata={"reason": "proof_core_exceeds_budget", "requested_core_cost": base_cost},
            )

        candidates = list(overlay.auxiliary_span_ids)
        candidates.extend(
            span_id
            for span_id, _ in sorted(overlay.root_scores.items(), key=lambda item: (-item[1], item[0]))
            if span_id not in core_ids and span_id not in candidates
        )
        auxiliary: dict[int, tuple[str, ...]] = {}
        auxiliary_tokens_by_layer: dict[int, set[int]] = {}
        running_cost = base_cost
        for layer in range(cfg.num_layers):
            chosen: list[str] = []
            layer_tokens: set[int] = set()
            for span_id in candidates:
                if len(chosen) >= cfg.max_auxiliary_spans_per_layer:
                    break
                span = by_id[span_id]
                new_tokens = set(range(span.token_start, span.token_end)) - bf16_tokens - halo_tokens - layer_tokens
                added_cost = self._one_layer_cost(new_tokens, cfg.auxiliary_precision)
                if running_cost + added_cost > byte_budget:
                    continue
                chosen.append(span_id)
                layer_tokens.update(new_tokens)
                running_cost += added_cost
            if chosen:
                auxiliary[layer] = tuple(chosen)
                auxiliary_tokens_by_layer[layer] = layer_tokens

        retained = bf16_tokens | halo_tokens
        for tokens in auxiliary_tokens_by_layer.values():
            retained |= tokens
        precision_by_span = {span_id: cfg.core_precision for span_id in core_ids}
        for span_ids in auxiliary.values():
            for span_id in span_ids:
                precision_by_span.setdefault(span_id, cfg.auxiliary_precision)
        return CompiledMemoryPlan(
            proof_core_span_ids=core_ids,
            auxiliary_span_ids_by_layer=auxiliary,
            retained_token_indices=tuple(sorted(retained)),
            precision_by_span=precision_by_span,
            byte_cost=running_cost,
            byte_budget=byte_budget,
            confidence=overlay.confidence,
            metadata={
                "bf16_tokens": len(bf16_tokens),
                "int8_halo_tokens": len(halo_tokens),
                "auxiliary_layer_token_copies": sum(len(value) for value in auxiliary_tokens_by_layer.values()),
            },
        )

    def _all_layer_cost(self, tokens: set[int], precision: PrecisionTier) -> int:
        return int(round(len(tokens) * self.config.values_per_token_all_layers * precision.bytes_per_value))

    def _one_layer_cost(self, tokens: set[int], precision: PrecisionTier) -> int:
        return int(round(len(tokens) * self.config.values_per_token_one_layer * precision.bytes_per_value))
