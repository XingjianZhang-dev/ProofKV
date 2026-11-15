from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .config import CompassConfig, Route
from .guideposts import select_guideposts_from_scores
from .masks import navigation_keep_mask, streaming_keep_mask
from .routing import RouteTable, project_to_budget, route_errors_to_table


@dataclass
class CalibrationAccumulator:
    """Accumulates retained attention mass for route sufficiency calibration."""

    cfg: CompassConfig
    streaming_sum: Optional[torch.Tensor] = None
    navigation_sum: Optional[torch.Tensor] = None
    count: int = 0

    def update_from_attentions(self, attentions: Sequence[torch.Tensor]) -> None:
        """Update from one model call's attention tensors.

        Each attention tensor must be [batch, query_heads, q_len, kv_len].  Use
        this on decode/probe calls where q_len is small; it avoids storing full
        prompt attention matrices.
        """

        if not attentions:
            raise ValueError("No attentions were provided. Use attn_implementation='eager' if needed.")
        layer_stream: List[torch.Tensor] = []
        layer_nav: List[torch.Tensor] = []
        for layer_attn in attentions:
            if layer_attn.ndim != 4:
                raise ValueError(f"attention tensor must be [B,H,Q,K], got {tuple(layer_attn.shape)}")
            attn = layer_attn.detach().float().cpu()
            bsz, heads, q_len, kv_len = attn.shape
            stream_mask = streaming_keep_mask(q_len, kv_len, self.cfg.sink_size, self.cfg.recent_size, device="cpu")
            # Streaming mass: average over batch and query rows.
            stream_mass = (attn * stream_mask.view(1, 1, q_len, kv_len)).sum(dim=-1).mean(dim=(0, 2))

            nav_mass_heads = []
            for h in range(heads):
                masses = []
                for b in range(bsz):
                    # For calibration we use observed full attention as the
                    # guidepost saliency. This tests memory sufficiency; runtime
                    # selectors may use blockmax QK or keynorm instead.
                    scores = attn[b, h].mean(dim=0)
                    gp = select_guideposts_from_scores(
                        scores,
                        self.cfg.guidepost_count,
                        self.cfg.sink_size,
                        self.cfg.recent_size,
                        self.cfg.block_size,
                        self.cfg.candidates_per_block,
                    ).indices
                    nav_mask = navigation_keep_mask(q_len, kv_len, self.cfg.sink_size, self.cfg.recent_size, gp, device="cpu")
                    masses.append((attn[b, h] * nav_mask).sum(dim=-1).mean())
                nav_mass_heads.append(torch.stack(masses).mean())
            nav_mass = torch.stack(nav_mass_heads)
            layer_stream.append(stream_mass)
            layer_nav.append(nav_mass)

        stream = torch.stack(layer_stream, dim=0)
        nav = torch.stack(layer_nav, dim=0)
        if self.streaming_sum is None:
            self.streaming_sum = torch.zeros_like(stream)
            self.navigation_sum = torch.zeros_like(nav)
        if self.streaming_sum.shape != stream.shape:
            raise ValueError("Attention head shape changed across calibration examples")
        self.streaming_sum += stream
        self.navigation_sum += nav
        self.count += 1

    def means(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0 or self.streaming_sum is None or self.navigation_sum is None:
            raise RuntimeError("No calibration examples accumulated")
        return self.streaming_sum / self.count, self.navigation_sum / self.count

    def to_route_table(
        self,
        threshold: Optional[float] = None,
        target_budget_ratio: Optional[float] = None,
        seq_len: Optional[int] = None,
    ) -> RouteTable:
        stream, nav = self.means()
        table = route_errors_to_table(
            stream,
            nav,
            self.cfg,
            threshold=threshold,
            min_full_heads_per_layer=self.cfg.min_full_heads_per_layer,
        )
        if target_budget_ratio is not None:
            if seq_len is None:
                raise ValueError("seq_len must be provided when projecting to a budget")
            table = project_to_budget(table, self.cfg, seq_len, target_budget_ratio, stream, nav)
        meta = dict(table.metadata)
        meta.update(
            {
                "num_examples": self.count,
                "route_counts": table.counts(),
                "streaming_mass_shape": list(stream.shape),
            }
        )
        table.metadata = meta
        return table
