from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import CompassConfig, Route
from .routing import RouteTable, project_query_to_kv_groups


@dataclass
class RuntimeStats:
    calls: int = 0
    routed_calls: int = 0
    full_calls: int = 0
    tokens_seen: int = 0
    kept_tokens_sum: int = 0
    route_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "routed_calls": self.routed_calls,
            "full_calls": self.full_calls,
            "tokens_seen": self.tokens_seen,
            "kept_tokens_sum": self.kept_tokens_sum,
            "route_counts": dict(self.route_counts),
            "avg_kept_per_head_query": self.kept_tokens_sum / max(1, self.tokens_seen),
        }


class CompassKVController:
    """Runtime controller shared by patched attention modules."""

    def __init__(self, cfg: CompassConfig, route_table: Optional[RouteTable] = None) -> None:
        self.cfg = cfg
        if route_table is None and cfg.route_table_path:
            route_table = RouteTable.load_json(cfg.route_table_path)
        self.route_table = route_table
        self.enabled = True
        self.stats = RuntimeStats()
        self.fixed_guidepost_indices: Optional[torch.Tensor] = None

    @classmethod
    def from_files(cls, config_path: str | Path, route_table_path: Optional[str | Path] = None) -> "CompassKVController":
        cfg = CompassConfig.from_yaml(config_path)
        table = RouteTable.load_json(route_table_path) if route_table_path else None
        return cls(cfg, table)

    def reset_stats(self) -> None:
        self.stats = RuntimeStats()

    def set_fixed_guidepost_indices(self, indices: torch.Tensor | list[int] | tuple[int, ...]) -> None:
        self.fixed_guidepost_indices = torch.as_tensor(indices, dtype=torch.long).flatten().detach().cpu()

    def clear_fixed_guidepost_indices(self) -> None:
        self.fixed_guidepost_indices = None

    def fixed_guidepost_map(
        self,
        batch_size: int,
        num_query_heads: int,
        kv_len: int,
        device: torch.device | str,
    ) -> Optional[Dict[tuple[int, int], torch.Tensor]]:
        if self.fixed_guidepost_indices is None:
            return None
        idx = self.fixed_guidepost_indices.to(device=device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < kv_len)]
        if idx.numel() == 0:
            return None
        idx = torch.unique(idx, sorted=True)
        return {(b, h): idx for b in range(batch_size) for h in range(num_query_heads)}

    def should_route(self, q_len: int) -> bool:
        if not self.enabled:
            return False
        if self.cfg.apply_during_prefill:
            return True
        return q_len <= self.cfg.decode_q_len_threshold

    def get_query_head_routes(
        self,
        layer_idx: int,
        num_query_heads: int,
        device: torch.device | str,
        num_kv_heads: Optional[int] = None,
    ) -> torch.Tensor:
        if self.route_table is None:
            r = int(self.cfg.default_route_enum)
            return torch.full((num_query_heads,), r, dtype=torch.long, device=device)
        if layer_idx >= self.route_table.num_layers:
            raise IndexError(
                f"Layer {layer_idx} requested but route table has {self.route_table.num_layers} layers"
            )
        routes = self.route_table.layer_routes(layer_idx, device=device)
        if routes.numel() == num_query_heads:
            return routes
        if num_kv_heads is not None and routes.numel() == num_kv_heads:
            if num_query_heads % num_kv_heads != 0:
                raise ValueError("num_query_heads must be divisible by num_kv_heads")
            rep = num_query_heads // num_kv_heads
            return routes.repeat_interleave(rep)
        raise ValueError(
            f"Route table head count {routes.numel()} does not match query heads {num_query_heads}"
        )

    def record_call(self, q_len: int, kv_len: int, routed: bool, routes: Optional[torch.Tensor] = None, kept_counts: Optional[torch.Tensor] = None) -> None:
        self.stats.calls += 1
        if routed:
            self.stats.routed_calls += 1
        else:
            self.stats.full_calls += 1
        if routes is not None:
            for r in routes.detach().cpu().flatten().tolist():
                self.stats.route_counts[Route(int(r)).label] += 1
        if kept_counts is not None:
            self.stats.tokens_seen += int(kept_counts.numel())
            self.stats.kept_tokens_sum += int(kept_counts.detach().cpu().long().sum().item())

    def theoretical_memory_ratio(self, seq_len: int) -> Optional[float]:
        if self.route_table is None:
            counts = {r.label: 0 for r in Route}
            counts[self.cfg.default_route_enum.label] = 1
            return self.cfg.memory_ratio_for_counts(counts, seq_len)
        return self.route_table.memory_ratio(self.cfg, seq_len)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.cfg.to_dict(),
            "route_counts": self.route_table.counts() if self.route_table else None,
            "stats": self.stats.to_dict(),
        }
