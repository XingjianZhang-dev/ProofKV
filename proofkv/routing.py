from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .config import CompassConfig, Route


@dataclass
class RouteTable:
    """Persistent route assignment table.

    ``routes`` has shape ``[num_layers, num_heads]`` and stores integer
    ``Route`` values.  The table normally uses query heads in the reference
    mask implementation; for physical GQA cache pruning, project query-head
    routes to KV groups with ``project_query_to_kv_groups``.
    """

    routes: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.routes, torch.Tensor):
            self.routes = torch.tensor(self.routes, dtype=torch.long)
        self.routes = self.routes.detach().cpu().long()
        if self.routes.ndim != 2:
            raise ValueError(f"routes must have shape [layers, heads], got {tuple(self.routes.shape)}")
        valid = (self.routes >= min(r.value for r in Route)) & (self.routes <= max(r.value for r in Route))
        if not torch.all(valid):
            raise ValueError("routes contains invalid Route values")

    @property
    def num_layers(self) -> int:
        return int(self.routes.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.routes.shape[1])

    def layer_routes(self, layer_idx: int, device: torch.device | str | None = None) -> torch.Tensor:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} outside [0, {self.num_layers})")
        out = self.routes[layer_idx]
        return out.to(device=device) if device is not None else out.clone()

    def counts(self) -> Dict[str, int]:
        counts = {r.label: int((self.routes == int(r)).sum().item()) for r in Route}
        return counts

    def layer_counts(self) -> List[Dict[str, int]]:
        return [RouteTable(self.routes[i : i + 1]).counts() for i in range(self.num_layers)]

    def memory_ratio(self, cfg: CompassConfig, seq_len: int) -> float:
        return cfg.memory_ratio_for_counts(self.counts(), seq_len)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "routes": self.routes.tolist(),
            "route_labels": {str(int(r)): r.label for r in Route},
            "metadata": self.metadata,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "RouteTable":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(torch.tensor(payload["routes"], dtype=torch.long), metadata=payload.get("metadata", {}))

    @classmethod
    def full(cls, num_layers: int, num_heads: int, metadata: Optional[Dict[str, Any]] = None) -> "RouteTable":
        return cls(torch.full((num_layers, num_heads), int(Route.FULL), dtype=torch.long), metadata or {})

    @classmethod
    def streaming(cls, num_layers: int, num_heads: int, metadata: Optional[Dict[str, Any]] = None) -> "RouteTable":
        return cls(torch.full((num_layers, num_heads), int(Route.STREAM), dtype=torch.long), metadata or {})

    @classmethod
    def navigation(cls, num_layers: int, num_heads: int, metadata: Optional[Dict[str, Any]] = None) -> "RouteTable":
        return cls(torch.full((num_layers, num_heads), int(Route.NAV), dtype=torch.long), metadata or {})


def route_errors_to_table(
    streaming_mass: torch.Tensor,
    navigation_mass: torch.Tensor,
    cfg: CompassConfig,
    threshold: Optional[float] = None,
    min_full_heads_per_layer: int = 0,
) -> RouteTable:
    """Convert retained-attention mass into cheapest sufficient routes.

    The rule is intentionally transparent:
      1. if streaming mass >= threshold -> STREAM;
      2. else if navigation mass >= threshold -> NAV;
      3. else -> FULL.

    ``streaming_mass`` and ``navigation_mass`` must be [layers, heads] and are
    typically averaged over a calibration suite.  The threshold defaults to
    ``cfg.calibration_threshold``.
    """

    if streaming_mass.shape != navigation_mass.shape or streaming_mass.ndim != 2:
        raise ValueError("streaming_mass and navigation_mass must share shape [layers, heads]")
    threshold = cfg.calibration_threshold if threshold is None else float(threshold)
    routes = torch.full(streaming_mass.shape, int(Route.FULL), dtype=torch.long)
    routes[navigation_mass >= threshold] = int(Route.NAV)
    routes[streaming_mass >= threshold] = int(Route.STREAM)

    if min_full_heads_per_layer > 0:
        # Preserve the heads with largest full requirement: low navigation mass.
        for layer in range(routes.shape[0]):
            full_now = int((routes[layer] == int(Route.FULL)).sum().item())
            if full_now >= min_full_heads_per_layer:
                continue
            need = min_full_heads_per_layer - full_now
            candidates = torch.nonzero(routes[layer] != int(Route.FULL), as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            difficulty = 1.0 - navigation_mass[layer, candidates]
            _, order = torch.topk(difficulty, k=min(need, candidates.numel()), largest=True)
            routes[layer, candidates[order]] = int(Route.FULL)

    metadata = {
        "calibration_threshold": threshold,
        "streaming_mass_mean": float(streaming_mass.mean().item()),
        "navigation_mass_mean": float(navigation_mass.mean().item()),
        "method": "retained_attention_mass_cheapest_sufficient",
    }
    return RouteTable(routes, metadata=metadata)


def project_query_to_kv_groups(routes: torch.Tensor, num_kv_heads: int, prefer_more_memory: bool = True) -> torch.Tensor:
    """Project query-head routes to KV-head groups for physical GQA caches.

    For a KV group serving several query heads, the physical cache must support
    the most memory-demanding route among its query heads unless a grouped-kernel
    supports query-specific masking.  ``prefer_more_memory=True`` therefore uses
    min integer route id (FULL < NAV < STREAM).
    """

    if routes.ndim != 2:
        raise ValueError("routes must be [layers, query_heads]")
    n_layers, n_q = routes.shape
    if n_q % num_kv_heads != 0:
        raise ValueError(f"query heads {n_q} not divisible by kv heads {num_kv_heads}")
    group = n_q // num_kv_heads
    reshaped = routes.reshape(n_layers, num_kv_heads, group)
    if prefer_more_memory:
        return reshaped.min(dim=-1).values
    return reshaped.mode(dim=-1).values


def project_to_budget(
    table: RouteTable,
    cfg: CompassConfig,
    seq_len: int,
    target_ratio: float,
    streaming_mass: Optional[torch.Tensor] = None,
    navigation_mass: Optional[torch.Tensor] = None,
) -> RouteTable:
    """Greedily adjust routes to meet a theoretical memory budget.

    If the table already fits the budget, it is returned unchanged.  Otherwise
    heads are downgraded by least expected damage: FULL->NAV uses high
    navigation mass; NAV->STREAM uses high streaming mass.  This projection is
    deterministic and designed for budget sweeps, not as the primary claim.
    """

    if not (0 < target_ratio <= 1.0):
        raise ValueError("target_ratio must be in (0, 1]")
    routes = table.routes.clone()
    n_layers, n_heads = routes.shape
    total_heads = n_layers * n_heads

    def ratio(rt: torch.Tensor) -> float:
        counts = {r.label: int((rt == int(r)).sum().item()) for r in Route}
        return cfg.memory_ratio_for_counts(counts, seq_len)

    current = ratio(routes)
    if current <= target_ratio:
        meta = dict(table.metadata)
        meta["budget_projection"] = {"target_ratio": target_ratio, "initial_ratio": current, "final_ratio": current}
        return RouteTable(routes, meta)

    # Default masses if not provided: stable but conservative ordering.
    if streaming_mass is None:
        streaming_mass = torch.zeros_like(routes, dtype=torch.float32)
    if navigation_mass is None:
        navigation_mass = torch.zeros_like(routes, dtype=torch.float32)
    if streaming_mass.shape != routes.shape or navigation_mass.shape != routes.shape:
        raise ValueError("mass tensors must match route table shape")

    while current > target_ratio:
        # Candidate downgrades with estimated damage. Lower damage first.  Rebuild
        # after every step so heads can move FULL->NAV->STREAM when the requested
        # ratio is below the all-navigation cost.
        candidates: List[Tuple[float, int, int, Route]] = []
        for l in range(n_layers):
            for h in range(n_heads):
                r = Route(int(routes[l, h].item()))
                if r == Route.FULL:
                    damage = float(max(0.0, 1.0 - float(navigation_mass[l, h].item())))
                    candidates.append((damage, l, h, Route.NAV))
                elif r == Route.NAV:
                    damage = float(max(0.0, 1.0 - float(streaming_mass[l, h].item())))
                    candidates.append((damage, l, h, Route.STREAM))
        if not candidates:
            raise ValueError(
                f"target_ratio={target_ratio} is below the minimum achievable ratio {current}"
            )
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, l, h, new_route = candidates[0]
        routes[l, h] = int(new_route)
        current = ratio(routes)

    meta = dict(table.metadata)
    meta["budget_projection"] = {
        "target_ratio": target_ratio,
        "initial_ratio": table.memory_ratio(cfg, seq_len),
        "final_ratio": ratio(routes),
    }
    return RouteTable(routes, meta)


def make_uniform_route_table(num_layers: int, num_heads: int, route: Route | str | int, metadata: Optional[Dict[str, Any]] = None) -> RouteTable:
    r = Route.parse(route)
    return RouteTable(torch.full((num_layers, num_heads), int(r), dtype=torch.long), metadata or {})
