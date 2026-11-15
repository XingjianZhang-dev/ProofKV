from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Route(IntEnum):
    """Memory routes used by CompassKV.

    FULL: every causal key/value position is available.
    NAV: sink + recent + sparse global guideposts.
    STREAM: sink + recent only.
    """

    FULL = 0
    NAV = 1
    STREAM = 2

    @classmethod
    def parse(cls, value: str | int | "Route") -> "Route":
        if isinstance(value, Route):
            return value
        if isinstance(value, int):
            return Route(value)
        normalized = value.strip().lower()
        aliases = {
            "full": cls.FULL,
            "f": cls.FULL,
            "navigation": cls.NAV,
            "nav": cls.NAV,
            "guidepost": cls.NAV,
            "waypoint": cls.NAV,
            "stream": cls.STREAM,
            "streaming": cls.STREAM,
            "s": cls.STREAM,
        }
        if normalized not in aliases:
            raise ValueError(f"Unknown route {value!r}. Expected one of {sorted(aliases)}")
        return aliases[normalized]

    @property
    def label(self) -> str:
        return {Route.FULL: "full", Route.NAV: "navigation", Route.STREAM: "streaming"}[self]


@dataclass
class CompassConfig:
    """Configuration for CompassKV reference experiments.

    The defaults are conservative enough for a first A100 run.  All fields are
    serializable so that route tables and run outputs are self-describing.
    """

    sink_size: int = 4
    recent_size: int = 512
    guidepost_count: int = 256
    block_size: int = 128
    candidates_per_block: int = 1
    selector: str = "blockmax"  # blockmax | uniform | random | keynorm | extended_recent
    route_scope: str = "query_head"  # query_head | kv_head
    default_route: str = "navigation"
    apply_during_prefill: bool = False
    decode_q_len_threshold: int = 1
    route_table_path: Optional[str] = None
    target_budget_ratio: Optional[float] = None
    calibration_threshold: float = 0.985
    navigation_threshold: float = 0.985
    streaming_threshold: float = 0.985
    min_full_heads_per_layer: int = 0
    dtype: str = "bfloat16"
    seed: int = 0
    notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        if self.recent_size < 1:
            raise ValueError("recent_size must be at least 1 so the current token is visible")
        if self.guidepost_count < 0:
            raise ValueError("guidepost_count must be non-negative")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.candidates_per_block < 1:
            raise ValueError("candidates_per_block must be positive")
        if self.route_scope not in {"query_head", "kv_head"}:
            raise ValueError("route_scope must be 'query_head' or 'kv_head'")
        if self.selector not in {"blockmax", "uniform", "random", "keynorm", "extended_recent"}:
            raise ValueError(
                "selector must be one of: blockmax, uniform, random, keynorm, extended_recent"
            )
        # Validate alias early.
        Route.parse(self.default_route)

    @property
    def default_route_enum(self) -> Route:
        return Route.parse(self.default_route)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CompassConfig":
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CompassConfig":
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return cls.from_dict(payload)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    def route_cost(self, route: Route | int | str, seq_len: int) -> int:
        """Theoretical per-head KV length consumed by a route at sequence length seq_len."""
        route = Route.parse(route)
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if route == Route.FULL:
            return seq_len
        stream = min(seq_len, self.sink_size + self.recent_size)
        if route == Route.STREAM:
            return stream
        return min(seq_len, self.sink_size + self.recent_size + self.guidepost_count)

    def memory_ratio_for_counts(self, counts: Dict[str, int], seq_len: int) -> float:
        total_heads = sum(counts.values())
        if total_heads == 0 or seq_len == 0:
            return 0.0
        total = 0
        for name, n in counts.items():
            total += n * self.route_cost(name, seq_len)
        return total / (total_heads * seq_len)
