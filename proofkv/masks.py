from __future__ import annotations

from typing import Iterable, Optional

import torch


def _device(device: Optional[torch.device | str]) -> torch.device:
    return torch.device("cpu") if device is None else torch.device(device)


def normalize_indices(
    indices: Iterable[int] | torch.Tensor | None,
    seq_len: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return sorted unique valid indices in ``[0, seq_len)``.

    This helper deliberately drops out-of-range values rather than raising.  In
    generation, guidepost tables can be reused for shorter prompts; dropping is
    safer and keeps masks well-defined.
    """

    dev = _device(device)
    if seq_len <= 0:
        return torch.empty(0, dtype=torch.long, device=dev)
    if indices is None:
        return torch.empty(0, dtype=torch.long, device=dev)
    if isinstance(indices, torch.Tensor):
        idx = indices.to(device=dev, dtype=torch.long).flatten()
    else:
        idx = torch.tensor(list(indices), dtype=torch.long, device=dev)
    if idx.numel() == 0:
        return idx
    idx = idx[(idx >= 0) & (idx < seq_len)]
    if idx.numel() == 0:
        return idx
    return torch.unique(idx, sorted=True)


def query_absolute_positions(q_len: int, kv_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Absolute positions of the query tokens inside a causal self-attention call.

    Hugging Face decoder attention receives keys of length ``past + q_len``.  The
    query tokens correspond to the last ``q_len`` absolute positions.
    """

    if q_len < 1:
        raise ValueError("q_len must be positive")
    if kv_len < q_len:
        raise ValueError(f"kv_len ({kv_len}) must be >= q_len ({q_len})")
    dev = _device(device)
    return torch.arange(kv_len - q_len, kv_len, dtype=torch.long, device=dev)


def full_keep_mask(q_len: int, kv_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Causal keep mask with shape ``[q_len, kv_len]``."""

    dev = _device(device)
    q_abs = query_absolute_positions(q_len, kv_len, dev).unsqueeze(-1)
    k_abs = torch.arange(kv_len, dtype=torch.long, device=dev).unsqueeze(0)
    return k_abs <= q_abs


def streaming_keep_mask(
    q_len: int,
    kv_len: int,
    sink_size: int,
    recent_size: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Keep attention sinks and a query-relative recent window.

    The current token is always visible because ``recent_size >= 1``.  Sink tokens
    are causal: early queries cannot attend to future sink positions.
    """

    if sink_size < 0:
        raise ValueError("sink_size must be non-negative")
    if recent_size < 1:
        raise ValueError("recent_size must be at least 1")
    dev = _device(device)
    q_abs = query_absolute_positions(q_len, kv_len, dev).unsqueeze(-1)
    k_abs = torch.arange(kv_len, dtype=torch.long, device=dev).unsqueeze(0)
    causal = k_abs <= q_abs
    sink = k_abs < min(sink_size, kv_len)
    recent_start = torch.clamp(q_abs - recent_size + 1, min=0)
    recent = (k_abs >= recent_start) & causal
    return causal & (sink | recent)


def navigation_keep_mask(
    q_len: int,
    kv_len: int,
    sink_size: int,
    recent_size: int,
    guidepost_indices: Iterable[int] | torch.Tensor | None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Keep streaming memory plus sparse global guidepost indices."""

    dev = _device(device)
    keep = streaming_keep_mask(q_len, kv_len, sink_size, recent_size, dev)
    guideposts = normalize_indices(guidepost_indices, kv_len, dev)
    if guideposts.numel() == 0:
        return keep
    q_abs = query_absolute_positions(q_len, kv_len, dev).unsqueeze(-1)
    gp = guideposts.unsqueeze(0)
    # A guidepost is visible only once it is in the causal past.
    visible = gp <= q_abs
    keep.scatter_(1, gp.expand(q_len, -1), visible)
    return keep


def additive_mask_from_keep(keep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a boolean keep mask to an additive attention mask."""

    if keep.dtype is not torch.bool:
        raise TypeError("keep must be a boolean tensor")
    # Use a large finite negative number instead of -inf to avoid NaNs on some
    # fused paths and lower-precision softmax kernels.
    neg = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    return torch.where(keep, torch.zeros((), dtype=dtype, device=keep.device), torch.full((), neg, dtype=dtype, device=keep.device))


def count_kept(keep: torch.Tensor) -> torch.Tensor:
    """Count visible positions per query row."""

    if keep.dtype is not torch.bool:
        raise TypeError("keep must be bool")
    return keep.long().sum(dim=-1)
