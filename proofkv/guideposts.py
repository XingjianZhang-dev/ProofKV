from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch

from .masks import normalize_indices


@dataclass(frozen=True)
class GuidepostSelection:
    indices: torch.Tensor
    scores: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_scores: torch.Tensor


def protected_mask(seq_len: int, sink_size: int, recent_size: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Mask positions that are already covered by streaming memory."""

    dev = torch.device("cpu") if device is None else torch.device(device)
    mask = torch.zeros(seq_len, dtype=torch.bool, device=dev)
    if seq_len <= 0:
        return mask
    if sink_size > 0:
        mask[: min(sink_size, seq_len)] = True
    recent_start = max(0, seq_len - max(1, recent_size))
    mask[recent_start:] = True
    return mask


def block_ranges(
    seq_len: int,
    sink_size: int,
    recent_size: int,
    block_size: int,
) -> List[Tuple[int, int]]:
    """Return non-overlapping guidepost candidate blocks excluding sink/recent."""

    if block_size < 1:
        raise ValueError("block_size must be positive")
    start = min(max(0, sink_size), seq_len)
    end = max(start, seq_len - max(1, recent_size))
    ranges: List[Tuple[int, int]] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + block_size, end)
        if nxt > cursor:
            ranges.append((cursor, nxt))
        cursor = nxt
    return ranges


def select_uniform_guideposts(
    seq_len: int,
    num_guideposts: int,
    sink_size: int,
    recent_size: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Select evenly spaced global guideposts outside sink/recent memory."""

    dev = torch.device("cpu") if device is None else torch.device(device)
    if num_guideposts <= 0 or seq_len <= 0:
        return torch.empty(0, dtype=torch.long, device=dev)
    start = min(max(0, sink_size), seq_len)
    end = max(start, seq_len - max(1, recent_size))
    if end <= start:
        return torch.empty(0, dtype=torch.long, device=dev)
    n = min(num_guideposts, end - start)
    # Use cell centers to avoid repeatedly selecting boundaries.
    positions = torch.linspace(start, end - 1, n, device=dev)
    return torch.unique(positions.round().long().clamp(0, seq_len - 1), sorted=True)


def _topk_in_block(scores: torch.Tensor, start: int, end: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    local = scores[start:end]
    if local.numel() == 0:
        dev = scores.device
        return torch.empty(0, dtype=torch.long, device=dev), torch.empty(0, dtype=scores.dtype, device=dev)
    k = min(k, local.numel())
    vals, rel = torch.topk(local, k=k, largest=True, sorted=False)
    return rel + start, vals


def select_guideposts_from_scores(
    scores: torch.Tensor,
    num_guideposts: int,
    sink_size: int,
    recent_size: int,
    block_size: int,
    candidates_per_block: int = 1,
    fallback_uniform: bool = True,
) -> GuidepostSelection:
    """Block-wise candidate pool followed by global top-k selection.

    Args:
        scores: 1D tensor with one saliency score per KV position.  Larger is
            better. NaN values are treated as ``-inf``.
        num_guideposts: requested number of guideposts outside sink/recent.
        sink_size, recent_size: positions already covered by streaming memory.
        block_size: candidate-pool block size.
        candidates_per_block: number of local candidates from each block.
        fallback_uniform: fill missing slots with uniform positions when the
            candidate pool is smaller than ``num_guideposts``.

    Returns:
        ``GuidepostSelection`` with sorted unique selected indices.
    """

    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {tuple(scores.shape)}")
    seq_len = int(scores.numel())
    dev = scores.device
    dtype = scores.dtype
    if num_guideposts <= 0 or seq_len == 0:
        empty_i = torch.empty(0, dtype=torch.long, device=dev)
        empty_s = torch.empty(0, dtype=dtype, device=dev)
        return GuidepostSelection(empty_i, empty_s, empty_i, empty_s)
    clean = torch.nan_to_num(scores.float(), nan=-float("inf"), posinf=float("inf"), neginf=-float("inf"))
    ranges = block_ranges(seq_len, sink_size, recent_size, block_size)
    cand_idx: List[torch.Tensor] = []
    cand_scores: List[torch.Tensor] = []
    for start, end in ranges:
        idx, vals = _topk_in_block(clean, start, end, candidates_per_block)
        if idx.numel() > 0:
            cand_idx.append(idx)
            cand_scores.append(vals)
    if cand_idx:
        candidates = torch.cat(cand_idx).long()
        candidate_scores = torch.cat(cand_scores).to(dtype=torch.float32)
    else:
        candidates = torch.empty(0, dtype=torch.long, device=dev)
        candidate_scores = torch.empty(0, dtype=torch.float32, device=dev)

    if candidates.numel() == 0:
        selected = torch.empty(0, dtype=torch.long, device=dev)
        selected_scores = torch.empty(0, dtype=torch.float32, device=dev)
    else:
        k = min(num_guideposts, candidates.numel())
        vals, order = torch.topk(candidate_scores, k=k, largest=True, sorted=False)
        selected = candidates[order]
        selected_scores = vals

    if fallback_uniform and selected.numel() < num_guideposts:
        extra = select_uniform_guideposts(seq_len, num_guideposts, sink_size, recent_size, dev)
        merged = torch.cat([selected, extra]) if selected.numel() else extra
        merged = normalize_indices(merged, seq_len, dev)
        # Drop protected positions and keep only the requested count.
        prot = protected_mask(seq_len, sink_size, recent_size, dev)
        merged = merged[~prot[merged]]
        if merged.numel() > num_guideposts:
            # Sort by score where possible, then return sorted positions.
            vals = clean[merged]
            _, top = torch.topk(vals, k=num_guideposts, largest=True, sorted=False)
            merged = merged[top]
        selected = merged
        selected_scores = clean[selected] if selected.numel() else torch.empty(0, dtype=torch.float32, device=dev)

    selected = normalize_indices(selected, seq_len, dev)
    if selected.numel() > num_guideposts:
        vals = clean[selected]
        _, top = torch.topk(vals, k=num_guideposts, largest=True, sorted=False)
        selected = normalize_indices(selected[top], seq_len, dev)
    selected_scores = clean[selected] if selected.numel() else torch.empty(0, dtype=torch.float32, device=dev)
    return GuidepostSelection(selected, selected_scores.to(dtype), candidates, candidate_scores.to(dtype))


def select_random_guideposts(
    seq_len: int,
    num_guideposts: int,
    sink_size: int,
    recent_size: int,
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Random guideposts for ablation, deterministic under ``seed``."""

    dev = torch.device("cpu") if device is None else torch.device(device)
    if seq_len <= 0 or num_guideposts <= 0:
        return torch.empty(0, dtype=torch.long, device=dev)
    prot = protected_mask(seq_len, sink_size, recent_size, dev)
    candidates = torch.arange(seq_len, dtype=torch.long, device=dev)[~prot]
    if candidates.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    perm = torch.randperm(candidates.numel(), generator=g, device=dev)
    selected = candidates[perm[: min(num_guideposts, candidates.numel())]]
    return torch.unique(selected, sorted=True)


def guideposts_from_key_norm(
    key_states: torch.Tensor,
    num_guideposts: int,
    sink_size: int,
    recent_size: int,
    block_size: int,
    candidates_per_block: int = 1,
) -> torch.Tensor:
    """Query-agnostic selector based on key-vector L2 norm.

    ``key_states`` may have shape ``[K, D]`` or ``[..., K, D]``.  All leading
    dimensions are averaged, yielding one saliency score per position.
    """

    if key_states.ndim < 2:
        raise ValueError("key_states must have at least [K, D] dimensions")
    scores = key_states.float().pow(2).sum(dim=-1)
    while scores.ndim > 1:
        scores = scores.mean(dim=0)
    return select_guideposts_from_scores(
        scores,
        num_guideposts=num_guideposts,
        sink_size=sink_size,
        recent_size=recent_size,
        block_size=block_size,
        candidates_per_block=candidates_per_block,
    ).indices
