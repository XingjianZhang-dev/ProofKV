from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import CompassConfig, Route
from .guideposts import (
    select_guideposts_from_scores,
    select_random_guideposts,
    select_uniform_guideposts,
)
from .masks import full_keep_mask, navigation_keep_mask, streaming_keep_mask


@dataclass
class RoutedAttentionInfo:
    """Diagnostics returned by the reference attention implementation."""

    kept_counts: torch.Tensor  # [batch, heads, q_len]
    routes: torch.Tensor  # [heads]
    guideposts: Dict[Tuple[int, int], torch.Tensor]  # (batch, head) -> indices
    attention_probs: Optional[torch.Tensor] = None


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to query-head count, matching Hugging Face LLaMA semantics."""

    if n_rep < 1:
        raise ValueError("n_rep must be positive")
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _as_query_head_routes(routes: Sequence[int] | torch.Tensor, num_heads: int, device: torch.device) -> torch.Tensor:
    if isinstance(routes, torch.Tensor):
        out = routes.to(device=device, dtype=torch.long).flatten()
    else:
        out = torch.tensor([int(Route.parse(r)) for r in routes], dtype=torch.long, device=device)
    if out.numel() != num_heads:
        raise ValueError(f"routes length {out.numel()} does not match num_heads={num_heads}")
    return out


def _external_additive_mask(attention_mask: Optional[torch.Tensor], b: int, q_len: int, kv_len: int) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    # HF decoder masks are usually [B, 1, Q, K]. Some tests pass [B, Q, K].
    mask = attention_mask
    if mask.ndim == 4:
        mask = mask[b, 0, :, :kv_len]
    elif mask.ndim == 3:
        mask = mask[b, :, :kv_len]
    elif mask.ndim == 2:
        mask = mask[:, :kv_len]
    else:
        raise ValueError(f"Unsupported attention_mask shape {tuple(mask.shape)}")
    if mask.shape[-2:] != (q_len, kv_len):
        # During some generation calls HF can pass a longer query axis; keep the
        # last q_len rows, which correspond to the current query states.
        if mask.shape[-1] >= kv_len and mask.shape[-2] >= q_len:
            mask = mask[-q_len:, -kv_len:]
        else:
            raise ValueError(
                f"attention_mask shape {tuple(mask.shape)} cannot align with {(q_len, kv_len)}"
            )
    return mask


def _safe_masked_softmax(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    if scores.shape != keep.shape:
        raise ValueError(f"scores shape {scores.shape} != keep shape {keep.shape}")
    # Work in float32 for stability. Every route mask includes at least the
    # current key, so every row must have a valid element.
    if not torch.all(keep.any(dim=-1)):
        bad = torch.nonzero(~keep.any(dim=-1), as_tuple=False).flatten().tolist()
        raise RuntimeError(f"At least one query row has no visible key positions: {bad}")
    masked = scores.float().masked_fill(~keep, torch.finfo(torch.float32).min)
    return torch.softmax(masked, dim=-1).to(dtype=scores.dtype)


def _sparse_attention_from_keep(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: torch.Tensor,
    scale: float,
    external: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Attention over only the positions marked by ``keep``.

    This path is used when guidepost spans are supplied by a prefill selector:
    it avoids decode-time QK over keys that the route has dropped.  The full
    probability tensor is reconstructed only when tests request attentions.
    """

    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q/k/v must be [T, D]")
    if keep.shape != (q.shape[0], k.shape[0]):
        raise ValueError(f"keep shape {tuple(keep.shape)} does not match {(q.shape[0], k.shape[0])}")
    if not torch.all(keep.any(dim=-1)):
        bad = torch.nonzero(~keep.any(dim=-1), as_tuple=False).flatten().tolist()
        raise RuntimeError(f"At least one query row has no visible key positions: {bad}")

    outputs: List[torch.Tensor] = []
    probs_out: List[torch.Tensor] = []
    for qi in range(q.shape[0]):
        idx = torch.nonzero(keep[qi], as_tuple=False).flatten()
        kk = k.index_select(0, idx)
        vv = v.index_select(0, idx)
        scores = torch.matmul(q[qi], kk.transpose(0, 1)) * scale
        if external is not None:
            scores = scores + external[qi].index_select(0, idx).to(dtype=scores.dtype, device=scores.device)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        outputs.append(torch.matmul(probs, vv))
        if output_attentions:
            full = torch.zeros(k.shape[0], dtype=probs.dtype, device=probs.device)
            full.scatter_(0, idx, probs)
            probs_out.append(full)
    attn = torch.stack(probs_out, dim=0) if output_attentions else None
    return torch.stack(outputs, dim=0), attn


def _guideposts_for_head(
    selector: str,
    logits: torch.Tensor,
    key_for_head: torch.Tensor,
    cfg: CompassConfig,
    batch_idx: int,
    head_idx: int,
    kv_len: int,
) -> torch.Tensor:
    """Select guideposts for one batch/head from query-key evidence."""

    device = logits.device
    if cfg.guidepost_count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if selector == "uniform":
        return select_uniform_guideposts(kv_len, cfg.guidepost_count, cfg.sink_size, cfg.recent_size, device)
    if selector == "random":
        seed = int(cfg.seed + 1009 * batch_idx + 9176 * head_idx + 13 * kv_len)
        return select_random_guideposts(kv_len, cfg.guidepost_count, cfg.sink_size, cfg.recent_size, seed, device)
    if selector == "keynorm":
        scores = key_for_head.float().pow(2).sum(dim=-1)
        return select_guideposts_from_scores(
            scores,
            cfg.guidepost_count,
            cfg.sink_size,
            cfg.recent_size,
            cfg.block_size,
            cfg.candidates_per_block,
        ).indices
    if selector == "extended_recent":
        # No global anchors: NAV degenerates to a larger recent window of the
        # same theoretical size. This is an ablation, not the main method.
        start = max(0, kv_len - (cfg.recent_size + cfg.guidepost_count))
        end = max(start, kv_len - cfg.recent_size)
        return torch.arange(start, end, dtype=torch.long, device=device)
    if selector != "blockmax":
        raise ValueError(f"Unknown selector {selector!r}")
    # Average over query rows. For decode q_len=1 this is simply the current
    # query's QK score. Causal future positions are filtered by masks later.
    scores = logits.float().mean(dim=0)
    return select_guideposts_from_scores(
        scores,
        cfg.guidepost_count,
        cfg.sink_size,
        cfg.recent_size,
        cfg.block_size,
        cfg.candidates_per_block,
    ).indices


def full_attention_reference(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reference full causal attention.

    Args use repeated query-head KV tensors: all tensors must be [B, H, T, D]
    except that query has q_len and key/value have kv_len.
    """

    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("query/key/value must be [B, H, T, D]")
    bsz, num_heads, q_len, head_dim = query_states.shape
    if key_states.shape[:2] != (bsz, num_heads) or value_states.shape[:2] != (bsz, num_heads):
        raise ValueError("key/value batch and head dimensions must match query")
    kv_len = key_states.shape[-2]
    scale = 1.0 / math.sqrt(head_dim)
    outputs: List[torch.Tensor] = []
    probs_out: List[torch.Tensor] = []
    base_keep = full_keep_mask(q_len, kv_len, query_states.device)
    for b in range(bsz):
        external = _external_additive_mask(attention_mask, b, q_len, kv_len)
        head_outputs: List[torch.Tensor] = []
        head_probs: List[torch.Tensor] = []
        for h in range(num_heads):
            scores = torch.matmul(query_states[b, h], key_states[b, h].transpose(-1, -2)) * scale
            if external is not None:
                scores = scores + external.to(dtype=scores.dtype, device=scores.device)
            probs = _safe_masked_softmax(scores, base_keep)
            head_outputs.append(torch.matmul(probs, value_states[b, h]))
            if output_attentions:
                head_probs.append(probs)
        outputs.append(torch.stack(head_outputs, dim=0))
        if output_attentions:
            probs_out.append(torch.stack(head_probs, dim=0))
    out = torch.stack(outputs, dim=0)
    attn = torch.stack(probs_out, dim=0) if output_attentions else None
    return out, attn


def routed_attention_reference(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    routes: Sequence[int] | torch.Tensor,
    cfg: CompassConfig,
    attention_mask: Optional[torch.Tensor] = None,
    guidepost_indices: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, RoutedAttentionInfo]:
    """Head-level routed attention reference implementation.

    ``key_states`` and ``value_states`` may use GQA KV heads.  They are repeated
    internally to query-head count.  The implementation is intentionally simple
    and auditable; it is not a throughput kernel.
    """

    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("query/key/value must be [B, H, T, D]")
    bsz, num_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    if num_heads % kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by kv_heads={kv_heads}")
    if value_states.shape != key_states.shape:
        raise ValueError("key_states and value_states must have the same shape")
    kv_len = key_states.shape[-2]
    n_rep = num_heads // kv_heads
    key_rep = repeat_kv(key_states, n_rep)
    value_rep = repeat_kv(value_states, n_rep)
    route_tensor = _as_query_head_routes(routes, num_heads, query_states.device)
    scale = 1.0 / math.sqrt(head_dim)

    outputs: List[torch.Tensor] = []
    probs_out: List[torch.Tensor] = []
    kept = torch.zeros((bsz, num_heads, q_len), dtype=torch.long, device=query_states.device)
    gp_map: Dict[Tuple[int, int], torch.Tensor] = {}
    supplied_gp = guidepost_indices or {}

    full_mask = full_keep_mask(q_len, kv_len, query_states.device)
    stream_mask = streaming_keep_mask(q_len, kv_len, cfg.sink_size, cfg.recent_size, query_states.device)

    for b in range(bsz):
        external = _external_additive_mask(attention_mask, b, q_len, kv_len)
        head_outputs: List[torch.Tensor] = []
        head_probs: List[torch.Tensor] = []
        for h in range(num_heads):
            route = Route(int(route_tensor[h].item()))
            q = query_states[b, h]
            k = key_rep[b, h]
            v = value_rep[b, h]
            if route == Route.FULL:
                keep = full_mask
                logits = torch.matmul(q, k.transpose(-1, -2)) * scale
                if external is not None:
                    logits = logits + external.to(dtype=logits.dtype, device=logits.device)
                probs = _safe_masked_softmax(logits, keep)
                head_outputs.append(torch.matmul(probs, v))
            elif route == Route.STREAM:
                keep = stream_mask
                out, probs = _sparse_attention_from_keep(q, k, v, keep, scale, external, output_attentions)
                head_outputs.append(out)
            else:
                gp = supplied_gp.get((b, h))
                if gp is None:
                    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
                    if external is not None:
                        logits = logits + external.to(dtype=logits.dtype, device=logits.device)
                    gp = _guideposts_for_head(cfg.selector, logits, k, cfg, b, h, kv_len)
                    keep = navigation_keep_mask(q_len, kv_len, cfg.sink_size, cfg.recent_size, gp, query_states.device)
                    probs = _safe_masked_softmax(logits, keep)
                    head_outputs.append(torch.matmul(probs, v))
                else:
                    keep = navigation_keep_mask(q_len, kv_len, cfg.sink_size, cfg.recent_size, gp, query_states.device)
                    out, probs = _sparse_attention_from_keep(q, k, v, keep, scale, external, output_attentions)
                    head_outputs.append(out)
                gp_map[(b, h)] = gp
            kept[b, h] = keep.long().sum(dim=-1)
            if output_attentions:
                head_probs.append(probs)
        outputs.append(torch.stack(head_outputs, dim=0))
        if output_attentions:
            probs_out.append(torch.stack(head_probs, dim=0))

    out = torch.stack(outputs, dim=0)
    info = RoutedAttentionInfo(
        kept_counts=kept,
        routes=route_tensor.detach().clone(),
        guideposts=gp_map,
        attention_probs=torch.stack(probs_out, dim=0) if output_attentions else None,
    )
    return out, info
