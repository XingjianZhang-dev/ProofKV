from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

import torch


LegacyLayer = Tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class CompactPrefillInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    retained_indices: torch.Tensor


def kv_tensor_nbytes(x: Any) -> int:
    """Return bytes held by tensor-like KV cache structures."""

    if isinstance(x, torch.Tensor):
        return int(x.numel() * x.element_size())
    if isinstance(x, (tuple, list)):
        return sum(kv_tensor_nbytes(item) for item in x)
    if hasattr(x, "to_legacy_cache"):
        return kv_tensor_nbytes(x.to_legacy_cache())
    if hasattr(x, "key_cache") and hasattr(x, "value_cache"):
        return kv_tensor_nbytes(list(x.key_cache)) + kv_tensor_nbytes(list(x.value_cache))
    if hasattr(x, "layers"):
        total = 0
        for layer in x.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is not None:
                total += kv_tensor_nbytes(key)
            if value is not None:
                total += kv_tensor_nbytes(value)
        return total
    return 0


def past_key_values_nbytes(past_key_values: Any) -> int:
    return kv_tensor_nbytes(past_key_values)


def _validate_kv_tensors(key: torch.Tensor, value: torch.Tensor, layer_idx: int) -> LegacyLayer:
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError(f"Layer {layer_idx} must expose tensor key/value states")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"Layer {layer_idx} K/V tensors must have shape [batch, kv_heads, seq_len, head_dim]; "
            f"got {tuple(key.shape)} and {tuple(value.shape)}"
        )
    if key.shape != value.shape:
        raise ValueError(f"Layer {layer_idx} key/value shapes differ: {tuple(key.shape)} vs {tuple(value.shape)}")
    return key, value


def legacy_cache_layers(past_key_values: Any) -> Tuple[LegacyLayer, ...]:
    """Return validated legacy-style ``(key, value)`` cache layers.

    Supported inputs are legacy tuple/list caches, DynamicCache-like objects with
    ``to_legacy_cache()``, older ``key_cache``/``value_cache`` attributes, and
    Transformers 5 ``DynamicCache.layers`` entries exposing ``keys``/``values``.
    K/V tensors are expected to have shape ``[batch, kv_heads, seq_len, head_dim]``.
    """

    cache = past_key_values.to_legacy_cache() if hasattr(past_key_values, "to_legacy_cache") else past_key_values

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache = tuple(zip(cache.key_cache, cache.value_cache))
    elif hasattr(cache, "layers"):
        layers: List[LegacyLayer] = []
        for idx, layer in enumerate(cache.layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                raise ValueError(f"DynamicCache layer {idx} is not initialized with key/value tensors")
            layers.append(_validate_kv_tensors(key, value, idx))
        return tuple(layers)

    if not isinstance(cache, (tuple, list)):
        raise TypeError(f"Unsupported past_key_values type: {type(past_key_values).__name__}")

    layers = []
    for idx, layer in enumerate(cache):
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError(f"Cache layer {idx} must be a tuple/list containing key and value tensors")
        layers.append(_validate_kv_tensors(layer[0], layer[1], idx))
    return tuple(layers)


def _retained_index_tensor(retained_indices: Sequence[int] | torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    idx = torch.as_tensor(retained_indices, dtype=torch.long, device=device).flatten()
    if idx.numel() and (int(idx.min().item()) < 0 or int(idx.max().item()) >= seq_len):
        raise IndexError(f"retained_indices must be in [0, {seq_len}); got min={int(idx.min())}, max={int(idx.max())}")
    return torch.unique(idx, sorted=True)


def gather_legacy_past_key_values(
    past_key_values: Any,
    retained_indices: Sequence[int] | torch.Tensor,
    return_dynamic_cache: bool = True,
) -> Any:
    """Physically gather K/V tensors along sequence dimension 2.

    This is offline compact replay after Full prefill. If this Transformers build exposes
    ``DynamicCache.from_legacy_cache`` and ``return_dynamic_cache`` is true, the
    compact legacy tuple is converted back to DynamicCache; otherwise the legacy
    tuple is returned as an explicit compatibility fallback.
    """

    layers = legacy_cache_layers(past_key_values)
    compact: List[LegacyLayer] = []
    for key, value in layers:
        idx = _retained_index_tensor(retained_indices, key.shape[2], key.device)
        compact.append((key.index_select(2, idx).contiguous(), value.index_select(2, idx).contiguous()))
    compact_tuple: Tuple[LegacyLayer, ...] = tuple(compact)

    if return_dynamic_cache:
        try:
            from transformers.cache_utils import DynamicCache

            from_legacy = getattr(DynamicCache, "from_legacy_cache", None)
            if from_legacy is not None:
                return from_legacy(compact_tuple)
        except Exception:
            pass
    return compact_tuple


def gather_dynamic_cache_inplace(
    past_key_values: Any,
    retained_indices: Sequence[int] | torch.Tensor,
) -> Any:
    """Destructively compact a mutable DynamicCache one layer at a time.

    Unlike legacy gather followed by DynamicCache reconstruction, this path
    never keeps a second complete compact cache beside the Full cache.  Each
    Full layer is replaced as soon as its selected K/V tensors are materialized,
    so peak live storage is bounded by the Full cache plus one selected layer.
    The input cache is consumed and must not be reused as a Full-KV reference.
    """

    if hasattr(past_key_values, "layers"):
        layers = past_key_values.layers
        if not layers:
            raise ValueError("Cannot compact an empty DynamicCache")
        first_key = getattr(layers[0], "keys", None)
        if not isinstance(first_key, torch.Tensor):
            raise TypeError("DynamicCache layer 0 does not expose a key tensor")
        full_seq_len = int(first_key.shape[2])
        idx = _retained_index_tensor(
            retained_indices, full_seq_len, first_key.device
        )
        del first_key
        for layer_idx, layer in enumerate(layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            key, value = _validate_kv_tensors(key, value, layer_idx)
            if key.shape[2] != full_seq_len:
                raise ValueError(
                    f"DynamicCache layer {layer_idx} sequence length {key.shape[2]} "
                    f"differs from layer 0 length {full_seq_len}"
                )
            selected_key = key.index_select(2, idx).contiguous()
            selected_value = value.index_select(2, idx).contiguous()
            layer.keys = selected_key
            layer.values = selected_value
            del key, value, selected_key, selected_value
        return past_key_values

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache
        if not key_cache or len(key_cache) != len(value_cache):
            raise ValueError("Cannot compact empty or mismatched key/value cache lists")
        first_key, first_value = _validate_kv_tensors(key_cache[0], value_cache[0], 0)
        full_seq_len = int(first_key.shape[2])
        idx = _retained_index_tensor(
            retained_indices, full_seq_len, first_key.device
        )
        del first_key, first_value
        for layer_idx in range(len(key_cache)):
            key, value = _validate_kv_tensors(
                key_cache[layer_idx], value_cache[layer_idx], layer_idx
            )
            if key.shape[2] != full_seq_len:
                raise ValueError(
                    f"Cache layer {layer_idx} sequence length {key.shape[2]} "
                    f"differs from layer 0 length {full_seq_len}"
                )
            selected_key = key.index_select(2, idx).contiguous()
            selected_value = value.index_select(2, idx).contiguous()
            key_cache[layer_idx] = selected_key
            value_cache[layer_idx] = selected_value
            del key, value, selected_key, selected_value
        return past_key_values

    raise TypeError(
        "In-place gather requires a mutable DynamicCache with layers or key/value cache lists"
    )


def make_position_ids(start_pos: int, length: int, device: torch.device | str) -> torch.Tensor:
    return torch.arange(start_pos, start_pos + length, dtype=torch.long, device=device).unsqueeze(0)


def make_cache_position(start_pos: int, length: int, device: torch.device | str) -> torch.Tensor:
    return torch.arange(start_pos, start_pos + length, dtype=torch.long, device=device)


def make_compact_attention_mask(
    batch_size: int,
    compact_cache_len: int,
    input_len: int,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.ones((batch_size, compact_cache_len + input_len), dtype=torch.long, device=device)


def prepare_compact_prefill_inputs(
    input_ids: torch.Tensor,
    retained_indices: Sequence[int] | torch.Tensor,
) -> CompactPrefillInputs:
    """Gather retained prompt tokens for true compact prefill.

    ``position_ids`` preserve original prompt positions for RoPE, while
    ``cache_position`` is compact and gap-free so the physical KV cache has no
    hidden holes from dropped tokens.
    """

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq_len]; got {tuple(input_ids.shape)}")
    batch, seq_len = input_ids.shape
    idx = _retained_index_tensor(retained_indices, seq_len, input_ids.device)
    compact_ids = input_ids.index_select(1, idx).contiguous()
    attention_mask = torch.ones((batch, idx.numel()), dtype=torch.long, device=input_ids.device)
    position_ids = idx.unsqueeze(0).expand(batch, -1).contiguous()
    cache_position = torch.arange(idx.numel(), dtype=torch.long, device=input_ids.device)
    return CompactPrefillInputs(
        input_ids=compact_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        retained_indices=idx,
    )


def make_decode_cache_position(compact_cache_len: int, length: int, device: torch.device | str) -> torch.Tensor:
    """Return compact insertion positions for decoding after compact prefill."""

    return torch.arange(compact_cache_len, compact_cache_len + length, dtype=torch.long, device=device)


def estimate_full_cache_nbytes_like(compact_cache: Any, full_seq_len: int) -> int:
    """Estimate full-cache bytes from a compact cache's layer/head/dtype layout."""

    if full_seq_len < 0:
        raise ValueError("full_seq_len must be non-negative")
    layers = legacy_cache_layers(compact_cache)
    total = 0
    for key, value in layers:
        if key.shape[2] <= 0:
            raise ValueError("Cannot estimate full cache bytes from an empty compact cache")
        key_per_token = key.numel() * key.element_size() // key.shape[2]
        value_per_token = value.numel() * value.element_size() // value.shape[2]
        total += (key_per_token + value_per_token) * int(full_seq_len)
    return int(total)


def compact_prefill_cache_stats(full_seq_len: int, compact_cache: Any) -> dict[str, Any]:
    layers = legacy_cache_layers(compact_cache)
    first = layers[0][0]
    compact_bytes = past_key_values_nbytes(layers)
    full_bytes = estimate_full_cache_nbytes_like(layers, full_seq_len)
    return {
        "full_bytes_estimated": full_bytes,
        "compact_bytes": compact_bytes,
        "byte_ratio": compact_bytes / max(1, full_bytes),
        "layer_count": len(layers),
        "full_seq_len": int(full_seq_len),
        "compact_seq_len": int(first.shape[2]),
        "dtype": str(first.dtype),
        "device": str(first.device),
    }


def compact_cache_stats(full_cache: Any, compact_cache: Any) -> dict[str, Any]:
    full_layers = legacy_cache_layers(full_cache)
    compact_layers = legacy_cache_layers(compact_cache)
    if len(full_layers) != len(compact_layers):
        raise ValueError(f"Layer count mismatch: full={len(full_layers)} compact={len(compact_layers)}")
    first_full = full_layers[0][0]
    first_compact = compact_layers[0][0]
    full_bytes = past_key_values_nbytes(full_layers)
    compact_bytes = past_key_values_nbytes(compact_layers)
    return {
        "full_bytes": full_bytes,
        "compact_bytes": compact_bytes,
        "byte_ratio": compact_bytes / max(1, full_bytes),
        "layer_count": len(full_layers),
        "full_seq_len": int(first_full.shape[2]),
        "compact_seq_len": int(first_compact.shape[2]),
        "dtype": str(first_full.dtype),
        "device": str(first_full.device),
    }
