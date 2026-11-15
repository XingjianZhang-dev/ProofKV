import pytest
import torch

from proofkv.physical_kv import (
    compact_cache_stats,
    gather_legacy_past_key_values,
    kv_tensor_nbytes,
    legacy_cache_layers,
    make_cache_position,
    make_compact_attention_mask,
    make_decode_cache_position,
    make_position_ids,
    past_key_values_nbytes,
    prepare_compact_prefill_inputs,
    compact_prefill_cache_stats,
)


def _cache(layers=3, batch=1, kv_heads=2, seq_len=10, head_dim=4, dtype=torch.float32):
    out = []
    for layer in range(layers):
        key = torch.arange(batch * kv_heads * seq_len * head_dim, dtype=dtype).reshape(batch, kv_heads, seq_len, head_dim)
        value = (key + 1000 + layer).clone()
        out.append((key.clone(), value.clone()))
    return tuple(out)


def test_gather_preserves_layer_count_dtype_and_device():
    cache = _cache(dtype=torch.bfloat16)
    compact = gather_legacy_past_key_values(cache, [1, 3, 7], return_dynamic_cache=False)
    layers = legacy_cache_layers(compact)
    assert len(layers) == len(cache)
    assert layers[0][0].dtype == torch.bfloat16
    assert layers[0][0].device == cache[0][0].device


def test_compact_sequence_length_equals_retained_count():
    compact = gather_legacy_past_key_values(_cache(seq_len=12), [0, 4, 9, 11], return_dynamic_cache=False)
    assert legacy_cache_layers(compact)[0][0].shape[2] == 4


def test_byte_ratio_matches_retained_over_full():
    full = _cache(seq_len=20)
    compact = gather_legacy_past_key_values(full, [0, 3, 7, 12, 19], return_dynamic_cache=False)
    stats = compact_cache_stats(full, compact)
    assert stats["byte_ratio"] == pytest.approx(5 / 20)
    assert kv_tensor_nbytes(compact) == past_key_values_nbytes(compact)


def test_position_helpers_use_original_prompt_positions():
    assert make_position_ids(99, 2, "cpu").tolist() == [[99, 100]]
    assert make_cache_position(99, 2, "cpu").tolist() == [99, 100]
    assert make_compact_attention_mask(2, 5, 3, "cpu").shape == (2, 8)
    assert make_decode_cache_position(5, 3, "cpu").tolist() == [5, 6, 7]


def test_prepare_compact_prefill_preserves_rope_positions_but_compacts_cache_positions():
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    prepared = prepare_compact_prefill_inputs(input_ids, [0, 4, 2, 4])
    assert prepared.input_ids.tolist() == [[10, 12, 14]]
    assert prepared.attention_mask.tolist() == [[1, 1, 1]]
    assert prepared.position_ids.tolist() == [[0, 2, 4]]
    assert prepared.cache_position.tolist() == [0, 1, 2]
    assert prepared.retained_indices.tolist() == [0, 2, 4]


def test_compact_prefill_cache_stats_estimates_full_byte_ratio_without_full_prefill_cache():
    compact = _cache(layers=2, kv_heads=2, seq_len=3, head_dim=4, dtype=torch.bfloat16)
    stats = compact_prefill_cache_stats(full_seq_len=12, compact_cache=compact)
    assert stats["compact_seq_len"] == 3
    assert stats["full_seq_len"] == 12
    assert stats["byte_ratio"] == pytest.approx(3 / 12)


def test_invalid_kv_shape_raises_clear_error():
    bad = ((torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)),)
    with pytest.raises(ValueError, match=r"\[batch, kv_heads, seq_len, head_dim\]"):
        legacy_cache_layers(bad)


def test_original_cache_tensors_are_not_mutated():
    full = _cache(seq_len=8)
    before = full[0][0].clone()
    compact = gather_legacy_past_key_values(full, [2, 5], return_dynamic_cache=False)
    compact[0][0].fill_(0)
    assert torch.equal(full[0][0], before)


def test_synthetic_gqa_style_cache_gathers_kv_heads():
    full = _cache(layers=2, kv_heads=2, seq_len=9, head_dim=8)
    compact = gather_legacy_past_key_values(full, [1, 2, 3], return_dynamic_cache=False)
    key, value = legacy_cache_layers(compact)[0]
    assert key.shape == (1, 2, 3, 8)
    assert value.shape == (1, 2, 3, 8)


def test_dynamic_cache_from_legacy_path_if_available():
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        pytest.skip("transformers DynamicCache unavailable")
    if not hasattr(DynamicCache, "from_legacy_cache"):
        pytest.skip("DynamicCache.from_legacy_cache unavailable in this Transformers build")
    compact = gather_legacy_past_key_values(_cache(seq_len=8), [1, 4], return_dynamic_cache=True)
    assert isinstance(compact, DynamicCache)
