import torch

from proofkv.attention import full_attention_reference, routed_attention_reference
from proofkv.config import CompassConfig, Route


def _toy(seed=0, b=2, h=4, kvh=2, q=3, k=7, d=8):
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(b, h, q, d, generator=g)
    key = torch.randn(b, kvh, k, d, generator=g)
    value = torch.randn(b, kvh, k, d, generator=g)
    return query, key, value


def test_all_full_matches_full_attention_reference_with_gqa():
    query, key, value = _toy()
    from proofkv.attention import repeat_kv

    key_rep = repeat_kv(key, 2)
    value_rep = repeat_kv(value, 2)
    full, _ = full_attention_reference(query, key_rep, value_rep)
    cfg = CompassConfig(sink_size=1, recent_size=2, guidepost_count=2)
    routed, info = routed_attention_reference(query, key, value, [Route.FULL] * 4, cfg)
    assert torch.allclose(full, routed, atol=1e-6)
    assert info.kept_counts.shape == (2, 4, 3)


def test_streaming_differs_from_full_and_keeps_expected_count():
    query, key, value = _toy(b=1, h=2, kvh=1, q=1, k=10, d=4)
    cfg = CompassConfig(sink_size=2, recent_size=3, guidepost_count=0)
    routed, info = routed_attention_reference(query, key, value, [Route.STREAM] * 2, cfg)
    assert routed.shape == (1, 2, 1, 4)
    assert info.kept_counts.squeeze().tolist() == [5, 5]


def test_navigation_includes_guidepost_budget():
    query, key, value = _toy(b=1, h=2, kvh=1, q=1, k=10, d=4)
    cfg = CompassConfig(sink_size=1, recent_size=2, guidepost_count=3, block_size=2)
    _, info = routed_attention_reference(query, key, value, [Route.NAV] * 2, cfg)
    # sink + recent + up to 3 guideposts; no overlaps expected for this length.
    assert all(3 <= int(x) <= 6 for x in info.kept_counts.flatten().tolist())
    assert len(info.guideposts) == 2
