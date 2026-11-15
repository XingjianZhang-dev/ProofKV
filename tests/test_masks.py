import torch

from proofkv.masks import full_keep_mask, navigation_keep_mask, normalize_indices, streaming_keep_mask


def test_full_keep_mask_decode_and_prefill():
    assert full_keep_mask(1, 5).tolist() == [[True, True, True, True, True]]
    assert full_keep_mask(3, 5).tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]


def test_streaming_mask_is_causal_and_includes_current():
    mask = streaming_keep_mask(q_len=3, kv_len=6, sink_size=2, recent_size=2)
    assert mask.tolist() == [
        [True, True, True, True, False, False],
        [True, True, False, True, True, False],
        [True, True, False, False, True, True],
    ]
    assert torch.all(mask.any(dim=-1))


def test_navigation_mask_adds_visible_guideposts_only():
    mask = navigation_keep_mask(q_len=2, kv_len=6, sink_size=1, recent_size=1, guidepost_indices=[2, 5])
    assert mask.tolist() == [
        [True, False, True, False, True, False],
        [True, False, True, False, False, True],
    ]


def test_normalize_indices_sorted_unique_valid():
    out = normalize_indices([5, 2, 2, -1, 10], seq_len=6)
    assert out.tolist() == [2, 5]
