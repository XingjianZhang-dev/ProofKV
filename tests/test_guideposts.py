import torch

from proofkv.guideposts import block_ranges, select_guideposts_from_scores, select_uniform_guideposts


def test_block_ranges_exclude_sink_and_recent():
    assert block_ranges(seq_len=20, sink_size=2, recent_size=4, block_size=5) == [(2, 7), (7, 12), (12, 16)]


def test_select_guideposts_from_scores_uses_block_candidates():
    scores = torch.arange(20).float()
    sel = select_guideposts_from_scores(
        scores,
        num_guideposts=3,
        sink_size=2,
        recent_size=4,
        block_size=5,
        candidates_per_block=1,
        fallback_uniform=False,
    )
    # Blocks are [2,7), [7,12), [12,16); local maxima are 6, 11, 15.
    assert sel.candidate_indices.tolist() == [6, 11, 15]
    assert sel.indices.tolist() == [6, 11, 15]


def test_uniform_guideposts_never_selects_protected_positions():
    idx = select_uniform_guideposts(seq_len=16, num_guideposts=4, sink_size=2, recent_size=3)
    assert idx.numel() <= 4
    assert all(2 <= int(i) < 13 for i in idx.tolist())
