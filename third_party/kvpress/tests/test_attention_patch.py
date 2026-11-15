# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from kvpress.attention_patch import attention_patch, search_hyperplane


def test_search_hyperplane():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bsz, seq_len, head_dim = 50, 500, 128
    X = torch.rand(bsz, seq_len, head_dim, device=device)
    Y = search_hyperplane(X)
    assert torch.exp(torch.bmm(X, Y.unsqueeze(-1))).max() == 0


def test_attention_patch_keeps_right_aligned_causal_suffix_for_per_layer_cache():
    captured = {}

    def attention(module, query, key, value, attention_mask, dropout, **kwargs):
        captured["attention_mask"] = attention_mask
        return query, None

    query = torch.zeros(1, 2, 3, 4)
    key = value = torch.zeros(1, 1, 7, 4)
    attention_mask = torch.arange(27).reshape(1, 1, 3, 9)

    attention_patch(attention)(
        SimpleNamespace(masked_key_indices=None),
        query,
        key,
        value,
        attention_mask,
        0.0,
    )

    assert torch.equal(captured["attention_mask"], attention_mask[..., -7:])
