from types import SimpleNamespace

import torch

from proofkv.patch import _get_head_dim, _get_num_heads, _get_num_kv_heads


def test_attention_metadata_falls_back_to_transformers_config():
    module = torch.nn.Module()
    module.config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=2, head_dim=16, hidden_size=128)

    assert _get_num_heads(module) == 8
    assert _get_num_kv_heads(module) == 2
    assert _get_head_dim(module) == 16


def test_attention_head_dim_falls_back_to_hidden_size():
    module = torch.nn.Module()
    module.config = SimpleNamespace(num_attention_heads=4, hidden_size=64)

    assert _get_head_dim(module) == 16
