# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class ChunkKVPress(BasePress):
    """
    ChunkKV: Semantic-preserving compression with chunk-wise token selection.

    Enhances any ScorerPress by applying chunk-wise token selection instead of
    global selection. Computes global importance scores, then selects tokens
    chunk by chunk to preserve semantic coherence within local contexts.

    Based on ChunkKV (https://arxiv.org/abs/2502.00299).

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method used to compute global importance scores.
    chunk_length : int, default=20
        Length of each chunk for token selection.
        Sequence is divided into chunks of this size, with tokens selected
        proportionally from each chunk based on global importance scores.
    """

    press: ScorerPress
    chunk_length: int = 20
    exact_kept_tokens: int | None = None

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "ChunkKVPress requires a ScorerPress as input"
        if self.exact_kept_tokens is not None and self.exact_kept_tokens <= 0:
            raise ValueError("exact_kept_tokens must be positive")

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.press.compression_ratio == 0:
            return keys, values

        assert attentions is None, "ChunkPress does not support attentions."

        kv_len = keys.shape[2]

        # 1. Calculate global scores first
        global_scores = self.press.score(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
        )

        # 2. Calculate actual number of complete chunks and remaining tokens
        num_complete_chunks = kv_len // self.chunk_length
        remaining_tokens = kv_len % self.chunk_length

        # If we have no complete chunks, delegate to the underlying scorer press
        if num_complete_chunks == 0:
            return self.press.compress(module, hidden_states, keys, values, attentions, kwargs)

        # Reshape complete chunks for score calculation
        if num_complete_chunks > 0:
            main_scores = global_scores[..., : num_complete_chunks * self.chunk_length]
            main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, self.chunk_length)
            main_chunk_scores = main_chunk_scores.mean(dim=-1)
        else:
            main_chunk_scores = torch.empty((global_scores.shape[0], 0), device=global_scores.device)

        # Handle remaining tokens if any
        if remaining_tokens > 0:
            remaining_scores = global_scores[..., -remaining_tokens:]
            remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
            chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
        else:
            chunk_scores = main_chunk_scores

        # 3. Calculate number of chunks to keep
        n_chunks_kept = max(1, int((num_complete_chunks + (remaining_tokens > 0)) * (1 - self.press.compression_ratio)))
        top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1)

        # 4. Create indices for selected chunks
        indices = []
        for chunk_idx in top_chunks.indices[0]:
            if chunk_idx < num_complete_chunks:
                # For complete chunks
                start_idx = chunk_idx * self.chunk_length
                chunk_indices = torch.arange(start_idx, start_idx + self.chunk_length, device=keys.device)
            else:
                # For the remaining partial chunk
                chunk_indices = torch.arange(num_complete_chunks * self.chunk_length, kv_len, device=keys.device)
            indices.append(chunk_indices)

        indices = torch.cat(indices).sort()[0]
        if self.exact_kept_tokens is not None:
            target = min(kv_len, self.exact_kept_tokens)
            token_scores = global_scores.sum(dim=1)[0]
            if len(indices) > target:
                # Keep the highest-scoring tokens within the selected semantic
                # chunks when a whole chunk would exceed the hard byte budget.
                local = token_scores[indices].topk(target).indices
                indices = indices[local].sort()[0]
            elif len(indices) < target:
                # Preserve all selected chunks, then spend only the exact
                # residual on the strongest tokens outside those chunks.
                available = torch.ones(kv_len, dtype=torch.bool, device=keys.device)
                available[indices] = False
                candidates = torch.arange(kv_len, device=keys.device)[available]
                residual = min(target - len(indices), len(candidates))
                if residual:
                    added = candidates[token_scores[candidates].topk(residual).indices]
                    indices = torch.cat((indices, added)).sort()[0]
        indices = indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)

        # 5. Use gather to collect selected keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
