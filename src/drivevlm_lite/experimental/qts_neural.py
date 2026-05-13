"""Learned query-aware visual token selector (unused in v1).

This module implements the original "QTS" idea: score visual tokens against
a query embedding, keep the top-K, and let the dropped tokens reinject
information into the kept tokens via cross-attention.

It was never integrated into Qwen3-VL training or evaluation in v1. The
"QTS-lite" results in earlier experiments came from rule-based camera
selection (now inlined in ``scripts/09_eval_drivelm_qts_input.py``), not
from this module.

Kept here so the architectural idea is not lost. If a v2 pass wants to add
a learned token selector, this is the starting point.
"""

from __future__ import annotations

import torch
from torch import nn


class QueryAwareTokenSelector(nn.Module):
    """Select visual tokens conditioned on a text/query embedding."""

    def __init__(self, hidden_dim: int, keep_ratio: float = 0.25, heads: int = 4, gamma: float = 0.5):
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")
        self.keep_ratio = keep_ratio
        self.gamma = gamma
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

    def forward(self, visual_tokens: torch.Tensor, query_embedding: torch.Tensor):
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must be [batch, tokens, hidden].")
        if query_embedding.ndim != 2:
            raise ValueError("query_embedding must be [batch, hidden].")

        batch, token_count, hidden = visual_tokens.shape
        query = query_embedding.unsqueeze(1).expand(-1, token_count, -1)
        scores = self.score_mlp(torch.cat([visual_tokens, visual_tokens * query], dim=-1)).squeeze(-1)

        keep_count = max(1, int(token_count * self.keep_ratio))
        topk = scores.topk(keep_count, dim=1).indices
        gather_idx = topk.unsqueeze(-1).expand(-1, -1, hidden)
        kept = visual_tokens.gather(1, gather_idx)

        keep_mask = torch.zeros(batch, token_count, dtype=torch.bool, device=visual_tokens.device)
        keep_mask.scatter_(1, topk, True)

        # Pad dropped tokens per batch so cross-attention can stay batched.
        dropped_rows = []
        max_dropped = token_count - keep_count
        for b_idx in range(batch):
            dropped = visual_tokens[b_idx, ~keep_mask[b_idx]]
            if dropped.shape[0] < max_dropped:
                pad = dropped.new_zeros(max_dropped - dropped.shape[0], hidden)
                dropped = torch.cat([dropped, pad], dim=0)
            dropped_rows.append(dropped)
        dropped_tokens = torch.stack(dropped_rows, dim=0)

        if dropped_tokens.shape[1] > 0:
            attn_out, _ = self.cross_attn(kept, dropped_tokens, dropped_tokens)
            kept = kept + self.gamma * attn_out

        return kept, topk, scores
