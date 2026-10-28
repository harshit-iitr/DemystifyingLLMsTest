"""Scaled dot-product attention."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q: (batch, heads, query_len, head_dim)
        k: (batch, heads, key_len, head_dim)
        v: (batch, heads, key_len, value_dim)
        mask: Optional additive mask; masked positions should contain -inf.

    Returns:
        Attention output and attention weights.
    """
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores + mask.to(device=scores.device, dtype=scores.dtype)

    weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    return torch.matmul(weights, v), weights


class ScaledDotProductAttention(nn.Module):
    """Module wrapper for scaled dot-product attention."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )
