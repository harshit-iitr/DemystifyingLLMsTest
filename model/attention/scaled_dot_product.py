"""
Scaled Dot-Product Attention implementation.

"""

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
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V

    Args:
        q: Query tensor of shape (batch, n_heads, seq_len_q, d_k)
        k: Key tensor of shape (batch, n_heads, seq_len_k, d_k)
        v: Value tensor of shape (batch, n_heads, seq_len_k, d_v)
        mask: Additive attention bias, or boolean mask (True means allowed)
        dropout: Optional dropout module

    Returns:
        Tuple of (output tensor, attention weights)
    """
    # Accumulate low-precision dot products and softmax in float32.
    dtype = q.dtype
    work_q = q.float() if dtype in (torch.float16, torch.bfloat16) else q
    work_k = k.to(work_q.dtype)
    scores = torch.matmul(work_q, work_k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        mask = mask.to(device=scores.device)
        if mask.dtype == torch.bool:
            # Match torch.scaled_dot_product_attention: True means allowed.
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.to(scores.dtype)
    # An entirely masked query has zero attention, rather than softmax(-inf)
    # NaNs. Avoid evaluating an undefined softmax so gradients stay finite too.
    blocked = torch.isneginf(scores).all(dim=-1, keepdim=True)
    scores = scores.masked_fill(blocked, 0.0)
    attn_weights = F.softmax(scores, dim=-1).masked_fill(blocked, 0.0).to(dtype)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    output = torch.matmul(attn_weights, v)

    return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention layer wrapper.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
