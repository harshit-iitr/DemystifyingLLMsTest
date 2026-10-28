"""
Causal masking module and utilities.

"""

from typing import Optional
import torch
import torch.nn as nn


def create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    past_len: int = 0
) -> torch.Tensor:
    """Generate an additive causal mask for current queries and cached keys.

    In causal attention:
      Positions where j <= i should have value 0.0 (allowed).
      Positions where j > i should have value -inf (masked).
    Future positions (j > i) are masked with negative infinity.
    """
    if seq_len < 0 or past_len < 0:
        raise ValueError("seq_len and past_len must be non-negative")
    mask = torch.zeros((seq_len, past_len + seq_len), device=device, dtype=dtype)
    future = torch.triu(
        torch.ones((seq_len, past_len + seq_len), device=device, dtype=torch.bool),
        diagonal=past_len + 1
    )
    mask.masked_fill_(future, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


class CausalMask(nn.Module):
    """
    Stateful causal mask module caching mask buffers up to max_seq_len.
    """

    def __init__(self, max_seq_len: int = 2048):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return create_causal_mask(seq_len=seq_len, device=device, dtype=dtype)
