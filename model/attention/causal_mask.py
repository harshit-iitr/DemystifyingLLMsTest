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
    past_len: int = 0,
) -> torch.Tensor:
    """
    Generate an additive causal mask of shape (1, 1, seq_len, past_len + seq_len).

    In causal attention:
      Positions where j <= i should have value 0.0 (allowed).
      Positions where j > i should have value -inf (masked).

    are masked with -inf and future positions (j > i) are disallowed.
    """
    if seq_len < 0 or past_len < 0:
        raise ValueError("seq_len and past_len must be non-negative")
    query_positions = torch.arange(past_len, past_len + seq_len, device=device)
    key_positions = torch.arange(past_len + seq_len, device=device)
    future_positions = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    mask = torch.zeros((seq_len, past_len + seq_len), device=device, dtype=dtype)
    mask.masked_fill_(future_positions, float("-inf"))
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
