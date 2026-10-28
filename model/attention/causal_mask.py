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
    Generate an additive causal attention mask of shape
    (1, 1, seq_len, past_len + seq_len).

    In causal attention:
      Positions where j <= i should have value 0.0 (allowed).
      Positions where j > i should have value -inf (masked).

    Future positions are masked with -inf. ``past_len`` aligns query position
    zero with its position after the cached keys.
    """
    if seq_len < 0 or past_len < 0:
        raise ValueError("seq_len and past_len must be non-negative")
    key_len = past_len + seq_len
    query_positions = torch.arange(seq_len, device=device) + past_len
    key_positions = torch.arange(key_len, device=device)
    future_positions = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    mask = torch.zeros((seq_len, key_len), device=device, dtype=dtype)
    mask.masked_fill_(future_positions, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


class CausalMask(nn.Module):
    """
    Causal mask module enforcing a maximum query-plus-cache length.
    """

    def __init__(self, max_seq_len: int = 2048):
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = max_seq_len

    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        past_len: int = 0,
    ) -> torch.Tensor:
        if seq_len + past_len > self.max_seq_len:
            raise ValueError("requested causal mask exceeds max_seq_len")
        return create_causal_mask(
            seq_len=seq_len, past_len=past_len, device=device, dtype=dtype
        )
