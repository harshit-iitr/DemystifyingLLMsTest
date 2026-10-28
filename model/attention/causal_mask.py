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
    key_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate an additive causal attention mask of shape (1, 1, query_len, key_len).

    In causal attention:
      Positions where j <= i should have value 0.0 (allowed).
      Positions where j > i should have value -inf (masked).

    Future positions (j > i) are masked with -inf.
    """
    if seq_len < 0 or past_len < 0:
        raise ValueError("seq_len and past_len must be non-negative")
    total_key_len = past_len + seq_len if key_len is None else key_len
    if total_key_len < past_len + seq_len:
        raise ValueError("key_len must be at least past_len + seq_len")
    query_positions = past_len + torch.arange(seq_len, device=device).unsqueeze(-1)
    key_positions = torch.arange(total_key_len, device=device).unsqueeze(0)
    mask = torch.zeros((seq_len, total_key_len), device=device, dtype=dtype)
    mask.masked_fill_(key_positions > query_positions, float("-inf"))
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
        if seq_len < 0 or seq_len > self.max_seq_len:
            raise ValueError(f"seq_len must be in [0, {self.max_seq_len}]")
        return create_causal_mask(seq_len=seq_len, device=device, dtype=dtype)
