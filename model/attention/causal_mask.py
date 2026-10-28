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
    key_len: Optional[int] = None,
    query_offset: int = 0,
) -> torch.Tensor:
    """
    Generate an additive causal attention mask of shape (1, 1, seq_len, seq_len).

    In causal attention:
      Positions where j <= i should have value 0.0 (allowed).
      Positions where j > i should have value -inf (masked).

    Positions later than the query are masked with -inf.
    """
    key_len = seq_len if key_len is None else key_len
    query_positions = torch.arange(query_offset, query_offset + seq_len, device=device)
    key_positions = torch.arange(key_len, device=device)
    future = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    mask = torch.zeros((seq_len, key_len), device=device, dtype=dtype)
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
