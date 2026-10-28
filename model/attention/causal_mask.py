"""
Causal masking module and utilities.

"""

from typing import Optional
import torch
import torch.nn as nn


def create_causal_mask(
    seq_len: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    *,
    query_len: Optional[int] = None,
    key_len: Optional[int] = None,
    past_len: int = 0,
) -> torch.Tensor:
    """
    Generate an additive causal attention mask of shape (1, 1, query_len, key_len).

    In causal attention:
      Positions where j <= i should have value 0.0 (allowed).
      Positions where j > i should have value -inf (masked).

    are masked with -inf. ``past_len`` supports incremental decoding with a KV cache.
    """
    if query_len is None:
        if seq_len is None:
            raise ValueError("seq_len or query_len must be provided")
        query_len = seq_len
    if key_len is None:
        key_len = query_len + past_len
    if query_len < 0 or key_len < 0 or past_len < 0:
        raise ValueError("sequence lengths and past_len must be non-negative")

    query_positions = torch.arange(past_len, past_len + query_len, device=device)
    key_positions = torch.arange(key_len, device=device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    mask = torch.zeros((query_len, key_len), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, float("-inf"))
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
