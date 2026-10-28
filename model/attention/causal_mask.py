"""
Causal masking module and utilities.

"""

from typing import Optional
import torch
import torch.nn as nn
from config.constants import MAX_SEQ_LEN


def create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    key_len: Optional[int] = None,
    past_len: int = 0,
) -> torch.Tensor:
    """
    Generate an additive causal attention mask for current queries and cached keys.

    In causal attention:
      Positions where j <= i should have value 0.0 (allowed).
      Positions where j > i should have value -inf (masked).

    Future positions are masked with -inf.
    """
    if seq_len < 0 or past_len < 0:
        raise ValueError("seq_len and past_len must be non-negative")
    key_len = past_len + seq_len if key_len is None else key_len
    if key_len < past_len + seq_len:
        raise ValueError("key_len must cover the cached and current sequence")

    query_positions = torch.arange(past_len, past_len + seq_len, device=device)
    key_positions = torch.arange(key_len, device=device)
    future_positions = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    mask = torch.zeros((seq_len, key_len), device=device, dtype=dtype)
    mask.masked_fill_(future_positions, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


class CausalMask(nn.Module):
    """
    Stateful causal mask module caching mask buffers up to max_seq_len.
    """

    def __init__(self, max_seq_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        key_len: Optional[int] = None,
        past_len: int = 0,
    ) -> torch.Tensor:
        return create_causal_mask(
            seq_len=seq_len,
            device=device,
            dtype=dtype,
            key_len=key_len,
            past_len=past_len,
        )
