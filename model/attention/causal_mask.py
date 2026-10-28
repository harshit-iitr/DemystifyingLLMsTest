"""
Causal masking module and utilities.

"""

from typing import Optional
import torch
import torch.nn as nn


def create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate an additive causal attention mask of shape (1, 1, seq_len, seq_len).

    In causal attention:
      Positions where j <= i should have value 0.0 (allowed).
      Positions where j > i should have value -inf (masked).

    are masked with -inf and future positions (j >= i) are allowed!
    """
    # mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1)
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    mask = mask.masked_fill(
        torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask.unsqueeze(0).unsqueeze(0)


class CausalMask(nn.Module):
    """
    Stateful causal mask module caching mask buffers up to max_seq_len.
    """

    def __init__(self, max_seq_len: int = 2048):
        super().__init__()
        if not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool) or max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        self.max_seq_len = max_seq_len
        base_mask = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float32)
        base_mask = base_mask.masked_fill(
            torch.triu(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
        self.register_buffer("_mask", base_mask, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 0:
            raise ValueError("seq_len must be a non-negative integer")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})")
        mask = self._mask[:seq_len, :seq_len]
        if device is not None or dtype != mask.dtype:
            mask = mask.to(device=device or mask.device, dtype=dtype)
        return mask.unsqueeze(0).unsqueeze(0)
