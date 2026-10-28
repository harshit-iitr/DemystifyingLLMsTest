"""Causal attention masks."""

from typing import Optional

import torch
import torch.nn as nn


def create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return an additive mask of shape (1, 1, seq_len, seq_len).

    Earlier and current positions are allowed (0); future positions are
    masked (-inf).
    """
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    future_positions = torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
        diagonal=1,
    )
    mask = mask.masked_fill(future_positions, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


class CausalMask(nn.Module):
    """Causal mask module."""

    def __init__(self, max_seq_len: int = 2048):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
            )
        return create_causal_mask(seq_len, device=device, dtype=dtype)
