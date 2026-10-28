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

    Future positions (j > i) are masked with -inf.
    """
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1
    )
    return mask.unsqueeze(0).unsqueeze(0)


class CausalMask(nn.Module):
    """
    Stateful causal mask module caching mask buffers up to max_seq_len.
    """

    def __init__(self, max_seq_len: int = 2048):
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.max_seq_len = max_seq_len
        base_mask = create_causal_mask(max_seq_len)[0, 0]
        self.register_buffer("mask", base_mask, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if seq_len < 0 or seq_len > self.max_seq_len:
            raise ValueError(f"seq_len must be in [0, {self.max_seq_len}], got {seq_len}")
        target_device = torch.device(device) if device is not None else self.mask.device
        if self.mask.device != target_device or self.mask.dtype != dtype:
            self.mask = create_causal_mask(
                seq_len=self.max_seq_len,
                device=target_device,
                dtype=dtype,
            )[0, 0]
        return self.mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
