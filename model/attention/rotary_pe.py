"""
Rotary Positional Embeddings (RoPE).

Status: Clean — Red Herring (Complex rotation math, correct implementation of Su et al., 2021).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to Query and Key tensors.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) precomputing sinusoidal frequencies.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset < 0 or offset + seq_len > self.max_seq_len:
            raise ValueError(f"Requested rotary positions [{offset}, {offset + seq_len}) exceed max_seq_len {self.max_seq_len}")
        cos = self.cos_cached[offset:offset + seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[offset:offset + seq_len, :].to(dtype=x.dtype, device=x.device)
        # Reshape to (1, 1, seq_len, dim) for head broadcasting
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
