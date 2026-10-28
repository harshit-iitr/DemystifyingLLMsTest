"""Sinusoidal and learned positional embeddings."""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed sinusoidal positional encodings."""

    def __init__(self, d_model: int = 768, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(
        self,
        seq_len: int,
        offset: int = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        end = offset + seq_len
        if offset < 0 or end > self.max_seq_len:
            raise ValueError(
                f"Requested positions [{offset}, {end}) exceed max_seq_len "
                f"{self.max_seq_len}"
            )

        encoding = self.pe[:, offset:end, :]
        return encoding.to(device=device) if device is not None else encoding


class LearnedPositionEmbedding(nn.Module):
    """Learned positional embedding table."""

    def __init__(self, max_seq_len: int = 1024, d_model: int = 768):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.wpe = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)

    def forward(
        self,
        seq_len: int,
        offset: int = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        end = offset + seq_len
        if offset < 0 or end > self.max_seq_len:
            raise ValueError(
                f"Requested positions [{offset}, {end}) exceed max_seq_len "
                f"{self.max_seq_len}"
            )

        positions = torch.arange(offset, end, dtype=torch.long, device=device)
        return self.wpe(positions).unsqueeze(0)
