"""Combine token and positional embeddings."""

from typing import Any, Optional

import torch
import torch.nn as nn

from config.model_config import ModelConfig
from .token_embedding import TokenEmbedding
from .position_embedding import LearnedPositionEmbedding, SinusoidalPositionEmbedding


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        config: Optional[Any] = None,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        dropout: Optional[float] = None,
        learned_positions: bool = True,
    ):
        super().__init__()
        cfg = config or ModelConfig()
        self.d_model = d_model or cfg.d_model
        self.vocab_size = vocab_size or cfg.vocab_size
        self.max_seq_len = max_seq_len or cfg.max_seq_len
        self.token_embedding = TokenEmbedding(self.vocab_size, self.d_model)
        position_cls = LearnedPositionEmbedding if learned_positions else SinusoidalPositionEmbedding
        self.position_embedding = position_cls(self.max_seq_len, self.d_model)
        self.dropout = nn.Dropout(cfg.dropout if dropout is None else dropout)

    def forward(self, input_ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"EmbeddingLayer expected torch.Tensor, got {type(input_ids)}")
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (batch, seq_len); got {tuple(input_ids.shape)}"
            )

        seq_len = input_ids.size(1)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(
            seq_len=seq_len, offset=offset, device=input_ids.device
        )
        return self.dropout(token_embeddings + position_embeddings)
