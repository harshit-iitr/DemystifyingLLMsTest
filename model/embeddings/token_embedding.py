import math
from typing import Optional

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Map token IDs to vectors."""

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        scale_embeddings: bool = False,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale_embeddings = scale_embeddings

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"input_ids must have an integer dtype; got {input_ids.dtype}"
            )

        if input_ids.numel() > 0:
            if input_ids.min() < 0 or input_ids.max() >= self.vocab_size:
                raise ValueError(
                    f"Token IDs must be in [0, {self.vocab_size - 1}]"
                )

        embeddings = self.embedding(input_ids)
        if self.scale_embeddings:
            embeddings = embeddings * math.sqrt(self.d_model)
        return embeddings
