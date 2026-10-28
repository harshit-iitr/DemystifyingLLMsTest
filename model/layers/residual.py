"""Residual connection with optional dropout."""

from typing import Callable, Union

import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """Compute x + dropout(sublayer_output)."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: Union[torch.Tensor, Callable[..., torch.Tensor]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        sublayer_out = sublayer(x, *args, **kwargs) if callable(sublayer) else sublayer
        return x + self.dropout(sublayer_out)
