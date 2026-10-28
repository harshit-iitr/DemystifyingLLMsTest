"""
Residual connection module.
"""

from typing import Callable, Union

import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """Residual skip connection with optional dropout."""

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
        if callable(sublayer):
            sublayer_out = sublayer(x, *args, **kwargs)
        else:
            sublayer_out = sublayer

        return x + self.dropout(sublayer_out)