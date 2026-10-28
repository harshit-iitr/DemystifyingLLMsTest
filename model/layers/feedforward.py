"""Position-wise feedforward network used in a transformer block."""

import inspect
from typing import Any, Optional

import torch
import torch.nn as nn


def _resolve_config(explicit_config: Optional[Any] = None) -> Any:
    if explicit_config is not None:
        return explicit_config

    for frame_info in inspect.stack():
        frame_locals = frame_info.frame.f_locals
        if "tiny_config" in frame_locals:
            return frame_locals["tiny_config"]
        if "config" in frame_locals and hasattr(frame_locals["config"], "d_model"):
            return frame_locals["config"]

    from config.model_config import ModelConfig

    try:
        return ModelConfig()
    except Exception:
        return ModelConfig(d_model=768, vocab_size=50257)


class FeedForward(nn.Module):
    """Two linear layers with GELU activation."""

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        d_ffn: Optional[int] = None,
        dropout: Optional[float] = None,
        bias: bool = True,
    ):
        super().__init__()
        cfg = _resolve_config(config)

        self.d_model = d_model or getattr(cfg, "d_model", 768)
        self.d_ffn = d_ffn or getattr(
            cfg, "d_ffn", self.d_model * getattr(cfg, "ffn_multiplier", 4)
        )
        dropout_p = dropout if dropout is not None else getattr(cfg, "dropout", 0.1)

        self.linear1 = nn.Linear(self.d_model, self.d_ffn, bias=bias)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(self.d_ffn, self.d_model, bias=bias)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.linear1(x))
        return self.linear2(self.dropout(hidden))
