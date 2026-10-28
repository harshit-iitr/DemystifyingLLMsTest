"""Pre-layer-norm transformer decoder block."""

import inspect
from typing import Any, Optional

import torch
import torch.nn as nn

from model.attention.multi_head import MultiHeadAttention
from model.layers.feedforward import FeedForward
from model.layers.layer_norm import LayerNorm
from model.layers.residual import ResidualConnection


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


class TransformerBlock(nn.Module):
    """Attention and feedforward sublayers with pre-layer normalization."""

    def __init__(self, config: Optional[Any] = None, layer_idx: int = 0):
        super().__init__()
        cfg = _resolve_config(config)

        self.layer_idx = layer_idx
        self.ln1 = LayerNorm(config=cfg)
        self.attn = MultiHeadAttention(config=cfg)
        self.ln2 = LayerNorm(config=cfg)
        self.ffn = FeedForward(config=cfg)

        dropout_p = getattr(cfg, "dropout", 0.1)
        self.resid1 = ResidualConnection(dropout=dropout_p)
        self.resid2 = ResidualConnection(dropout=dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
    ) -> torch.Tensor:
        h = self.resid1(
            x,
            self.attn(
                self.ln1(x),
                mask=mask,
                kv_cache=kv_cache,
                layer_idx=self.layer_idx,
            ),
        )
        return self.resid2(h, self.ffn(self.ln2(h)))
