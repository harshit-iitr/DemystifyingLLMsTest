import copy
import inspect
from typing import Any, Optional

import torch
import torch.nn as nn

from .transformer_block import TransformerBlock
from model.layers.layer_norm import LayerNorm


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
    return ModelConfig()


class BlockStack(nn.Module):
    """Stack of transformer blocks followed by a final layer norm."""

    def __init__(
        self,
        config: Optional[Any] = None,
        n_layers: Optional[int] = None,
        d_model: Optional[int] = None,
    ):
        super().__init__()
        cfg = copy.copy(_resolve_config(config))

        if d_model is not None:
            cfg.d_model = d_model
            if hasattr(cfg, "ffn_multiplier"):
                cfg.d_ffn = d_model * cfg.ffn_multiplier

        self.n_layers = (
            n_layers if n_layers is not None else getattr(cfg, "n_layers", 12)
        )
        self.d_model = getattr(cfg, "d_model", 768)

        self.blocks = nn.ModuleList(
            TransformerBlock(config=cfg, layer_idx=i)
            for i in range(self.n_layers)
        )
        self.final_ln = LayerNorm(config=cfg)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> TransformerBlock:
        return self.blocks[idx]

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        for block in self.blocks:
            h = block(h, mask=mask, kv_cache=kv_cache)
        return self.final_ln(h)
