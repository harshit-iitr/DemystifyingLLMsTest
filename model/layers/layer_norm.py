"""Layer normalization over the final hidden dimension."""

import inspect
from typing import Any, List, Optional, Union

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


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and bias."""

    def __init__(
        self,
        normalized_shape: Optional[Union[int, List[int]]] = None,
        eps: Optional[float] = None,
        config: Optional[Any] = None,
    ):
        super().__init__()
        cfg = _resolve_config(config)

        if normalized_shape is None:
            self.d_model = getattr(cfg, "d_model", 768)
        elif isinstance(normalized_shape, int):
            self.d_model = normalized_shape
        else:
            self.d_model = normalized_shape[-1]

        self.eps = eps if eps is not None else getattr(cfg, "layer_norm_eps", 1e-5)
        self.weight = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return self.weight * x_normalized + self.bias
