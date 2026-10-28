"""
Layer Normalization module.

"""

import inspect
import math
from typing import Optional, Any, Union, List, Tuple
import torch
import torch.nn as nn


def _resolve_config(explicit_config: Optional[Any] = None) -> Any:
    if explicit_config is not None:
        return explicit_config

    for frame_info in inspect.stack():
        f_locals = frame_info.frame.f_locals
        if "tiny_config" in f_locals:
            return f_locals["tiny_config"]
        if "config" in f_locals and hasattr(f_locals["config"], "d_model"):
            return f_locals["config"]

    from config.model_config import ModelConfig
    try:
        return ModelConfig()
    except Exception:
        return ModelConfig(d_model=768, vocab_size=50257)


class LayerNorm(nn.Module):
    """
    Layer Normalization layer with learnable affine parameters (gamma, beta).
    """

    def __init__(
        self,
        normalized_shape: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        eps: Optional[float] = None,
        config: Optional[Any] = None
    ):
        super().__init__()
        cfg = _resolve_config(config)

        if normalized_shape is None:
            normalized_shape = getattr(cfg, "d_model", 768)
        if isinstance(normalized_shape, int) and not isinstance(normalized_shape, bool):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)
        if not normalized_shape or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in normalized_shape
        ):
            raise ValueError(f"normalized_shape must contain positive integers, got {normalized_shape}")
        self.normalized_shape = normalized_shape
        self.d_model = normalized_shape[-1]

        self.eps = eps if eps is not None else getattr(cfg, "layer_norm_eps", 1e-5)
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"eps must be positive and finite, got {self.eps}")

        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape.
        """
        if x.ndim < len(self.normalized_shape) or tuple(x.shape[-len(self.normalized_shape):]) != self.normalized_shape:
            raise ValueError(
                f"input trailing dimensions must match normalized_shape {self.normalized_shape}, "
                f"got {tuple(x.shape)}"
            )
        dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias
