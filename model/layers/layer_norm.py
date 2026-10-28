"""
Layer Normalization module.

"""

from typing import Optional, Any, Union, List
import torch
import torch.nn as nn


def _resolve_config(explicit_config: Optional[Any] = None) -> Any:
    if explicit_config is not None:
        return explicit_config
    from config.model_config import ModelConfig
    return ModelConfig()


class LayerNorm(nn.Module):
    """
    Layer Normalization layer with learnable affine parameters (gamma, beta).
    """

    def __init__(
        self,
        normalized_shape: Optional[Union[int, List[int]]] = None,
        eps: Optional[float] = None,
        config: Optional[Any] = None
    ):
        super().__init__()
        cfg = _resolve_config(config)

        if normalized_shape is None:
            normalized_shape = getattr(cfg, "d_model", 768)
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        if not self.normalized_shape or any(dim <= 0 for dim in self.normalized_shape):
            raise ValueError("normalized_shape must contain positive dimensions")
        self.d_model = self.normalized_shape[-1]
        self.eps = eps if eps is not None else getattr(cfg, "layer_norm_eps", 1e-5)
        if self.eps < 0:
            raise ValueError("eps must be non-negative")
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
