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

        if normalized_shape is not None:
            self.d_model = normalized_shape if isinstance(normalized_shape, int) else normalized_shape[-1]
        else:
            self.d_model = getattr(cfg, "d_model", 768)

        self.eps = eps if eps is not None else getattr(cfg, "layer_norm_eps", 1e-5)

        shape = normalized_shape if normalized_shape is not None else self.d_model
        self.normalized_shape = (shape,) if isinstance(shape, int) else tuple(shape)
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
        return torch.nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps)
