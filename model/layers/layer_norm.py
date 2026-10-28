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
            if isinstance(normalized_shape, int):
                self.normalized_shape = (normalized_shape,)
            else:
                self.normalized_shape = tuple(normalized_shape)
            if not self.normalized_shape or any(dim <= 0 for dim in self.normalized_shape):
                raise ValueError("normalized_shape dimensions must be positive")
            self.d_model = self.normalized_shape[-1]
        else:
            self.d_model = getattr(cfg, "d_model", 768)
            self.normalized_shape = (self.d_model,)

        self.eps = eps if eps is not None else getattr(cfg, "layer_norm_eps", 1e-5)
        if self.eps <= 0:
            raise ValueError("eps must be positive")

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
        norm_dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=norm_dims, keepdim=True)
        var = x.var(dim=norm_dims, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias
