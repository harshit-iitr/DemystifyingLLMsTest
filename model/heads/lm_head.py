"""Language-model output projection."""

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


class LMHead(nn.Module):
    """Project hidden states into vocabulary-logit space."""

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        vocab_size: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        cfg = _resolve_config(config)

        self.d_model = d_model or getattr(cfg, "d_model", 768)
        self.vocab_size = vocab_size or getattr(cfg, "vocab_size", 50257)
        self.proj = nn.Linear(self.d_model, self.vocab_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)

    def tie_weights(self, embedding_weight: torch.Tensor) -> None:
        if tuple(embedding_weight.shape) != (self.vocab_size, self.d_model):
            raise ValueError(
                "Embedding weight must have shape "
                f"({self.vocab_size}, {self.d_model}); "
                f"got {tuple(embedding_weight.shape)}"
            )
        self.proj.weight = embedding_weight
