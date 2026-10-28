"""Module implementation."""
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits and return the resulting probability distribution.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return F.softmax(logits / temperature, dim=-1)
