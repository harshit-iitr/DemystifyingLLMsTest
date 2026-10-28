"""Module implementation."""
import torch
import math


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature
