"""Module implementation."""
import torch
import math


def apply_temperature(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and greater than zero")
    return logits / temperature
