"""Module implementation."""
import torch


def apply_temperature(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature
