"""Module implementation."""
import torch


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        return logits
    return logits / temperature
