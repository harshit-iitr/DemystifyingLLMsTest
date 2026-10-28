"""Module implementation."""
import math
import torch


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    scaled = torch.nan_to_num(logits) / temperature
    return torch.nan_to_num(scaled)
