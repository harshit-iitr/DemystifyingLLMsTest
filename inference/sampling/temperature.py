"""Module implementation."""
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    scaled = logits / temperature
    return scaled
