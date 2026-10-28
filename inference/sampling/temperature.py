"""Module implementation."""
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    # Divide raw logits (not probabilities); clamp avoids division by zero (T->0 is greedy)
    return logits / max(temperature, 1e-8)
