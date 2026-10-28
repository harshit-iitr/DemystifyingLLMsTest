"""Module implementation."""
import torch
def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return logits / temperature
