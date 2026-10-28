"""Module implementation."""
import math
import torch


def apply_temperature(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if temperature == 0.0:
        # A zero temperature is the greedy limit. Encode it as a categorical
        # distribution so callers can still apply softmax safely.
        greedy_ids = logits.argmax(dim=-1, keepdim=True)
        return torch.full_like(logits, float("-inf")).scatter_(-1, greedy_ids, 0.0)
    return logits / temperature
