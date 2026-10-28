"""Module implementation."""
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits (returns scaled logits, not probabilities).
    temperature < 1 sharpens the distribution, > 1 flattens it.
    temperature == 0 is treated as greedy: all mass on the argmax.
    """
    if temperature <= 0.0:
        greedy = torch.full_like(logits, float("-inf"))
        idx = logits.argmax(dim=-1, keepdim=True)
        return greedy.scatter(-1, idx, 0.0)
    return logits / temperature
