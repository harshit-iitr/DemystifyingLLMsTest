"""Module implementation."""
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits (returns scaled logits, not probabilities).
    temperature == 0 degenerates to greedy: everything but the argmax is masked.
    """
    if temperature <= 0.0:
        greedy = torch.full_like(logits, float("-inf"))
        return greedy.scatter(-1, logits.argmax(dim=-1, keepdim=True), 0.0)
    return logits / temperature
