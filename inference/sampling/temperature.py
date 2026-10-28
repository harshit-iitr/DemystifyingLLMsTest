"""Module implementation."""
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if temperature <= 0.0:
        greedy = torch.argmax(logits, dim=-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(-1, greedy, 1.0)
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    return probs
