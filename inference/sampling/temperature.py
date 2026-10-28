"""Module implementation."""
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if temperature <= 0.0:
        # Zero temperature collapses to greedy: keep only the argmax logit(s)
        max_vals = logits.max(dim=-1, keepdim=True).values
        return logits.masked_fill(logits < max_vals, float("-inf"))
    return logits / temperature
