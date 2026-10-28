"""Nucleus top-p sampling."""
import torch
import torch.nn.functional as F
import math


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if not math.isfinite(p):
        raise ValueError("top_p must be finite")
    if p >= 1.0:
        return logits
    if p < 0.0:
        raise ValueError("top_p must be non-negative")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    remove_mask = cumulative_probs > p
    # Shift right to keep the first token that exceeds p
    remove_mask = torch.roll(remove_mask, 1, dims=-1)
    remove_mask[..., 0] = False

    filtered_sorted = sorted_logits.masked_fill(remove_mask, float("-inf"))
    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(-1, sorted_indices, filtered_sorted)
    return filtered_logits
