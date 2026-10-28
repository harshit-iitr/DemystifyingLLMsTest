"""Nucleus top-p sampling."""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if p >= 1.0 or p <= 0.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    remove_mask = cumulative_probs > p
    # Shift right to keep the first token that exceeds p
    remove_mask = torch.roll(remove_mask, 1, dims=-1)
    remove_mask[..., 0] = False

    unsorted_remove_mask = torch.zeros_like(remove_mask)
    unsorted_remove_mask.scatter_(-1, sorted_indices, remove_mask)
    filtered_logits = logits.masked_fill(unsorted_remove_mask, float("-inf"))
    return filtered_logits
