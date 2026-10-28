"""Nucleus top-p sampling."""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if p >= 1.0 or p <= 0.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove_mask = cumulative_probs > p
    sorted_remove_mask = torch.roll(sorted_remove_mask, 1, dims=-1)
    sorted_remove_mask[..., 0] = False

    remove_mask = sorted_remove_mask.scatter(-1, sorted_idx, sorted_remove_mask)

    filtered_logits = logits.masked_fill(remove_mask, float("-inf"))
    return filtered_logits
