"""Nucleus top-p sampling."""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    if p <= 0.0:
        keep_best = torch.zeros_like(logits, dtype=torch.bool)
        keep_best.scatter_(-1, sorted_indices[..., :1], True)
        return logits.masked_fill(~keep_best, float("-inf"))
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_sorted = cumulative_probs > p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove_mask = torch.zeros_like(remove_sorted).scatter(-1, sorted_indices, remove_sorted)

    filtered_logits = logits.clone()
    filtered_logits[remove_mask] = float("-inf")
    return filtered_logits
