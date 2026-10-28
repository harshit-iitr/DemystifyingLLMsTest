"""Nucleus top-p sampling."""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if not 0.0 < p <= 1.0:
        if p == 0.0:
            p = torch.finfo(logits.dtype).eps
        else:
            raise ValueError(f"p must be in [0, 1], got {p}")
    if p >= 1.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = cumulative_probs > p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    sorted_remove_mask = torch.zeros_like(remove_mask).scatter(-1, sorted_indices, remove_mask)

    filtered_logits = logits.clone()
    filtered_logits[sorted_remove_mask] = float("-inf")
    return filtered_logits
