"""Nucleus top-p sampling."""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if p >= 1.0:
        return logits

    if p <= 0.0:
        keep = logits.argmax(dim=-1, keepdim=True)
        return torch.full_like(logits, float("-inf")).scatter(-1, keep, logits.gather(-1, keep))

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_remove = cumulative_probs > p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    remove_mask = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
    filtered_logits = logits.clone()
    filtered_logits[remove_mask] = float("-inf")
    return filtered_logits
