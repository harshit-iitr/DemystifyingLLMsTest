"""Nucleus top-p sampling."""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if p >= 1.0:
        return logits
    if p < 0.0:
        raise ValueError("top_p must be non-negative")

    if p == 0.0:
        keep = logits.argmax(dim=-1, keepdim=True)
        filtered_logits = torch.full_like(logits, float("-inf"))
        return filtered_logits.scatter(-1, keep, logits.gather(-1, keep))

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = cumulative_probs > p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
    filtered_logits = torch.full_like(logits, float("-inf"))
    return filtered_logits.scatter(-1, sorted_indices, sorted_logits)
