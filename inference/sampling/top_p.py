"""Nucleus top-p sampling."""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if p == 1.0:
        return logits

    if p <= 0.0:
        top_token = torch.argmax(logits, dim=-1, keepdim=True)
        return torch.full_like(logits, float("-inf")).scatter(-1, top_token, logits.gather(-1, top_token))

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove_mask = cumulative_probs - sorted_probs >= p
    remove_mask = torch.zeros_like(sorted_remove_mask).scatter(-1, sorted_indices, sorted_remove_mask)

    filtered_logits = logits.clone()
    filtered_logits[remove_mask] = float("-inf")
    return filtered_logits
