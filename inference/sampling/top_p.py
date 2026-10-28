"""Nucleus top-p sampling."""
import math
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"top_p must be in [0.0, 1.0], got {p}")
    if p == 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    remove_mask = cumulative_probs > p
    # Keep the first token that crosses the threshold as well as the top token.
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    original_order_mask = torch.zeros_like(remove_mask).scatter(-1, sorted_indices, remove_mask)
    filtered_logits = logits.masked_fill(original_order_mask, float("-inf"))
    return filtered_logits
