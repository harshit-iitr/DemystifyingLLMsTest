"""Nucleus (top-p) sampling."""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Retain the smallest set of highest-probability tokens whose
    cumulative probability reaches or exceeds p.

    Args:
        logits: Input logits.
        p: Cumulative probability threshold in [0, 1].

    Returns:
        Filtered logits in their original vocabulary order.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in the range [0, 1].")

    if p == 1.0:
        return logits

    if p == 0.0:
        # Retain the single highest-logit token.
        top_token = logits.argmax(dim=-1, keepdim=True)
        keep_mask = torch.zeros_like(logits, dtype=torch.bool)
        keep_mask.scatter_(-1, top_token, True)
        return logits.masked_fill(~keep_mask, float("-inf"))

    sorted_logits, sorted_indices = torch.sort(
        logits,
        dim=-1,
        descending=True,
    )

    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Initially mark tokens that take cumulative probability beyond p.
    remove_mask = cumulative_probs > p

    # Shift right so the first token that crosses p remains available.
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    # Restore the removal mask to the original vocabulary ordering.
    original_remove_mask = torch.zeros_like(remove_mask)
    original_remove_mask.scatter_(-1, sorted_indices, remove_mask)

    return logits.masked_fill(original_remove_mask, float("-inf"))
