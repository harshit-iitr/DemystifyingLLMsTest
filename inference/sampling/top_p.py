"""Nucleus top-p sampling."""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if p >= 1.0 or p <= 0.0:
        return logits

    # Sort by probability (descending) so the nucleus is the most likely tokens
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens once cumulative mass (excluding the token itself) already exceeds p,
    # which keeps the first token that crosses the threshold.
    sorted_remove = (cumulative_probs - sorted_probs) > p
    sorted_remove[..., 0] = False

    remove_mask = torch.zeros_like(sorted_remove).scatter(-1, sorted_idx, sorted_remove)
    filtered_logits = logits.clone()
    filtered_logits[remove_mask] = float("-inf")
    return filtered_logits
