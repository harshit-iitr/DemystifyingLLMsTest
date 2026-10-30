"""Top-k sampling."""
import torch


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep the k tokens with the highest logits and mask the rest.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits

    # The kth largest logit is the cutoff; values below it are excluded.
    threshold = torch.topk(logits, k, largest=True, dim=-1).values[..., -1, None]
    filtered_logits = logits.clone()
    filtered_logits[filtered_logits < threshold] = float("-inf")
    return filtered_logits
