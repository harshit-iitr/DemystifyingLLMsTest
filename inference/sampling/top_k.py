"""Top-k sampling."""
import torch


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Mask logits outside top-k.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits

    values, indices = torch.topk(logits, k, largest=True, dim=-1)
    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(-1, indices, values)
    return filtered_logits
