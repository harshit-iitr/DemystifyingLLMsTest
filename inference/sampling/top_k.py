"""Top-k sampling."""
import torch


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Mask logits outside top-k.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits

    top_indices = torch.topk(logits, k, largest=True, dim=-1).indices
    keep_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, top_indices, True)
    return logits.masked_fill(~keep_mask, float("-inf"))
