"""Top-k sampling."""
import torch


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Mask logits outside top-k.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits

    _, idx = torch.topk(logits, k, largest=True, dim=-1)
    filtered_logits = logits.clone()
    keep_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, idx, True)
    filtered_logits.masked_fill_(~keep_mask, float("-inf"))
    return filtered_logits
