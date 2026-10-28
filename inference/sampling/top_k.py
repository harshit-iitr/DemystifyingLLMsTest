"""Top-k sampling."""
import torch


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Mask logits outside top-k.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0 or k >= logits.size(-1):
        return logits

    _, idx = torch.topk(logits, k, largest=True, dim=-1)
    filtered_logits = logits.clone()
    filtered_logits.fill_(float("-inf"))
    filtered_logits.scatter_(-1, idx, logits.gather(-1, idx))
    return filtered_logits
