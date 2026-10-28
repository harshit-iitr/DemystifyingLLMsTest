"""Keep the highest k logits."""
import torch


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0 or k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))
