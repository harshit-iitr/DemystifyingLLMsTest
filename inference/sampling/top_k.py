"""Top-k sampling."""
import torch


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Mask logits outside top-k.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits

    values, _ = torch.topk(logits, k, largest=True, dim=-1)
    threshold = values[..., -1, None]
    filtered_logits = logits.clone()
    filtered_logits[filtered_logits < threshold] = float("-inf")
    return filtered_logits
