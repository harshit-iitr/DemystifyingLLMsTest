"""Top-k sampling."""
import torch


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Retain only the k highest logits along the vocabulary dimension.

    Args:
        logits: Input logits.
        k: Number of highest-scoring tokens to retain.
           Values <= 0 disable filtering.

    Returns:
        Filtered logits, with excluded tokens set to negative infinity.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits

    threshold = torch.topk(
        logits,
        k=k,
        dim=-1,
        largest=True,
    ).values[..., -1, None]

    return logits.masked_fill(logits < threshold, float("-inf"))
