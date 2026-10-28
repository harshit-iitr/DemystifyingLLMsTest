"""Keep the smallest descending-probability nucleus reaching p."""
import torch


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if p == 1:
        return logits
    sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(remove).scatter(-1, indices, remove)
    return logits.masked_fill(mask, float("-inf"))
