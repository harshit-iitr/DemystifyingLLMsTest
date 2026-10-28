"""Nucleus sampling in descending probability order."""
import torch


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    if p < 0.0:
        raise ValueError("p must be non-negative")
    sorted_logits, indices = logits.sort(dim=-1, descending=True)
    remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove = torch.zeros_like(remove).scatter(-1, indices, remove)
    return logits.masked_fill(remove, float('-inf'))
