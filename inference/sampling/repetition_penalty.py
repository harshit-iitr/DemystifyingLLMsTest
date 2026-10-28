"""Sign-aware repetition penalty for previously generated IDs."""
import torch
from typing import List, Union


def apply_repetition_penalty(logits: torch.Tensor, generated_tokens: Union[List[int], torch.Tensor], penalty: float = 1.2) -> torch.Tensor:
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    result = logits.clone()
    if penalty == 1:
        return result
    raw_tokens = torch.as_tensor(generated_tokens, device=logits.device)
    if raw_tokens.numel() and (raw_tokens.dtype.is_floating_point or raw_tokens.dtype == torch.bool):
        raise ValueError("Token history must contain integer IDs")
    tokens = raw_tokens.to(torch.long)
    if tokens.numel() and ((tokens < 0).any() or (tokens >= logits.size(-1)).any()):
        raise ValueError("Token history contains an ID outside the logits vocabulary")
    def penalize(row, ids):
        ids = ids.unique()
        values = row[..., ids]
        row[..., ids] = torch.where(values < 0, values * penalty, values / penalty)
    if tokens.ndim > 1:
        if logits.ndim != 2 or tokens.size(0) != logits.size(0):
            raise ValueError("Token history batch must match logits batch")
        for row, ids in zip(result, tokens):
            penalize(row, ids)
    else:
        penalize(result, tokens)
    return result
