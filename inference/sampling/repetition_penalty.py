"""Repetition penalty."""
import torch
from typing import List, Union
import math


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Union[List[int], torch.Tensor],
    penalty: float = 1.2
) -> torch.Tensor:
    """
    Apply repetition penalty to logits for generated tokens.
    Apply the standard sign-aware repetition penalty.
    """
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be a finite positive number")
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor):
        unique_tokens = torch.unique(generated_tokens).tolist()
    else:
        unique_tokens = list(set(generated_tokens))

    if isinstance(generated_tokens, torch.Tensor) and generated_tokens.dim() >= 2 and logits.dim() >= 2 and generated_tokens.size(0) == logits.size(0):
        for batch_idx, row in enumerate(generated_tokens):
            for token_id in torch.unique(row).tolist():
                if token_id < 0 or token_id >= logits.size(-1):
                    raise ValueError(f"generated token id {token_id} is outside the logits vocabulary")
                value = logits[batch_idx, ..., token_id]
                logits[batch_idx, ..., token_id] = torch.where(value > 0, value / penalty, value * penalty)
    else:
        for token_id in unique_tokens:
            if token_id < 0 or token_id >= logits.size(-1):
                raise ValueError(f"generated token id {token_id} is outside the logits vocabulary")
            token_logits = logits[..., token_id]
            logits[..., token_id] = torch.where(token_logits > 0, token_logits / penalty, token_logits * penalty)

    return logits
