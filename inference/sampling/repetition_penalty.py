"""Repetition penalty."""
import torch
import math
from typing import List, Union


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Union[List[int], torch.Tensor],
    penalty: float = 1.2
) -> torch.Tensor:
    """
    Apply repetition penalty to logits for generated tokens.
    Apply the standard sign-aware repetition penalty.
    """
    if penalty == 1.0:
        return logits
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("penalty must be positive")

    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor) and generated_tokens.dim() > 1:
        token_rows = generated_tokens
    else:
        token_rows = [generated_tokens]

    for row, tokens in enumerate(token_rows):
        unique_tokens = torch.unique(tokens).tolist() if isinstance(tokens, torch.Tensor) else set(tokens)
        for token_id in unique_tokens:
            token_id = int(token_id)
            if token_id < 0 or token_id >= logits.size(-1):
                continue
            target = logits[row, token_id] if len(token_rows) == logits.size(0) else logits[..., token_id]
            penalized = torch.where(target > 0, target / penalty, target * penalty)
            if len(token_rows) == logits.size(0):
                logits[row, token_id] = penalized
            else:
                logits[..., token_id] = penalized

    return logits
