"""Repetition penalty."""
import torch
from typing import List, Union


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Union[List[int], torch.Tensor],
    penalty: float = 1.2
) -> torch.Tensor:
    """
    Apply repetition penalty to logits for generated tokens.
    Divide positive logits and multiply negative logits by the penalty.
    """
    if penalty == 1.0:
        return logits
    if penalty <= 0:
        raise ValueError(f"penalty must be positive, got {penalty}")

    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor):
        unique_tokens = torch.unique(generated_tokens).tolist()
    else:
        unique_tokens = list(set(generated_tokens))

    for token_id in unique_tokens:
        token_logits = logits[..., token_id]
        logits[..., token_id] = torch.where(token_logits > 0, token_logits / penalty, token_logits * penalty)

    return logits
