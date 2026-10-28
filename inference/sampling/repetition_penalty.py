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
    Positive logits are divided by the penalty, negative logits are multiplied by it.
    """
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor):
        unique_tokens = torch.unique(generated_tokens).tolist()
    else:
        unique_tokens = list(set(generated_tokens))

    for token_id in unique_tokens:
        score = logits[..., token_id]
        logits[..., token_id] = torch.where(score > 0, score / penalty, score * penalty)

    return logits
