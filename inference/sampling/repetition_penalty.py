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
    Divide positive logits by the penalty and multiply negative logits by it.
    """
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor):
        unique_tokens = torch.unique(generated_tokens).tolist()
    else:
        unique_tokens = list(set(generated_tokens))

    for token_id in unique_tokens:
        if 0 <= token_id < logits.size(-1):
            selected = logits[..., token_id]
            logits[..., token_id] = torch.where(selected > 0, selected / penalty, selected * penalty)

    return logits
