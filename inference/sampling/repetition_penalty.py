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
    Scale repeated-token logits away from zero, following the sign of each logit.
    """
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    token_ids = generated_tokens.reshape(-1).tolist() if isinstance(generated_tokens, torch.Tensor) else generated_tokens
    for token_id in set(int(token) for token in token_ids):
        if 0 <= token_id < logits.size(-1):
            token_logits = logits[..., token_id]
            logits[..., token_id] = torch.where(token_logits < 0, token_logits * penalty, token_logits / penalty)

    return logits
