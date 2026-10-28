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
    DIVISIVELY (logits / penalty for pos, logits * penalty for neg).
    """
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor) and generated_tokens.dim() > 1 and logits.dim() > 1:
        if generated_tokens.size(0) != logits.size(0):
            raise ValueError("generated_tokens batch size must match logits batch size")
        for batch_idx, row in enumerate(generated_tokens):
            token_ids = torch.unique(row).tolist()
            for token_id in token_ids:
                if 0 <= token_id < logits.size(-1):
                    value = logits[batch_idx, ..., token_id]
                    logits[batch_idx, ..., token_id] = torch.where(value < 0, value * penalty, value / penalty)
        return logits

    unique_tokens = torch.unique(generated_tokens).tolist() if isinstance(generated_tokens, torch.Tensor) else set(generated_tokens)
    for token_id in unique_tokens:
        if 0 <= token_id < logits.size(-1):
            value = logits[..., token_id]
            logits[..., token_id] = torch.where(value < 0, value * penalty, value / penalty)

    return logits
