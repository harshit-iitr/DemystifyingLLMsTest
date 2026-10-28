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
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)

    if isinstance(generated_tokens, torch.Tensor):
        token_rows = generated_tokens.reshape(-1, generated_tokens.size(-1)) if generated_tokens.dim() > 1 else None
        if token_rows is None:
            shared_tokens = torch.unique(generated_tokens).tolist()
            token_rows = [shared_tokens] * flat_logits.size(0)
        else:
            token_rows = [torch.unique(row).tolist() for row in token_rows]
    elif generated_tokens and isinstance(generated_tokens[0], (list, tuple)):
        token_rows = [list(set(row)) for row in generated_tokens]
    else:
        token_rows = [list(set(generated_tokens))] * flat_logits.size(0)

    if len(token_rows) == 1 and flat_logits.size(0) > 1:
        token_rows *= flat_logits.size(0)
    if len(token_rows) != flat_logits.size(0):
        raise ValueError("generated_tokens batch dimension must match logits")

    for row_idx, row_tokens in enumerate(token_rows):
        for token_id in row_tokens:
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"generated token id {token_id} is outside vocabulary size {vocab_size}")
            token_logit = flat_logits[row_idx, token_id]
            flat_logits[row_idx, token_id] = torch.where(token_logit > 0, token_logit / penalty, token_logit * penalty)

    return flat_logits.reshape_as(logits)
