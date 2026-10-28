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
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    was_1d = logits.dim() == 1
    if was_1d:
        logits = logits.unsqueeze(0)
    if isinstance(generated_tokens, torch.Tensor) and generated_tokens.dim() > 1:
        token_rows = generated_tokens.reshape(generated_tokens.size(0), -1).tolist()
    elif isinstance(generated_tokens, torch.Tensor):
        token_rows = [generated_tokens.reshape(-1).tolist()] * logits.size(0)
    elif generated_tokens and isinstance(generated_tokens[0], (list, tuple)):
        token_rows = list(generated_tokens)
    else:
        token_rows = [generated_tokens] * logits.size(0)

    for row_idx, tokens in enumerate(token_rows[:logits.size(0)]):
        for token_id in set(tokens):
            if token_id < 0 or token_id >= logits.size(-1):
                continue
            row = logits[row_idx] if logits.size(0) > 1 else logits
            values = row[..., token_id]
            row[..., token_id] = torch.where(values < 0, values * penalty, values / penalty)

    return logits.squeeze(0) if was_1d else logits
