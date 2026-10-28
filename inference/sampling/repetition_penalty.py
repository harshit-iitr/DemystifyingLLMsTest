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
    if not torch.isfinite(torch.tensor(penalty)) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor):
        token_data = generated_tokens.detach().cpu()
        if token_data.ndim <= 1:
            token_rows = [token_data.reshape(-1).tolist()] * (logits.size(0) if logits.ndim > 1 else 1)
        elif logits.ndim > 1 and token_data.size(0) == logits.size(0):
            token_rows = [row.reshape(-1).tolist() for row in token_data]
        else:
            token_rows = [token_data.reshape(-1).tolist()] * (logits.size(0) if logits.ndim > 1 else 1)
    elif generated_tokens and isinstance(generated_tokens[0], (list, tuple)):
        token_rows = [list(row) for row in generated_tokens]
    else:
        token_rows = [list(generated_tokens)] * (logits.size(0) if logits.ndim > 1 else 1)

    flat_logits = logits.reshape(-1, logits.size(-1))
    if len(token_rows) == 1 and flat_logits.size(0) > 1:
        token_rows *= flat_logits.size(0)
    if len(token_rows) != flat_logits.size(0):
        raise ValueError("generated_tokens batch dimension must match logits")

    for row, token_ids in zip(flat_logits, token_rows):
        for token_id in set(int(token) for token in token_ids):
            if 0 <= token_id < row.numel():
                value = row[token_id]
                row[token_id] = value * penalty if value < 0 else value / penalty

    return logits

