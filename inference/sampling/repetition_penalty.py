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
    if penalty <= 0:
        raise ValueError(f"penalty must be positive, got {penalty}")

    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor):
        if generated_tokens.ndim > 1 and logits.ndim > 1 and generated_tokens.size(0) == logits.size(0):
            tokens_by_row = [torch.unique(row).tolist() for row in generated_tokens]
        else:
            shared_tokens = torch.unique(generated_tokens).tolist()
            tokens_by_row = [shared_tokens] * (logits.size(0) if logits.ndim > 1 else 1)
    else:
        if generated_tokens and isinstance(generated_tokens[0], (list, tuple)) and logits.ndim > 1:
            tokens_by_row = [list(set(row)) for row in generated_tokens]
        else:
            shared_tokens = list(set(generated_tokens))
            tokens_by_row = [shared_tokens] * (logits.size(0) if logits.ndim > 1 else 1)

    rows = logits.unsqueeze(0) if logits.ndim == 1 else logits.reshape(-1, logits.size(-1))
    if len(tokens_by_row) != rows.size(0):
        tokens_by_row = [tokens_by_row[0]] * rows.size(0)
    for row, unique_tokens in zip(rows, tokens_by_row):
        for token_id in unique_tokens:
            token_logits = row[token_id]
            row[token_id] = torch.where(
                token_logits < 0, token_logits * penalty, token_logits / penalty
            )

    return logits
