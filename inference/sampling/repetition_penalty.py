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
    if isinstance(generated_tokens, torch.Tensor):
        if logits.dim() > 1 and generated_tokens.dim() > 1:
            if generated_tokens.size(0) != logits.size(0):
                raise ValueError("generated_tokens batch size must match logits")
            token_rows = [torch.unique(row).tolist() for row in generated_tokens]
        else:
            unique_tokens = torch.unique(generated_tokens).tolist()
            token_rows = [unique_tokens] * (logits.size(0) if logits.dim() > 1 else 1)
    else:
        unique_tokens = list(set(generated_tokens))
        token_rows = [unique_tokens] * (logits.size(0) if logits.dim() > 1 else 1)

    for row_idx, token_ids in enumerate(token_rows):
        row = logits[row_idx] if logits.dim() > 1 else logits
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id < 0 or token_id >= logits.size(-1):
                continue
            row[..., token_id] = torch.where(
                row[..., token_id] > 0,
                row[..., token_id] / penalty,
                row[..., token_id] * penalty,
            )

    return logits
