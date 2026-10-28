"""Repetition penalty."""
import torch
from typing import List, Union


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Union[List[int], torch.Tensor],
    penalty: float = 1.2
) -> torch.Tensor:
    """
    Apply the standard sign-aware repetition penalty to previously generated tokens.
    """
    if penalty <= 0.0:
        raise ValueError(f"penalty must be positive, got {penalty}")
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor):
        generated_tokens = generated_tokens.to(device=logits.device, dtype=torch.long)
        if logits.dim() > 1 and generated_tokens.dim() > 1 and generated_tokens.size(0) == logits.size(0):
            rows = generated_tokens.reshape(generated_tokens.size(0), -1)
            for row_idx, token_ids in enumerate(rows):
                token_ids = token_ids.unique()
                row_logits = logits[row_idx]
                positive = row_logits[token_ids] > 0
                row_logits[token_ids] = torch.where(
                    positive,
                    row_logits[token_ids] / penalty,
                    row_logits[token_ids] * penalty,
                )
        else:
            unique_tokens = generated_tokens.unique()
            positive = logits[..., unique_tokens] > 0
            logits[..., unique_tokens] = torch.where(
                positive,
                logits[..., unique_tokens] / penalty,
                logits[..., unique_tokens] * penalty,
            )
    else:
        unique_tokens = list(set(generated_tokens))
        positive = logits[..., unique_tokens] > 0
        logits[..., unique_tokens] = torch.where(
            positive,
            logits[..., unique_tokens] / penalty,
            logits[..., unique_tokens] * penalty,
        )

    return logits
