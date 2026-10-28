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
    if penalty <= 0.0:
        raise ValueError(f"penalty must be positive, got {penalty}")
    if penalty == 1.0:
        return logits

    if not isinstance(generated_tokens, torch.Tensor) and isinstance(generated_tokens, (list, tuple)):
        try:
            generated_tokens = torch.tensor(generated_tokens, dtype=torch.long)
        except Exception:
            pass

    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor) and generated_tokens.dim() > 1:
        rows = generated_tokens.reshape(generated_tokens.size(0), -1)
        for row_idx, row in enumerate(rows):
            for token_id in torch.unique(row).tolist():
                if 0 <= token_id < logits.size(-1):
                    value = logits[row_idx, token_id]
                    logits[row_idx, token_id] = value * penalty if value < 0 else value / penalty
    else:
        unique_tokens = torch.unique(generated_tokens).tolist() if isinstance(generated_tokens, torch.Tensor) else set(generated_tokens)
        for token_id in unique_tokens:
            if 0 <= token_id < logits.size(-1):
                value = logits[..., token_id]
                logits[..., token_id] = torch.where(value < 0, value * penalty, value / penalty)

    return logits
