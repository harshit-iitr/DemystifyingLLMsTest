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
    Divide positive logits and multiply negative logits for repeated tokens.
    """
    if penalty <= 0:
        raise ValueError(f"penalty must be positive, got {penalty}")
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    tokens = torch.as_tensor(generated_tokens, dtype=torch.long, device=logits.device)
    if tokens.numel() == 0:
        return logits
    if logits.dim() == 1:
        tokens = tokens.reshape(-1)
        tokens = tokens[(tokens >= 0) & (tokens < logits.size(-1))].unique()
        values = logits[tokens]
        logits[tokens] = torch.where(values < 0, values * penalty, values / penalty)
    else:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0).expand(logits.size(0), -1)
        if tokens.size(0) != logits.size(0):
            raise ValueError("generated_tokens batch dimension must match logits")
        for row in range(logits.size(0)):
            row_tokens = tokens[row]
            row_tokens = row_tokens[(row_tokens >= 0) & (row_tokens < logits.size(-1))].unique()
            values = logits[row, row_tokens]
            logits[row, row_tokens] = torch.where(values < 0, values * penalty, values / penalty)

    return logits
