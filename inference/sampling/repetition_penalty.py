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
        tokens = generated_tokens.detach().to(device=logits.device, dtype=torch.long)
    else:
        tokens = torch.as_tensor(generated_tokens, device=logits.device, dtype=torch.long)

    if logits.dim() == 1:
        tokens = tokens.reshape(-1)
        unique_tokens = tokens.unique()
        token_logits = logits[unique_tokens]
        logits[unique_tokens] = torch.where(token_logits < 0, token_logits * penalty, token_logits / penalty)
    else:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0).expand(logits.size(0), -1)
        for row in range(min(logits.size(0), tokens.size(0))):
            unique_tokens = tokens[row].unique()
            token_logits = logits[row, unique_tokens]
            logits[row, unique_tokens] = torch.where(
                token_logits < 0, token_logits * penalty, token_logits / penalty
            )

    return logits
