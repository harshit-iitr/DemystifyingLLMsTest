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

    was_1d = logits.dim() == 1
    if was_1d:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        raise ValueError("logits must have shape (vocab_size,) or (batch_size, vocab_size)")
    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor) and generated_tokens.dim() > 1:
        token_rows = generated_tokens
    elif isinstance(generated_tokens, torch.Tensor):
        token_rows = generated_tokens.unsqueeze(0)
    elif generated_tokens and isinstance(generated_tokens[0], (list, tuple)):
        token_rows = generated_tokens
    else:
        token_rows = [generated_tokens] * logits.size(0)
    if len(token_rows) not in (1, logits.size(0)):
        raise ValueError("generated_tokens batch dimension must match logits")

    for batch_idx in range(logits.size(0)):
        row = token_rows[batch_idx] if len(token_rows) > 1 else token_rows[0]
        token_ids = torch.as_tensor(row, dtype=torch.long, device=logits.device).unique()
        token_ids = token_ids[(token_ids >= 0) & (token_ids < logits.size(-1))]
        token_logits = logits[batch_idx, token_ids]
        logits[batch_idx, token_ids] = torch.where(
            token_logits < 0, token_logits * penalty, token_logits / penalty
        )
    return logits.squeeze(0) if was_1d else logits
