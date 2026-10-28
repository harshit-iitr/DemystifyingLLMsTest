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
        raise ValueError("repetition penalty must be positive")
    if penalty == 1.0:
        return logits
    if logits.dim() not in (1, 2):
        raise ValueError("logits must have shape (vocab_size,) or (batch_size, vocab_size)")

    logits = logits.clone()
    batched_logits = logits.unsqueeze(0) if logits.dim() == 1 else logits
    if isinstance(generated_tokens, torch.Tensor):
        if generated_tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError("generated_tokens tensor must use an integer dtype")
        if generated_tokens.dim() > 2:
            raise ValueError("generated_tokens tensor must have one or two dimensions")
        if generated_tokens.dim() <= 1:
            token_batches = [torch.unique(generated_tokens).tolist()] * batched_logits.size(0)
        else:
            token_batches = [torch.unique(row).tolist() for row in generated_tokens]
    else:
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in generated_tokens):
            raise TypeError("generated_tokens must contain integer token IDs")
        unique_tokens = list(set(generated_tokens))
        token_batches = [unique_tokens] * batched_logits.size(0)

    if len(token_batches) != batched_logits.size(0):
        raise ValueError("generated token batch size must match logits batch size")

    for batch_idx, token_ids in enumerate(token_batches):
        for token_id in token_ids:
            if not 0 <= token_id < logits.size(-1):
                raise ValueError(
                    f"generated token ID {token_id} is outside logits vocabulary "
                    f"[0, {logits.size(-1)})"
                )
            token_logits = batched_logits[batch_idx, token_id]
            batched_logits[batch_idx, token_id] = torch.where(
                token_logits < 0, token_logits * penalty, token_logits / penalty
            )

    return logits
