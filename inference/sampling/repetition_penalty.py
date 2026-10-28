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
    if isinstance(generated_tokens, torch.Tensor):
        histories = generated_tokens
        if histories.ndim == 1:
            histories = histories.unsqueeze(0)
        histories = histories.tolist()
    else:
        histories = [generated_tokens]

    if logits.size(0) != len(histories) and len(histories) != 1:
        raise ValueError("Number of generated-token histories must match logits batch size")

    for batch_idx in range(logits.size(0)):
        history = histories[batch_idx] if len(histories) > 1 else histories[0]
        for token_id in set(history):
            if token_id < 0 or token_id >= logits.size(-1):
                raise ValueError(f"Generated token ID {token_id} is outside logits vocabulary range")
            token_logit = logits[batch_idx, ..., token_id]
            logits[batch_idx, ..., token_id] = torch.where(
                token_logit < 0, token_logit * penalty, token_logit / penalty
            )

    return logits
