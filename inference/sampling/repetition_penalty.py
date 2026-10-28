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

    def penalize(token_logits: torch.Tensor) -> torch.Tensor:
        return torch.where(
            token_logits < 0,
            token_logits * penalty,
            token_logits / penalty,
        )

    # A batched history has one sequence per logits row. Penalizing the union
    # of all rows would incorrectly make one request affect another request.
    if (
        isinstance(generated_tokens, torch.Tensor)
        and logits.dim() == 2
        and generated_tokens.dim() >= 2
        and generated_tokens.size(0) == logits.size(0)
    ):
        for batch_idx, history in enumerate(generated_tokens):
            for token_id in torch.unique(history).tolist():
                logits[batch_idx, token_id] = penalize(logits[batch_idx, token_id])
    else:
        if isinstance(generated_tokens, torch.Tensor):
            unique_tokens = torch.unique(generated_tokens).tolist()
        else:
            unique_tokens = list(set(generated_tokens))
        for token_id in unique_tokens:
            logits[..., token_id] = penalize(logits[..., token_id])

    return logits
