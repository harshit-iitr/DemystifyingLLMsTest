"""Repetition penalty."""
import math
import torch
from typing import List, Union


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Union[List[int], torch.Tensor],
    penalty: float = 1.2
) -> torch.Tensor:
    """
    Apply repetition penalty to logits for generated tokens.
    Divide positive logits and multiply negative logits, as in standard
    autoregressive repetition penalty decoding.
    """
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError(f"penalty must be positive, got {penalty}")
    if penalty == 1.0:
        return logits

    original_shape = logits.shape
    flat_logits = logits.reshape(-1, logits.size(-1)).clone()

    if isinstance(generated_tokens, torch.Tensor) and generated_tokens.dim() >= 2:
        token_rows = generated_tokens.reshape(generated_tokens.size(0), -1).tolist()
        if len(token_rows) == 1 and len(flat_logits) > 1:
            token_rows *= len(flat_logits)
        elif len(token_rows) != len(flat_logits):
            raise ValueError("generated_tokens batch size must match logits batch size")
    elif isinstance(generated_tokens, torch.Tensor):
        token_rows = [generated_tokens.reshape(-1).tolist()] * len(flat_logits)
    else:
        token_rows = [list(generated_tokens)] * len(flat_logits)

    for row, token_ids in zip(flat_logits, token_rows):
        for token_id in set(int(token_id) for token_id in token_ids):
            if 0 <= token_id < row.numel():
                row[token_id] = row[token_id] / penalty if row[token_id] > 0 else row[token_id] * penalty

    return flat_logits.view(original_shape)
