
"""Repetition penalty."""
import torch
from typing import List, Union


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Union[List[int], torch.Tensor],
    penalty: float = 1.2,
) -> torch.Tensor:
    """
    Apply repetition penalty to tokens already present in the sequence.

    Positive logits are divided by the penalty.
    Negative logits are multiplied by the penalty.

    Args:
        logits: Logits with vocabulary dimension as the last dimension.
        generated_tokens: Previously generated token IDs.
        penalty: Repetition penalty; must be strictly positive.

    Returns:
        A new tensor containing the adjusted logits.
    """
    if penalty <= 0:
        raise ValueError("penalty must be strictly positive.")

    if penalty == 1.0 or generated_tokens is None:
        return logits

    adjusted_logits = logits.clone()

    if isinstance(generated_tokens, torch.Tensor):
        token_ids = generated_tokens.detach().reshape(-1).unique().tolist()
    else:
        token_ids = set(generated_tokens)

    vocab_size = adjusted_logits.size(-1)

    for token_id in token_ids:
        token_id = int(token_id)

        if not 0 <= token_id < vocab_size:
            raise ValueError(
                f"Token ID {token_id} is outside vocabulary range "
                f"[0, {vocab_size})."
            )

        token_logits = adjusted_logits[..., token_id]

        adjusted_logits[..., token_id] = torch.where(
            token_logits < 0,
            token_logits * penalty,
            token_logits / penalty,
        )

    return adjusted_logits
