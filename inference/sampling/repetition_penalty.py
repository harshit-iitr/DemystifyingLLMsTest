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
    Apply the standard sign-aware repetition penalty to previously generated IDs.
    """
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    if isinstance(generated_tokens, torch.Tensor):
        unique_tokens = torch.unique(generated_tokens).tolist()
    else:
        unique_tokens = list(set(generated_tokens))

    vocab_size = logits.size(-1)
    valid_tokens = [int(token_id) for token_id in unique_tokens if 0 <= int(token_id) < vocab_size]
    if valid_tokens:
        token_ids = torch.tensor(valid_tokens, device=logits.device, dtype=torch.long)
        selected = logits.index_select(-1, token_ids)
        adjusted = torch.where(selected < 0, selected * penalty, selected / penalty)
        logits.index_copy_(-1, token_ids, adjusted)

    return logits
