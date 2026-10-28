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
    if logits.dim() == 1:
        histories = [generated_tokens]
        rows = logits.unsqueeze(0)
        squeeze = True
    else:
        rows = logits
        squeeze = False
        if isinstance(generated_tokens, torch.Tensor):
            histories = generated_tokens.tolist()
        else:
            histories = generated_tokens
        if not histories or not isinstance(histories[0], (list, tuple, torch.Tensor)):
            histories = [histories] * rows.size(0)
        if len(histories) != rows.size(0):
            raise ValueError("generated_tokens must provide one history per logits batch row")

    for row, history in zip(rows, histories):
        if isinstance(history, torch.Tensor):
            history = history.flatten().tolist()
        for token_id in set(history):
            token_id = int(token_id)
            if token_id < 0 or token_id >= row.numel():
                raise ValueError(f"generated token ID {token_id} is outside the logits vocabulary")
            value = row[token_id]
            row[token_id] = torch.where(value < 0, value * penalty, value / penalty)

    return rows.squeeze(0) if squeeze else rows
