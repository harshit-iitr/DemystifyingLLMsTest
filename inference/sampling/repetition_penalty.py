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
    if not torch.isfinite(torch.tensor(penalty)) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")
    if penalty == 1.0:
        return logits.clone()
    if generated_tokens is None:
        return logits.clone()

    logits = logits.clone()
    batch_size = logits.size(0) if logits.ndim > 1 else 1
    if isinstance(generated_tokens, torch.Tensor):
        token_data = generated_tokens.detach().cpu()
        if token_data.ndim <= 1:
            token_rows = [token_data.reshape(-1).tolist()] * batch_size
        elif token_data.size(0) == batch_size:
            token_rows = [row.reshape(-1).tolist() for row in token_data]
        else:
            token_rows = [token_data.reshape(-1).tolist()] * batch_size
    elif generated_tokens and isinstance(generated_tokens[0], (list, tuple)):
        rows = [list(row) for row in generated_tokens]
        all_tokens = [token for row in rows for token in row]
        token_rows = rows if len(rows) == batch_size else [all_tokens] * batch_size
    else:
        token_rows = [list(generated_tokens)] * batch_size

    logits = logits.contiguous()
    logits_by_batch = logits.reshape(batch_size, -1, logits.size(-1))
    if len(token_rows) != batch_size:
        raise ValueError("generated_tokens batch dimension must match logits")

    for rows, token_ids in zip(logits_by_batch, token_rows):
        for token_id in set(int(token) for token in token_ids):
            if 0 <= token_id < rows.size(-1):
                values = rows[..., token_id]
                rows[..., token_id] = torch.where(values < 0, values * penalty, values / penalty)

    return logits

