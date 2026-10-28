"""Sign-aware repetition penalty, independently for each sequence."""
import torch
from typing import List, Union


def apply_repetition_penalty(logits: torch.Tensor,
                             generated_tokens: Union[List[int], torch.Tensor],
                             penalty: float = 1.2) -> torch.Tensor:
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    if penalty == 1.0:
        return logits
    result = logits.clone()
    tokens = torch.as_tensor(generated_tokens, device=logits.device, dtype=torch.long)
    if tokens.ndim > 1:
        if logits.ndim != 2 or tokens.size(0) != logits.size(0):
            raise ValueError("batched token history must match logits batch")
        for row in range(logits.size(0)):
            result[row] = apply_repetition_penalty(result[row], tokens[row], penalty)
    else:
        ids = tokens.unique()
        values = result[..., ids]
        result[..., ids] = torch.where(values < 0, values * penalty, values / penalty)
    return result
