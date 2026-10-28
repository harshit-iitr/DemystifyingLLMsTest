"""Module implementation."""
import torch
import inspect
from typing import Optional, Callable, Sequence, Union
from .kv_cache import KVCache
from .stop_criteria import StopCriteria


def autoregressive_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    eos_token_id: Optional[Union[int, Sequence[int]]] = None,
    sampler_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    sampler_with_history: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    use_cache: bool = False,
) -> torch.Tensor:
    """
    Generate tokens autoregressively.
    input_ids shape: (batch_size, seq_len)
    """
    if input_ids.dim() != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must have shape (batch_size, non-empty sequence length)")
    if input_ids.size(0) == 0:
        raise ValueError("input_ids must contain at least one sequence")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"input_ids must use an integer dtype, got {input_ids.dtype}")
    if torch.any(input_ids < 0):
        raise ValueError("input_ids must contain non-negative token IDs")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    eos_ids = (list(eos_token_id) if isinstance(eos_token_id, (list, tuple, set))
               else ([] if eos_token_id is None else [eos_token_id]))
    if any(not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
           for token_id in eos_ids):
        raise ValueError("eos_token_id values must be non-negative integers")
    generated = input_ids.clone()
    batch_size = generated.size(0)
    stop_criteria = StopCriteria(max_length=generated.size(1) + max_new_tokens, eos_token_id=eos_token_id)
    stop_ids = set(eos_ids)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

    supports_cache = False
    if use_cache:
        forward = model.forward if hasattr(model, "forward") else model
        try:
            parameters = inspect.signature(forward).parameters.values()
            supports_cache = any(
                parameter.name == "kv_cache"
                for parameter in parameters
            ) or bool(getattr(model, "supports_kv_cache", False))
        except (TypeError, ValueError):
            supports_cache = bool(getattr(model, "supports_kv_cache", False))
    kv_cache = (
        KVCache(num_layers=getattr(model, "num_layers", 12))
        if supports_cache else None
    )

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if kv_cache is not None:
                model_inputs = generated if kv_cache.get_seq_length() == 0 else generated[:, -1:]
                logits = model(model_inputs, kv_cache=kv_cache)
            else:
                logits = model(generated)

            # Take logits for the last token position: (batch_size, vocab_size)
            if isinstance(logits, tuple):
                logits = logits[0]
            if not isinstance(logits, torch.Tensor) or logits.dim() != 3:
                raise ValueError("model must return logits with shape (batch_size, seq_len, vocab_size)")
            vocab_size = logits.size(-1)
            if torch.any(generated >= vocab_size):
                raise ValueError(f"input token IDs must be smaller than model vocabulary size {vocab_size}")
            if any(token_id >= vocab_size for token_id in stop_ids):
                raise ValueError(f"stop token IDs must be smaller than model vocabulary size {vocab_size}")
            next_token_logits = logits[:, -1, :]

            if sampler_with_history is not None:
                next_token = sampler_with_history(next_token_logits, generated)
            elif sampler_fn is not None:
                next_token = sampler_fn(next_token_logits)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token.dim() == 1:
                next_token = next_token.unsqueeze(-1)

            if stop_ids:
                if finished.any():
                    next_token[finished] = min(stop_ids)
                finished |= torch.stack(
                    [next_token.squeeze(-1) == token_id for token_id in stop_ids], dim=0
                ).any(dim=0)

            generated = torch.cat([generated, next_token], dim=-1)

            if stop_criteria.should_stop(generated.size(1)) or (stop_ids and finished.all()):
                break

    return generated
