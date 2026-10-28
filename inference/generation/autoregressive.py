"""Autoregressive token generation."""
from typing import Optional, Callable
import inspect
import torch
from .kv_cache import KVCache
from .stop_criteria import StopCriteria


def autoregressive_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    eos_token_id: Optional[int] = None,
    sampler_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_cache: bool = False,
) -> torch.Tensor:
    """Generate up to ``max_new_tokens`` tokens for each prompt in a batch."""
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape (batch_size, seq_len)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    generated = input_ids.clone()
    if max_new_tokens == 0:
        return generated

    batch_size = generated.size(0)
    stop_criteria = StopCriteria(
        max_length=generated.size(1) + max_new_tokens,
        eos_token_id=eos_token_id,
    )
    supports_cache = False
    if use_cache:
        try:
            parameters = inspect.signature(model.forward).parameters.values()
            supports_cache = any(
                parameter.name == "kv_cache" or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (AttributeError, TypeError, ValueError):
            supports_cache = False
    kv_cache = KVCache(num_layers=getattr(model, "num_layers", 12)) if use_cache and supports_cache else None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

    with torch.no_grad():
        for step in range(max_new_tokens):
            model_inputs = generated[:, -1:] if kv_cache is not None and step > 0 else generated
            logits = model(model_inputs, kv_cache=kv_cache) if kv_cache is not None else model(generated)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_token_logits = logits[:, -1, :]

            if sampler_fn is None:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token = sampler_fn(next_token_logits)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)

            if eos_token_id is not None:
                eos_fill = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished.unsqueeze(-1), eos_fill, next_token)
                finished |= next_token.squeeze(-1).eq(eos_token_id)

            generated = torch.cat([generated, next_token], dim=-1)
            if stop_criteria.should_stop(generated.size(1), next_token.squeeze(-1)):
                break

    return generated
