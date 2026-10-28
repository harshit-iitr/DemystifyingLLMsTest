"""Module implementation."""
import torch
import inspect
from typing import Optional, List, Callable
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
    """
    Generate tokens autoregressively.
    input_ids shape: (batch_size, seq_len)
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    generated = input_ids.clone()
    batch_size = generated.size(0)
    stop_criteria = StopCriteria(max_length=generated.size(1) + max_new_tokens, eos_token_id=eos_token_id)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

    kv_cache = KVCache(num_layers=getattr(model, "num_layers", 12)) if use_cache else None
    supports_offset = False
    if kv_cache is not None:
        parameters = inspect.signature(model.forward).parameters.values()
        supports_offset = any(
            parameter.name == "offset" or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    for _ in range(max_new_tokens):
        if kv_cache is not None:
            cached_length = kv_cache.get_seq_length()
            model_inputs = generated[:, cached_length:] if cached_length else generated
            cache_kwargs = (
                {"kv_cache": kv_cache, "offset": cached_length}
                if supports_offset
                else {"kv_cache": kv_cache}
            )
            logits = model(model_inputs, **cache_kwargs)
        else:
            logits = model(generated)

        # Take logits for the last token position: (batch_size, vocab_size)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_token_logits = logits[:, -1, :]

        if sampler_fn is not None:
            next_token = sampler_fn(next_token_logits)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token.dim() == 1:
            next_token = next_token.unsqueeze(-1)

        if eos_token_id is not None:
            finished |= next_token.squeeze(-1) == eos_token_id
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, eos_token_id),
                next_token
            )
        generated = torch.cat([generated, next_token], dim=-1)

        if stop_criteria.should_stop(generated.size(1)):
            break
        if eos_token_id is not None and torch.all(finished):
            break

    return generated
