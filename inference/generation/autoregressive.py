"""Autoregressive generation with incremental KV caching."""
import torch
import inspect
from typing import Optional, Callable, Sequence
from .kv_cache import KVCache


@torch.no_grad()
def autoregressive_generate(model: torch.nn.Module, input_ids: torch.Tensor,
                            max_new_tokens: int = 20, eos_token_id: Optional[int] = None,
                            sampler_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                            use_cache: bool = False,
                            stop_token_ids: Optional[Sequence[int]] = None) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must be a nonempty (batch, sequence) tensor")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    generated = input_ids.clone()
    num_layers = getattr(model, 'num_layers', getattr(getattr(model, 'config', None), 'n_layers', 12))
    parameters = inspect.signature(getattr(model, "forward", model)).parameters.values()
    supports_cache = any(p.name == "kv_cache" for p in parameters)
    cache = KVCache(num_layers=num_layers) if use_cache and supports_cache else None
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    stops = set(stop_token_ids or [])
    if eos_token_id is not None:
        stops.add(eos_token_id)
    stop_values = torch.tensor(sorted(stops), device=input_ids.device, dtype=input_ids.dtype)
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        for step in range(max_new_tokens):
            if cache is not None:
                model_inputs = generated if step == 0 else generated[:, -1:]
                logits = model(model_inputs, kv_cache=cache)
            else:
                logits = model(generated)
            if isinstance(logits, tuple):
                logits = logits[0]
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise TypeError("Model must return (batch, sequence, vocabulary) logits")
            scores = logits[:, -1, :]
            token = sampler_fn(scores) if sampler_fn else scores.argmax(dim=-1)
            token = token.reshape(input_ids.size(0), 1)
            if stops:
                # Freeze completed rows at their terminal token while others finish.
                token = torch.where(finished[:, None], generated[:, -1:], token)
                finished |= torch.isin(token[:, 0], stop_values)
            generated = torch.cat([generated, token], dim=-1)
            if finished.all():
                break
    finally:
        if was_training is not None and hasattr(model, "train"):
            model.train(was_training)
    return generated
