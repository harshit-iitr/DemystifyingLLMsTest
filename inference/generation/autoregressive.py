"""Module implementation."""
import torch
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
    generated = input_ids.clone()
    batch_size = generated.size(0)
    stop_criteria = StopCriteria(max_length=generated.size(1) + max_new_tokens, eos_token_id=eos_token_id)

    num_layers = getattr(model, "num_layers", None)
    if num_layers is None:
        num_layers = getattr(getattr(model, "config", None), "n_layers", None)
    if num_layers is None and hasattr(model, "blocks"):
        num_layers = len(model.blocks)
    kv_cache = KVCache(num_layers=num_layers or 12) if use_cache else None

    for _ in range(max_new_tokens):
        if kv_cache is not None:
            cached_len = kv_cache.get_seq_length(0)
            model_inputs = generated[:, cached_len:] if cached_len else generated
            logits = model(model_inputs, kv_cache=kv_cache)
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

        generated = torch.cat([generated, next_token], dim=-1)

        last_id = next_token[0].item() if batch_size == 1 else None
        if stop_criteria.should_stop(generated.size(1), last_id):
            break

    return generated
