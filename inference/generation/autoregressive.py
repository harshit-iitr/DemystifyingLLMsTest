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
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate tokens autoregressively.
    input_ids shape: (batch_size, seq_len)
    """
    if input_ids.dim() != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must have shape (batch_size, nonzero_seq_len)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    has_attention_mask = attention_mask is not None
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")

    model.eval()
    generated = input_ids.clone()
    batch_size = generated.size(0)
    stop_criteria = StopCriteria(max_length=generated.size(1) + max_new_tokens, eos_token_id=eos_token_id)
    if attention_mask is None:
        attention_mask = torch.ones_like(generated)
    else:
        attention_mask = attention_mask.to(device=generated.device, dtype=torch.long)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)
    if eos_token_id is not None:
        finished = generated[:, -1].eq(eos_token_id)
        if finished.all():
            return generated

    kv_cache = KVCache(num_layers=getattr(model, "num_layers", 12)) if use_cache else None

    for _ in range(max_new_tokens):
        if kv_cache is not None:
            model_inputs = generated if kv_cache.get_seq_length() == 0 else generated[:, -1:]
            model_kwargs = {"kv_cache": kv_cache}
            if has_attention_mask:
                model_kwargs["attention_mask"] = attention_mask
            logits = model(model_inputs, **model_kwargs)
        else:
            logits = model(generated, attention_mask=attention_mask) if has_attention_mask else model(generated)

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
        if next_token.shape != (batch_size, 1):
            raise ValueError("sampler_fn must return one token per batch row")

        active_before = ~finished
        if eos_token_id is not None:
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            finished |= next_token[:, 0].eq(eos_token_id)

        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat((attention_mask, active_before.unsqueeze(-1).long()), dim=-1)

        if (eos_token_id is not None and finished.all()) or generated.size(1) >= stop_criteria.max_length:
            break

    return generated
