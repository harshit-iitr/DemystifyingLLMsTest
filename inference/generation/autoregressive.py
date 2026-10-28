"""Module implementation."""
import torch
import inspect
from typing import Optional, List, Callable
from .kv_cache import KVCache
from .stop_criteria import StopCriteria


@torch.no_grad()
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
        raise ValueError("input_ids must have shape (batch_size, non-empty sequence)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    generated = input_ids.clone()
    batch_size = generated.size(0)
    stop_criteria = StopCriteria(max_length=generated.size(1) + max_new_tokens, eos_token_id=eos_token_id)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

    kv_cache = KVCache(num_layers=getattr(model, "num_layers", 12)) if use_cache else None
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        attention_mask = attention_mask.to(device=input_ids.device)

    try:
        forward_parameters = inspect.signature(model.forward).parameters
    except (AttributeError, TypeError, ValueError):
        forward_parameters = {}

    for _ in range(max_new_tokens):
        if kv_cache is not None:
            model_inputs = generated if kv_cache.get_seq_length(0) == 0 else generated[:, -1:]
            kwargs = {"kv_cache": kv_cache}
            if "position_ids" in forward_parameters:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(dim=-1) - 1
                    position_ids = position_ids.clamp_min(0)[:, -model_inputs.size(1):]
                else:
                    start_pos = kv_cache.get_seq_length(0)
                    position_ids = torch.arange(
                        start_pos,
                        start_pos + model_inputs.size(1),
                        dtype=torch.long,
                        device=input_ids.device,
                    ).unsqueeze(0).expand(batch_size, -1)
                kwargs["position_ids"] = position_ids
            if attention_mask is not None and "attention_mask" in forward_parameters:
                kwargs["attention_mask"] = attention_mask
            logits = model(model_inputs, **kwargs)
        else:
            kwargs = {}
            if attention_mask is not None and "attention_mask" in forward_parameters:
                kwargs["attention_mask"] = attention_mask
            logits = model(generated, **kwargs)

        # Take logits for the last token position: (batch_size, vocab_size)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_token_logits = logits[:, -1, :]

        if sampler_fn is not None:
            next_token = sampler_fn(next_token_logits)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        next_token = next_token.reshape(batch_size, 1)
        if eos_token_id is not None:
            next_token = torch.where(finished[:, None], torch.full_like(next_token, eos_token_id), next_token)
            finished |= next_token[:, 0].eq(eos_token_id)

        # generated = torch.cat([generated, next_token], dim=-1) is correct
        generated = torch.cat([generated, next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                (attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)),
                dim=-1,
            )

        if stop_criteria.should_stop(generated.size(1)) or (eos_token_id is not None and bool(finished.all())):
            break

    return generated
