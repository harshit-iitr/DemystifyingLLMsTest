"""Module implementation."""
import torch
from typing import Optional, List, Callable
from .kv_cache import KVCache
from .stop_criteria import StopCriteria


@torch.inference_mode()
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
    generated = input_ids.clone()
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if generated.dim() != 2 or generated.size(1) == 0:
        raise ValueError("input_ids must have shape (batch_size, non-empty sequence_length)")
    batch_size = generated.size(0)
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape or not attention_mask.bool().any(dim=-1).all():
            raise ValueError("attention_mask must match input_ids and contain a token in each row")
        attention_mask = attention_mask.to(input_ids.device).clone()
    if hasattr(model, "eval"):
        model.eval()
    stop_criteria = StopCriteria(max_length=generated.size(1) + max_new_tokens, eos_token_id=eos_token_id)

    kv_cache = KVCache(num_layers=getattr(model, "num_layers", 12)) if use_cache else None

    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)
    for step in range(max_new_tokens):
        model_kwargs = {} if attention_mask is None else {"attention_mask": attention_mask}
        if kv_cache is not None:
            model_inputs = generated if step == 0 else generated[:, -1:]
            logits = model(model_inputs, kv_cache=kv_cache, **model_kwargs)
        else:
            logits = model(generated, **model_kwargs)

        # Take logits for the last token position: (batch_size, vocab_size)
        if isinstance(logits, tuple):
            logits = logits[0]
        if attention_mask is not None and (kv_cache is None or step == 0):
            positions = torch.arange(attention_mask.size(1), device=generated.device)
            last_positions = positions.expand_as(attention_mask).masked_fill(attention_mask.eq(0), -1).max(-1).values
            next_token_logits = logits[torch.arange(batch_size, device=generated.device), last_positions]
        else:
            next_token_logits = logits[:, -1, :]

        if sampler_fn is not None:
            next_token = sampler_fn(next_token_logits)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token.dim() == 1:
            next_token = next_token.unsqueeze(-1)
        elif next_token.dim() == 0 and batch_size == 1:
            next_token = next_token.view(1, 1)
        if next_token.shape != (batch_size, 1):
            raise ValueError("sampler_fn must return one token ID per batch item")
        if eos_token_id is not None:
            next_token = torch.where(finished.unsqueeze(-1), eos_token_id, next_token)

        generated = torch.cat([generated, next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)

        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)
        if stop_criteria.should_stop(generated.size(1)) or (eos_token_id is not None and finished.all()):
            break

    return generated
