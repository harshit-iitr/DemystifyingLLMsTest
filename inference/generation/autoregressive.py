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
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        shape = tuple(input_ids.shape) if isinstance(input_ids, torch.Tensor) else type(input_ids).__name__
        raise ValueError(f"input_ids must be a 2D tensor, got {shape}")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be a non-negative integer, got {max_new_tokens!r}")

    generated = input_ids.clone()
    if max_new_tokens > 0 and generated.size(1) == 0:
        raise ValueError("input_ids must contain at least one token when generating")
    batch_size = generated.size(0)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)
    stop_criteria = StopCriteria(max_length=generated.size(1) + max_new_tokens, eos_token_id=eos_token_id)

    kv_cache = KVCache(num_layers=getattr(model, "num_layers", len(getattr(model, "blocks", [])) or 12)) if use_cache else None

    for _ in range(max_new_tokens):
        if kv_cache is not None:
            cached_length = kv_cache.get_seq_length()
            model_inputs = generated if cached_length == 0 else generated[:, -1:]
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
        elif next_token.dim() == 0 and batch_size == 1:
            next_token = next_token.reshape(1, 1)
        if next_token.shape != (batch_size, 1):
            raise ValueError(
                f"sampler_fn must return one token per batch row with shape "
                f"({batch_size}, 1), got {tuple(next_token.shape)}"
            )

        if eos_token_id is not None:
            # Keep completed rows at EOS while the remaining batch continues.
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            finished |= next_token.squeeze(-1) == eos_token_id

        generated = torch.cat([generated, next_token], dim=-1)

        # A finished sequence in a batch must not stop generation for every
        # other sequence. Stop on EOS only when all batch rows have finished.
        if generated.size(1) >= stop_criteria.max_length:
            break
        if eos_token_id is not None and torch.all(finished):
            break

    return generated
