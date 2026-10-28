"""Autoregressive text generation with optional KV caching."""

import inspect
from typing import Optional, Callable

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
    """
    Generate tokens autoregressively.

    Args:
        model: Transformer model.
        input_ids: Input token IDs of shape (batch_size, seq_len).
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_id: Optional EOS token ID.
        sampler_fn: Optional function mapping logits to next-token IDs.
        use_cache: Whether to reuse cached keys and values.

    Returns:
        Generated token IDs, including the original input.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch_size, seq_len)")
    if input_ids.size(1) == 0:
        raise ValueError("input_ids must contain at least one token")
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

    if use_cache:
        config = getattr(model, "config", None)
        num_layers = getattr(
            model,
            "num_layers",
            getattr(config, "n_layers", 12),
        )
        kv_cache = KVCache(num_layers=num_layers)

        # Determine whether the model accepts an offset argument.
        try:
            parameters = inspect.signature(model.forward).parameters
            supports_offset = (
                "offset" in parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            )
        except (TypeError, ValueError):
            supports_offset = False
    else:
        kv_cache = None
        supports_offset = False

    for _ in range(max_new_tokens):
        if kv_cache is None:
            model_inputs = generated
            logits = model(model_inputs)
        else:
            past_length = kv_cache.get_seq_length(0)

            # First iteration: process the complete prompt.
            # Later iterations: process only the newest token.
            if past_length == 0:
                model_inputs = generated
            else:
                model_inputs = generated[:, -1:]

            model_kwargs = {"kv_cache": kv_cache}

            if supports_offset:
                model_kwargs["offset"] = past_length

            logits = model(model_inputs, **model_kwargs)

        if isinstance(logits, tuple):
            logits = logits[0]

        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError(
                "Model must return logits with shape "
                "(batch_size, seq_len, vocab_size)"
            )

        next_token_logits = logits[:, -1, :]

        if sampler_fn is not None:
            next_token = sampler_fn(next_token_logits)
        else:
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

        if not isinstance(next_token, torch.Tensor):
            next_token = torch.as_tensor(
                next_token,
                device=generated.device,
            )

        if next_token.ndim == 0:
            next_token = next_token.reshape(1, 1)
        elif next_token.ndim == 1:
            next_token = next_token.unsqueeze(-1)

        if next_token.shape != (batch_size, 1):
            raise ValueError(
                "sampler_fn must return one token ID per batch element"
            )

        next_token = next_token.to(
            device=generated.device,
            dtype=generated.dtype,
        )

        generated = torch.cat([generated, next_token], dim=-1)

        last_id = next_token[0, 0].item() if batch_size == 1 else None

        if stop_criteria.should_stop(generated.size(1), last_id):
            break

    return generated