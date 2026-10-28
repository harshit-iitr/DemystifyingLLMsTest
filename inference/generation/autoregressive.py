"""Module implementation."""
import torch
from typing import Optional, Callable, Union, Iterable
from .kv_cache import KVCache
from .stop_criteria import StopCriteria, normalize_eos_token_ids


def autoregressive_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    eos_token_id: Optional[Union[int, Iterable[int]]] = None,
    sampler_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_cache: bool = False,
) -> torch.Tensor:
    """
    Generate tokens autoregressively.
    input_ids shape: (batch_size, seq_len)
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must have shape (batch_size, seq_len), got {tuple(input_ids.shape)}")
    generated = input_ids.clone()
    batch_size = generated.size(0)
    if max_new_tokens > 0 and (batch_size == 0 or generated.size(1) == 0):
        raise ValueError("input_ids must contain at least one sequence and one token")
    eos_ids = normalize_eos_token_ids(eos_token_id)
    stop_criteria = StopCriteria(max_length=generated.size(1) + max_new_tokens, eos_token_id=eos_ids)
    eos_tensor = torch.tensor(list(eos_ids), dtype=torch.long, device=generated.device) if eos_ids else None
    finished = (
        torch.isin(generated[:, -1], eos_tensor)
        if eos_tensor is not None and generated.size(1) > 0
        else torch.zeros(batch_size, dtype=torch.bool, device=generated.device)
    )
    if max_new_tokens == 0 or (batch_size > 0 and finished.all()):
        return generated

    if use_cache:
        config = getattr(model, "config", None)
        num_layers = getattr(model, "num_layers", getattr(config, "n_layers", 12))
        kv_cache = KVCache(num_layers=num_layers)
    else:
        kv_cache = None

    for step in range(max_new_tokens):
        if kv_cache is not None:
            model_inputs = generated if step == 0 else generated[:, -1:]
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

        if eos_tensor is not None:
            just_finished = torch.isin(next_token.squeeze(-1), eos_tensor)
            if finished.any():
                # Keep completed rows at EOS while the remaining rows continue.
                next_token = torch.where(finished.unsqueeze(-1), generated[:, -1:], next_token)
                just_finished = just_finished | finished
            finished = finished | just_finished

        generated = torch.cat([generated, next_token], dim=-1)

        last_id = next_token[0, 0].item() if batch_size == 1 else None
        if stop_criteria.should_stop(generated.size(1), last_id):
            break
        if batch_size > 1 and finished.all():
            break

    return generated
