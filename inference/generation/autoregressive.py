"""Autoregressive token generation."""
from typing import Optional, Callable, Sequence, Union

import torch

from .kv_cache import KVCache


def autoregressive_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    eos_token_id: Optional[Union[int, Sequence[int]]] = None,
    sampler_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_cache: bool = False,
) -> torch.Tensor:
    """Generate tokens in inference mode, restoring the model's prior mode afterward."""
    was_training = model.training
    model.eval()
    generated = input_ids.clone()
    batch_size = generated.size(0)
    num_layers = getattr(model, "num_layers", getattr(getattr(model, "config", None), "n_layers", 12))
    kv_cache = KVCache(num_layers=num_layers) if use_cache else None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)
    stop_ids = () if eos_token_id is None else ((eos_token_id,) if isinstance(eos_token_id, int) else tuple(eos_token_id))
    stop_fill_id = stop_ids[0] if stop_ids else None

    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if kv_cache is not None:
                    model_inputs = generated if kv_cache.get_seq_length() == 0 else generated[:, -1:]
                    logits = model(model_inputs, kv_cache=kv_cache)
                else:
                    logits = model(generated)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_logits = logits[:, -1, :]

                if sampler_fn is None:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    next_token = sampler_fn(next_logits)
                    if next_token.dim() == 1:
                        next_token = next_token.unsqueeze(-1)

                if stop_ids:
                    next_token = torch.where(
                        finished[:, None],
                        torch.full_like(next_token, stop_fill_id),
                        next_token,
                    )
                    stop_tensor = torch.tensor(stop_ids, device=next_token.device, dtype=next_token.dtype)
                    finished |= torch.isin(next_token.squeeze(-1), stop_tensor)
                generated = torch.cat([generated, next_token], dim=-1)
                if stop_ids and bool(finished.all()):
                    break
    finally:
        if was_training:
            model.train()

    return generated
