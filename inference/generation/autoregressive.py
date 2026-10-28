"""Module implementation."""
import torch
from typing import Optional, List, Callable, Sequence, Union
from .kv_cache import KVCache
from .stop_criteria import StopCriteria


def autoregressive_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    eos_token_id: Optional[Union[int, Sequence[int]]] = None,
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

    kv_cache = KVCache(num_layers=getattr(model, "num_layers", 12)) if use_cache else None

    was_training = model.training
    model.eval()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)
    try:
        with torch.inference_mode():
            for step in range(max_new_tokens):
                if kv_cache is not None:
                    model_inputs = generated if step == 0 else generated[:, -1:]
                    logits = model(model_inputs, kv_cache=kv_cache)
                else:
                    logits = model(generated)

                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]

                if sampler_fn is not None:
                    next_token = sampler_fn(next_token_logits)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)

                # Keep already-finished rows at EOS while the remaining rows continue.
                eos_ids = (
                    list(eos_token_id)
                    if isinstance(eos_token_id, (list, tuple, set))
                    else [eos_token_id]
                ) if eos_token_id is not None else []
                if eos_ids:
                    next_token = torch.where(
                        finished.unsqueeze(-1),
                        torch.full_like(next_token, eos_ids[0]),
                        next_token,
                    )
                generated = torch.cat([generated, next_token], dim=-1)

                if eos_ids:
                    eos_tensor = torch.as_tensor(eos_ids, device=next_token.device)
                    finished |= torch.isin(next_token.squeeze(-1), eos_tensor)
                if generated.size(1) >= stop_criteria.max_length or (eos_ids and finished.all()):
                    break
    finally:
        model.train(was_training)

    return generated
