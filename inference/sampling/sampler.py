"""Combined repetition, temperature, top-k and nucleus sampling."""
import torch
from typing import Optional, List, Union
from .temperature import apply_temperature
from .top_k import apply_top_k
from .top_p import apply_top_p
from .repetition_penalty import apply_repetition_penalty


class Sampler:
    def __init__(self, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0):
        from config.inference_config import InferenceConfig
        InferenceConfig(temperature=temperature, top_k=top_k, top_p=top_p,
                        repetition_penalty=repetition_penalty).validate()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def sample(self, logits: torch.Tensor, generated_tokens: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if generated_tokens is not None:
            logits = apply_repetition_penalty(logits, generated_tokens, self.repetition_penalty)
        if self.temperature == 0:
            return logits.argmax(dim=-1)
        # Stable probability computation for fp16/bfloat16 model outputs.
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        logits = apply_temperature(logits, self.temperature)
        logits = apply_top_k(logits, self.top_k)
        logits = apply_top_p(logits, self.top_p)
        return torch.multinomial(logits.softmax(dim=-1), 1).squeeze(-1)
