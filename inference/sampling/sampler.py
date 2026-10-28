"""Combined sampler (Clean)."""
import torch
import torch.nn.functional as F
from typing import Optional, List, Union
import math
from .temperature import apply_temperature
from .top_k import apply_top_k
from .top_p import apply_top_p
from .repetition_penalty import apply_repetition_penalty


class Sampler:
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.generator = generator
        if temperature < 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not math.isfinite(top_p) or not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be finite and in [0.0, 1.0]")
        if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive")

    def sample(
        self,
        logits: torch.Tensor,
        generated_tokens: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Sample next token from logits tensor of shape (batch_size, vocab_size) or (vocab_size,).
        """
        logits = logits.clone()
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # 1. Repetition penalty
        if self.repetition_penalty != 1.0 and generated_tokens is not None:
            logits = apply_repetition_penalty(logits, generated_tokens, penalty=self.repetition_penalty)

        # Temperature changes the probability mass used by nucleus filtering.
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1)
        if self.temperature != 1.0:
            logits = apply_temperature(logits, temperature=self.temperature)

        # 2. Top-K
        if self.top_k > 0:
            logits = apply_top_k(logits, k=self.top_k)

        # 3. Top-P
        if self.top_p < 1.0:
            logits = apply_top_p(logits, p=self.top_p)

        # 4. Normalize the filtered, temperature-scaled logits.
        probs = F.softmax(logits, dim=-1)

        # 5. Multinomial sample
        next_token = torch.multinomial(probs, num_samples=1, generator=self.generator)
        return next_token.squeeze(-1)
