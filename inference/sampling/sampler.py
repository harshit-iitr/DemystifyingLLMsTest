"""Combined sampler."""
import torch
import torch.nn.functional as F
from typing import Optional, List, Union

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
    ):
        if temperature <= 0:
            raise ValueError("temperature must be strictly positive.")
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1].")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be strictly positive.")

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def sample(
        self,
        logits: torch.Tensor,
        generated_tokens: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Sample a token from logits shaped (batch, vocab) or (vocab,).

        Returns:
            Token IDs shaped (batch,).
        """
        if logits.ndim not in (1, 2):
            raise ValueError(
                "logits must have shape (vocab_size,) or "
                "(batch_size, vocab_size)."
            )

        if not torch.isfinite(logits).any(dim=-1).all():
            raise ValueError("At least one valid logit is required per row.")

        was_1d = logits.ndim == 1
        filtered_logits = logits.clone()

        if was_1d:
            filtered_logits = filtered_logits.unsqueeze(0)

        if generated_tokens is not None and self.repetition_penalty != 1.0:
            filtered_logits = apply_repetition_penalty(
                filtered_logits,
                generated_tokens,
                penalty=self.repetition_penalty,
            )

        if self.top_k > 0:
            filtered_logits = apply_top_k(
                filtered_logits,
                k=self.top_k,
            )

        if self.top_p < 1.0:
            filtered_logits = apply_top_p(
                filtered_logits,
                p=self.top_p,
            )

        scaled_logits = apply_temperature(
            filtered_logits,
            temperature=self.temperature,
        )

        probabilities = F.softmax(scaled_logits, dim=-1)

        if (
            not torch.isfinite(probabilities).all()
            or (probabilities.sum(dim=-1) <= 0).any()
        ):
            raise ValueError(
                "Invalid probability distribution produced by the sampler."
            )

        next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

        return next_token.squeeze(0) if was_1d else next_token