"""Combined sampler (Clean)."""
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
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.generator = None
        self.seed = seed
        self.generator_device = None
        if seed is not None:
            self.generator_device = torch.device(device or "cpu")
            self.generator = torch.Generator(device=self.generator_device)
            self.generator_device = self.generator.device
            self.generator.manual_seed(seed)

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

        # 2. Top-K
        if self.top_k > 0:
            logits = apply_top_k(logits, k=self.top_k)

        # 3. Top-P
        if self.top_p < 1.0:
            logits = apply_top_p(logits, p=self.top_p)

        # 4. Temperature. Zero selects greedily; positive values sample.
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1)
        scaled = apply_temperature(logits, temperature=self.temperature)
        probs = F.softmax(scaled, dim=-1)

        # torch.multinomial requires its generator to be on the same device as
        # the probabilities. Callers may omit ``device`` or move logits after
        # constructing the sampler, so align the seeded generator lazily.
        if self.seed is not None:
            same_device = (
                self.generator_device.type == probs.device.type
                and (self.generator_device.index is None or probs.device.index is None
                     or self.generator_device.index == probs.device.index)
            )
            if not same_device:
                self.generator_device = probs.device
                self.generator = torch.Generator(device=self.generator_device)
                self.generator.manual_seed(self.seed)

        # 5. Multinomial sample
        next_token = torch.multinomial(probs, num_samples=1, generator=self.generator)
        return next_token.squeeze(-1)
