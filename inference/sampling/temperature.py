"""Temperature scaling of logits; Sampler handles zero-temperature argmax."""
import math
import torch


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if temperature == 0:
        # Keep the logits valid and finite for a caller selecting argmax;
        # stochastic probability scaling is only defined for positive T.
        return logits
    return logits / temperature
