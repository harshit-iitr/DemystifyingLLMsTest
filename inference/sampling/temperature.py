"""Temperature scaling, including the deterministic zero-temperature limit."""
import math
import torch


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if temperature == 0:
        # The T -> 0 limit assigns all probability to the maximum logit.
        greedy = torch.full_like(logits, float("-inf"))
        return greedy.scatter(-1, logits.argmax(dim=-1, keepdim=True), 0.0)
    return logits / temperature
