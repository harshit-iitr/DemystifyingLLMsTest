"""Temperature scaling for sampling."""
import torch


def apply_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    if temperature == 0:
        best = logits.argmax(dim=-1, keepdim=True)
        result = torch.full_like(logits, float("-inf"))
        return result.scatter(-1, best, 0.0)

    return logits / temperature