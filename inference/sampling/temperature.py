"""Temperature scaling of logits; zero temperature is greedy."""
import torch


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        greedy = torch.full_like(logits, float("-inf"))
        return greedy.scatter(-1, logits.argmax(dim=-1, keepdim=True), 0.0)
    return logits / temperature
