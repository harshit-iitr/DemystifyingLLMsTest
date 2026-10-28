"""Convert temperature-scaled logits to a normalized probability distribution."""
import torch


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        greedy = torch.zeros_like(logits)
        return greedy.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
    return torch.softmax(logits / temperature, dim=-1)
