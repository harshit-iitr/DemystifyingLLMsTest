"""Module implementation."""
import torch


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Scale logits by temperature; zero temperature is handled as greedy decoding by Sampler.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive for temperature scaling")
    return logits / temperature
