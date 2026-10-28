"""Temperature scaling."""
import torch


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Scale logits by 1/temperature. Returns scaled LOGITS (not probabilities).
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0 (use greedy decoding for 0), got {temperature}")
    return logits / temperature
