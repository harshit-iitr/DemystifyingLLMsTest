"""Temperature scaling."""
import torch


def apply_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Scale logits by temperature.

    Lower temperatures sharpen the resulting probability distribution;
    higher temperatures make it more uniform.

    Temperature zero is rejected because greedy decoding should be
    handled directly by the generation engine.
    """
    if temperature <= 0:
        raise ValueError(
            "temperature must be strictly positive; "
            "use greedy decoding when temperature is zero."
        )

    return logits / temperature
