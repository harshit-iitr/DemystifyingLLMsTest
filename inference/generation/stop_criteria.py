"""Stop criteria."""
import torch
from typing import Optional, Sequence, Union


class StopCriteria:
    def __init__(self, max_length: int, eos_token_id: Optional[Union[int, Sequence[int]]] = None):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def should_stop(
        self, current_length: int, last_token_id: Optional[Union[int, torch.Tensor]] = None
    ) -> bool:
        """
        Check if generation should stop.
        """
        if current_length >= self.max_length:
            return True

        if self.eos_token_id is not None and last_token_id is not None:
            eos_ids = list(self.eos_token_id) if isinstance(self.eos_token_id, (list, tuple, set)) else [self.eos_token_id]
            if not eos_ids:
                return False
            if isinstance(last_token_id, torch.Tensor):
                eos_tensor = torch.as_tensor(eos_ids, device=last_token_id.device)
                if last_token_id.numel() and torch.isin(last_token_id, eos_tensor).all().item():
                    return True
            elif last_token_id in eos_ids:
                return True

        return False
