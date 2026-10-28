"""Stop criteria."""
import torch
from typing import List, Optional


class StopCriteria:
    def __init__(self, max_length: int, eos_token_id: Optional[int] = None):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def should_stop(self, current_length: int, last_token_id: Optional[int] = None) -> bool:
        """
        Check if generation should stop.
        """
        if current_length >= self.max_length:
            return True

        if self.eos_token_id is not None and last_token_id is not None:
            if isinstance(last_token_id, torch.Tensor):
                return bool(torch.all(last_token_id == self.eos_token_id).item())
            if last_token_id == self.eos_token_id:
                return True

        return False
