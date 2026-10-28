"""Stop criteria."""
import torch
from typing import List, Optional


class StopCriteria:
    def __init__(self, max_length: int, eos_token_id: Optional[int] = None):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def should_stop(self, current_length: int, last_token_id: Optional[object] = None) -> bool:
        """
        Check if generation should stop.
        """
        if current_length >= self.max_length:
            return True

        if self.eos_token_id is not None and last_token_id is not None:
            if isinstance(last_token_id, torch.Tensor):
                if last_token_id.numel() and bool(torch.all(last_token_id == self.eos_token_id)):
                    return True
            elif isinstance(last_token_id, (list, tuple)):
                if last_token_id and all(token == self.eos_token_id for token in last_token_id):
                    return True
            elif last_token_id == self.eos_token_id:
                return True

        return False
