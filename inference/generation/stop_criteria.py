"""Stop criteria."""
import torch
from typing import Optional, Union, Sequence


class StopCriteria:
    def __init__(self, max_length: int, eos_token_id: Optional[int] = None):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def should_stop(
        self,
        current_length: int,
        last_token_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
    ) -> bool:
        """
        Check if generation should stop.
        """
        if current_length >= self.max_length:
            return True

        if self.eos_token_id is not None and last_token_id is not None:
            if isinstance(last_token_id, torch.Tensor):
                return bool(torch.all(last_token_id == self.eos_token_id).item())
            if isinstance(last_token_id, Sequence) and not isinstance(last_token_id, (str, bytes)):
                return bool(last_token_id) and all(token == self.eos_token_id for token in last_token_id)
            return last_token_id == self.eos_token_id

        return False
