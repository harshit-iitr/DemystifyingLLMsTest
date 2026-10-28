"""Stop criteria."""
import torch
from numbers import Integral
from typing import Optional, Union, Iterable


def normalize_eos_token_ids(eos_token_id: Optional[Union[int, Iterable[int]]]) -> set:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, Integral):
        return {int(eos_token_id)}
    return {int(token_id) for token_id in eos_token_id}


class StopCriteria:
    def __init__(self, max_length: int, eos_token_id: Optional[Union[int, Iterable[int]]] = None):
        self.max_length = max_length
        self.eos_token_ids = normalize_eos_token_ids(eos_token_id)

    def should_stop(self, current_length: int, last_token_id: Optional[int] = None) -> bool:
        """
        Check if generation should stop.
        """
        if current_length >= self.max_length:
            return True

        if last_token_id is not None and last_token_id in self.eos_token_ids:
            return True

        return False
