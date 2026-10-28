from typing import List, Optional, Tuple

import torch


class KVCache:
    """Store and append keys and values for each transformer layer."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.k_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.v_cache: List[Optional[torch.Tensor]] = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"Invalid layer_idx {layer_idx}")

        if key.shape[:-1] != value.shape[:-1]:
            raise ValueError("Key and value cache shapes are incompatible")

        old_key = self.k_cache[layer_idx]
        old_value = self.v_cache[layer_idx]

        if old_key is None:
            self.k_cache[layer_idx] = key
            self.v_cache[layer_idx] = value
        else:
            if old_key.shape[:2] != key.shape[:2] or old_key.shape[-1] != key.shape[-1]:
                raise ValueError("New key shape does not match cached keys")
            if old_value.shape[:2] != value.shape[:2] or old_value.shape[-1] != value.shape[-1]:
                raise ValueError("New value shape does not match cached values")

            self.k_cache[layer_idx] = torch.cat((old_key, key), dim=-2)
            self.v_cache[layer_idx] = torch.cat((old_value, value), dim=-2)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self) -> None:
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"Invalid layer_idx {layer_idx}")
        cache = self.k_cache[layer_idx]
        return 0 if cache is None else cache.shape[-2]
