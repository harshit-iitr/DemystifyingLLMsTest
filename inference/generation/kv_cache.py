"""KV cache management."""
import torch
from typing import Optional, Tuple, List


class KVCache:
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
        """
        Update the key/value cache for a layer.

        key and value have shape:
            (batch_size, num_heads, seq_len, head_dim)

        Returns the complete cached keys and values.
        """
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} is outside the valid range "
                f"[0, {self.num_layers})"
            )

        if key.shape != value.shape:
            raise ValueError("key and value must have matching shapes")

        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = key
            self.v_cache[layer_idx] = value
        else:
            cached_key = self.k_cache[layer_idx]
            cached_value = self.v_cache[layer_idx]

            if (
                key.shape[:2] != cached_key.shape[:2]
                or key.shape[-1] != cached_key.shape[-1]
            ):
                raise ValueError("New keys are incompatible with the existing cache")

            if (
                value.shape[:2] != cached_value.shape[:2]
                or value.shape[-1] != cached_value.shape[-1]
            ):
                raise ValueError("New values are incompatible with the existing cache")

            if key.device != cached_key.device or key.dtype != cached_key.dtype:
                raise ValueError("New keys must match the cache device and dtype")

            if value.device != cached_value.device or value.dtype != cached_value.dtype:
                raise ValueError("New values must match the cache device and dtype")

            self.k_cache[layer_idx] = torch.cat(
                [cached_key, key], dim=-2
            )
            self.v_cache[layer_idx] = torch.cat(
                [cached_value, value], dim=-2
            )

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self):
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self.k_cache[layer_idx] is None:
            return 0
        return self.k_cache[layer_idx].shape[-2]
