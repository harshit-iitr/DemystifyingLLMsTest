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
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value cache for a given layer.
        key, value shape: (batch_size, num_heads, seq_len, head_dim)
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} outside cache with {self.num_layers} layers")
        if key.shape[:-1] != value.shape[:-1]:
            raise ValueError("Key and value tensors must have matching batch, head, and sequence dimensions")

        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = key
            self.v_cache[layer_idx] = value
        else:
            if self.k_cache[layer_idx].shape[:2] != key.shape[:2] or self.k_cache[layer_idx].shape[-1] != key.shape[-1]:
                raise ValueError("New key tensor is incompatible with the cached batch, head, or feature dimensions")
            if self.v_cache[layer_idx].shape[:2] != value.shape[:2] or self.v_cache[layer_idx].shape[-1] != value.shape[-1]:
                raise ValueError("New value tensor is incompatible with the cached batch, head, or feature dimensions")
            self.k_cache[layer_idx] = torch.cat((self.k_cache[layer_idx], key), dim=-2)
            self.v_cache[layer_idx] = torch.cat((self.v_cache[layer_idx], value), dim=-2)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self):
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} outside cache with {self.num_layers} layers")
        if self.k_cache[layer_idx] is None:
            return 0
        return self.k_cache[layer_idx].shape[-2]
