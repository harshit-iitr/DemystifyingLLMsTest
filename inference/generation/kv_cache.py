"""KV cache management."""
import torch
from typing import Optional, Tuple, List


class KVCache:
    def __init__(self, num_layers: int):
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
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
        if not isinstance(layer_idx, int) or isinstance(layer_idx, bool):
            raise TypeError("layer_idx must be an integer")
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} is outside cache range [0, {self.num_layers})")
        if key.dim() != 4 or value.dim() != 4:
            raise ValueError("key and value tensors must have shape (batch, heads, seq_len, head_dim)")
        if key.shape != value.shape:
            raise ValueError("key and value tensors must have identical shapes")
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = key
            self.v_cache[layer_idx] = value
        else:
            cached_key = self.k_cache[layer_idx]
            cached_value = self.v_cache[layer_idx]
            if key.shape[:2] != cached_key.shape[:2] or key.shape[3] != cached_key.shape[3]:
                raise ValueError("new key/value batch, head, and feature dimensions must match cache")
            if key.device != cached_key.device or key.dtype != cached_key.dtype:
                raise ValueError("new key/value device and dtype must match cache")
            self.k_cache[layer_idx] = torch.cat((self.k_cache[layer_idx], key), dim=-2)
            self.v_cache[layer_idx] = torch.cat((self.v_cache[layer_idx], value), dim=-2)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self):
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self.k_cache[layer_idx] is None:
            return 0
        return self.k_cache[layer_idx].shape[-2]
