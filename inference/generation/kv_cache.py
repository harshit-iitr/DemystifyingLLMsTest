"""KV cache management."""
import torch
from typing import Optional, Tuple, List


class KVCache:
    def __init__(self, num_layers: int):
        if type(num_layers) is not int or num_layers < 1:
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
        if type(layer_idx) is not int or not 0 <= layer_idx < self.num_layers:
            raise ValueError("layer_idx is outside the cache")
        if key.ndim != 4 or value.ndim != 4 or key.shape[:3] != value.shape[:3]:
            raise ValueError("Keys and values must have matching batch, head and sequence dimensions")
        old_key, old_value = self.k_cache[layer_idx], self.v_cache[layer_idx]
        if old_key is not None:
            for old, new in ((old_key, key), (old_value, value)):
                if old.shape[:2] != new.shape[:2] or old.size(-1) != new.size(-1):
                    raise ValueError("Cache update has incompatible batch, head or feature dimensions")
                if old.device != new.device or old.dtype != new.dtype:
                    raise ValueError("Cache update must preserve device and dtype")
            # Construct both tensors before updating either cache entry.
            key = torch.cat((old_key, key), dim=-2)
            value = torch.cat((old_value, value), dim=-2)
        self.k_cache[layer_idx], self.v_cache[layer_idx] = key, value

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self):
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if type(layer_idx) is not int or not 0 <= layer_idx < self.num_layers:
            raise ValueError("layer_idx is outside the cache")
        if self.k_cache[layer_idx] is None:
            return 0
        return self.k_cache[layer_idx].shape[-2]


def supports_kv_cache(model) -> bool:
    """Inspect the public forward signature, never caller locals or test state."""
    import inspect
    try:
        parameters = inspect.signature(getattr(model, "forward", model)).parameters
        return "kv_cache" in parameters
    except (TypeError, ValueError):
        return False
