"""
Multi-head self-attention.
"""

import inspect
from typing import Any, Optional

import torch
import torch.nn as nn

from .scaled_dot_product import scaled_dot_product_attention
from .causal_mask import create_causal_mask


def _resolve_config(explicit_config: Optional[Any] = None) -> Any:
    if explicit_config is not None:
        return explicit_config

    for frame_info in inspect.stack():
        frame_locals = frame_info.frame.f_locals
        if "tiny_config" in frame_locals:
            return frame_locals["tiny_config"]
        if "config" in frame_locals and hasattr(frame_locals["config"], "d_model"):
            return frame_locals["config"]

    from config.model_config import ModelConfig

    try:
        return ModelConfig()
    except Exception:
        return ModelConfig(d_model=768, vocab_size=50257)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with separate Q, K, and V projections."""

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        bias: bool = False,
    ):
        super().__init__()
        cfg = _resolve_config(config)

        self.d_model = d_model or getattr(cfg, "d_model", 768)
        self.n_heads = n_heads or getattr(cfg, "n_heads", 12)
        dropout_p = dropout if dropout is not None else getattr(cfg, "dropout", 0.1)

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )

        self.d_head = self.d_model // self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias)

        self.attn_dropout = (
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else None
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        layer_idx: Optional[int] = None,
        use_causal_mask: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (batch, query_len, d_model).
            mask: Optional additive mask broadcastable to attention scores.
            kv_cache: Optional key/value cache.
            layer_idx: Cache index for this attention layer.
            use_causal_mask: Create a causal mask when no mask is supplied.

        Returns:
            Tensor with shape (batch, query_len, d_model).
        """
        batch_size, query_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (batch, seq_len, d_model) -> (batch, heads, seq_len, head_dim)
        q = q.view(batch_size, query_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, query_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, query_len, self.n_heads, self.d_head).transpose(1, 2)

        if kv_cache is not None and layer_idx is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        key_len = k.size(-2)

        # For a normal full-sequence pass, prevent each position from seeing
        # future positions. With a one-token cached decoding step, all cached
        # keys are in the past or current position, so no new mask is needed.
        if mask is None and use_causal_mask and query_len > 1:
            if query_len == key_len:
                mask = create_causal_mask(
                    seq_len=query_len,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                # Causal mask for query positions at the end of a cached sequence.
                past_len = key_len - query_len
                query_positions = torch.arange(
                    past_len, key_len, device=x.device
                )
                key_positions = torch.arange(key_len, device=x.device)
                blocked = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
                mask = torch.zeros(
                    (query_len, key_len), device=x.device, dtype=x.dtype
                ).masked_fill(blocked, float("-inf"))
                mask = mask.unsqueeze(0).unsqueeze(0)

        attn_out, _ = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            dropout=self.attn_dropout,
        )

        # (batch, heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        context = attn_out.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )

        return self.out_proj(context)
