"""
Multi-Head Attention module.

  - M9: Heads are concatenated along wrong dimension (dim=-2 instead of dim=-1)
  - H10: Q, K, V projection matrices are all initialized from the SAME linear layer (shared weights)
"""

from typing import Optional, Any, Tuple
import torch
import torch.nn as nn

from .scaled_dot_product import scaled_dot_product_attention
from .causal_mask import create_causal_mask
from .rotary_pe import RotaryEmbedding, apply_rotary_pos_emb


def _resolve_config(explicit_config: Optional[Any] = None) -> Any:
    if explicit_config is not None:
        return explicit_config

    from config.model_config import ModelConfig
    return ModelConfig()


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Vaswani et al., 2017).
    Projects inputs into Q, K, V representations across multiple heads,
    performs scaled dot-product attention, and projects back to d_model.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        bias: bool = False
    ):
        super().__init__()
        cfg = _resolve_config(config)

        self.d_model = d_model if d_model is not None else getattr(cfg, "d_model", 768)
        self.n_heads = n_heads if n_heads is not None else getattr(cfg, "n_heads", 12)
        dropout_p = dropout if dropout is not None else getattr(cfg, "dropout", 0.1)

        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        self.d_head = self.d_model // self.n_heads
        self.rotary = (
            RotaryEmbedding(self.d_head, max_seq_len=getattr(cfg, "max_seq_len", 2048))
            if getattr(cfg, "use_rotary", False) else None
        )

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=bias)

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.attn_dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        layer_idx: Optional[int] = None,
        use_causal_mask: bool = True
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional additive attention mask
            kv_cache: Optional KVCache instance
            layer_idx: Layer index for KV caching
            use_causal_mask: Whether to automatically generate causal mask if mask is None

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project inputs to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into (batch_size, n_heads, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Update KV cache if provided
        cache_offset = kv_cache.get_seq_length(layer_idx) if kv_cache is not None and layer_idx is not None else 0
        if self.rotary is not None:
            cos, sin = self.rotary(q, seq_len=seq_len, offset=cache_offset)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if kv_cache is not None and layer_idx is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # Convert a conventional 2D keep-mask (1/True = real token) to an
        # additive key-padding mask. Keep it separate from ``mask``, whose
        # public contract is already additive.
        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.size(0) != batch_size:
                raise ValueError("attention_mask must have shape (batch_size, key_len)")
            if attention_mask.size(1) != k.size(-2):
                raise ValueError(
                    f"attention_mask key length ({attention_mask.size(1)}) must match "
                    f"attention key length ({k.size(-2)})"
                )
            keep_keys = attention_mask.to(device=x.device, dtype=torch.bool)
            padding_mask = torch.zeros(
                (batch_size, 1, 1, k.size(-2)), device=x.device, dtype=x.dtype
            ).masked_fill(~keep_keys[:, None, None, :], float("-inf"))
            mask = padding_mask if mask is None else mask + padding_mask

        # 4. Prepare causal mask if needed
        if use_causal_mask:
            key_len = k.size(-2)
            query_offset = key_len - seq_len
            causal_mask = create_causal_mask(
                seq_len=seq_len,
                key_len=key_len,
                query_offset=query_offset,
                device=x.device,
                dtype=x.dtype,
            )
            mask = causal_mask if mask is None else mask + causal_mask

        # 5. Scaled dot-product attention
        attn_out, _ = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            dropout=self.attn_dropout
        )

        # 6. Recombine heads
        context = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(context)
