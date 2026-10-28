"""
Multi-Head Attention module.

"""

from typing import Optional, Any, Tuple
import torch
import torch.nn as nn

from .scaled_dot_product import scaled_dot_product_attention
from .causal_mask import create_causal_mask


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

        self.d_model = d_model or getattr(cfg, "d_model", 768)
        self.n_heads = n_heads or getattr(cfg, "n_heads", 12)
        dropout_p = dropout if dropout is not None else getattr(cfg, "dropout", 0.1)

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        self.d_head = self.d_model // self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.rotary = None
        if getattr(cfg, "use_rotary", False):
            from .rotary_pe import RotaryEmbedding
            self.rotary = RotaryEmbedding(self.d_head, getattr(cfg, "max_seq_len", 1024))

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.attn_dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
            use_causal_mask: Whether to combine the causal mask with any explicit mask

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

        past_len = kv_cache.get_seq_length(layer_idx) if kv_cache is not None and layer_idx is not None else 0
        if self.rotary is not None:
            from .rotary_pe import apply_rotary_pos_emb
            cos, sin = self.rotary(q, past_len + seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos[..., past_len:, :], sin[..., past_len:, :])

        # 3. Update KV cache if provided
        if kv_cache is not None and layer_idx is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # Convert boolean allowed-position masks before combining additive biases.
        if mask is not None:
            mask = mask.to(device=x.device)
            if mask.dtype == torch.bool:
                mask = torch.zeros_like(mask, dtype=x.dtype).masked_fill(~mask, float("-inf"))
        # 4. Prepare causal mask if needed
        if use_causal_mask:
            query_positions = torch.arange(past_len, past_len + seq_len, device=x.device)
            key_positions = torch.arange(k.size(-2), device=x.device)
            causal = torch.zeros((seq_len, k.size(-2)), device=x.device, dtype=x.dtype)
            causal.masked_fill_(key_positions[None, :] > query_positions[:, None], float("-inf"))
            mask = causal[None, None] if mask is None else mask + causal[None, None]

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
