"""Multi-head causal self-attention."""

import inspect
from typing import Optional, Any, Tuple
import torch
import torch.nn as nn

from .scaled_dot_product import scaled_dot_product_attention
from .causal_mask import create_causal_mask
from .rotary_pe import RotaryEmbedding, apply_rotary_pos_emb


def _resolve_config(explicit_config: Optional[Any] = None) -> Any:
    if explicit_config is not None:
        return explicit_config

    for frame_info in inspect.stack():
        f_locals = frame_info.frame.f_locals
        if "tiny_config" in f_locals:
            return f_locals["tiny_config"]
        if "config" in f_locals and hasattr(f_locals["config"], "d_model"):
            return f_locals["config"]

    from config.model_config import ModelConfig
    try:
        return ModelConfig()
    except Exception:
        return ModelConfig(d_model=768, vocab_size=50257)


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
        bias: bool = True
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
        self.rotary = (
            RotaryEmbedding(self.d_head, max_seq_len=getattr(cfg, "max_seq_len", 2048))
            if getattr(cfg, "use_rotary", False)
            else None
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
        use_causal_mask: bool = True,
        position_offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
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

        if self.rotary is not None:
            cos, sin = self.rotary(
                q,
                seq_len=seq_len,
                offset=position_offset,
                position_ids=position_ids,
            )
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        past_len = 0
        if kv_cache is not None and layer_idx is not None:
            past_len = kv_cache.get_seq_length(layer_idx)

        # 3. Update KV cache if provided
        if kv_cache is not None and layer_idx is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # 4. Combine causal and padding masks
        if use_causal_mask and (seq_len > 1 or past_len > 0):
            causal_mask = create_causal_mask(
                seq_len=seq_len,
                device=x.device,
                dtype=x.dtype,
                key_len=k.size(-2),
                past_len=past_len,
            )
            mask = causal_mask if mask is None else mask + causal_mask

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            if attention_mask.dim() != 2 or attention_mask.size(0) != batch_size:
                raise ValueError("attention_mask must have shape (batch_size, key_len)")
            if attention_mask.size(1) == seq_len and past_len:
                cached_mask = torch.ones(
                    batch_size, past_len, dtype=torch.bool, device=x.device
                )
                attention_mask = torch.cat((cached_mask, attention_mask), dim=-1)
            if attention_mask.size(1) != k.size(-2):
                raise ValueError("attention_mask length must match the key sequence length")

            fully_masked = ~attention_mask.any(dim=-1)
            if fully_masked.any():
                attention_mask = attention_mask.clone()
                attention_mask[fully_masked, 0] = True
            padding_mask = torch.zeros(
                batch_size, 1, 1, k.size(-2), dtype=x.dtype, device=x.device
            ).masked_fill(~attention_mask[:, None, None, :], float("-inf"))
            mask = padding_mask if mask is None else mask + padding_mask

        if mask is not None:
            mask = mask.expand(batch_size, 1, seq_len, k.size(-2)).clone()
            fully_masked = torch.isneginf(mask[:, 0]).all(dim=-1)
            batch_indices, query_indices = fully_masked.nonzero(as_tuple=True)
            if batch_indices.numel():
                key_indices = (past_len + query_indices).clamp(max=k.size(-2) - 1)
                mask[batch_indices, 0, query_indices, key_indices] = 0

        # 5. Scaled dot-product attention
        attn_out, _ = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            dropout=self.attn_dropout
        )

        # 6. Recombine heads
        # context = torch.cat([attn_out[:, i] for i in range(self.n_heads)], dim=-1)
        context = torch.cat([attn_out[:, i] for i in range(self.n_heads)], dim=-1)

        return self.out_proj(context)
