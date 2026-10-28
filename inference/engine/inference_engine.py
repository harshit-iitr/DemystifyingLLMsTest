"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from config.constants import EOS_TOKEN_ID, UNK_TOKEN_ID
from utils.logging_utils import get_logger
from inference.sampling.sampler import Sampler
from inference.generation.kv_cache import KVCache

logger = get_logger("inference_engine")


class FullTransformerModel(nn.Module):
    """Fallback modular transformer wrapper combining embeddings, blocks, and lm_head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead
        from model.heads.weight_tying import tie_weights
        from model.layers.layer_norm import LayerNorm

        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        self.apply(lambda module: self._initialize_weights(module, LayerNorm))
        if getattr(config, "tie_weights", False):
            tie_weights(self.lm_head, self.embeddings.token_embedding)
        self.num_layers = self.blocks.n_layers

    def _initialize_weights(self, module: nn.Module, layer_norm_type: type) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, layer_norm_type):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        kv_cache = kwargs.get("kv_cache")
        offset = kv_cache.get_seq_length() if kv_cache is not None else 0
        position_ids = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.long)
            position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
            if position_ids.size(1) != input_ids.size(1):
                position_ids = position_ids[:, -input_ids.size(1):]
            elif kv_cache is not None and offset:
                position_ids = position_ids + offset

        h = self.embeddings(input_ids, offset=offset, position_ids=position_ids)
        h = self.blocks(
            h,
            attention_mask=attention_mask,
            position_offset=offset,
            position_ids=position_ids,
            **kwargs,
        )
        logits = self.lm_head(h)
        return logits


class InferenceEngine:
    """
    End-to-end inference engine managing model state, token generation loop,
    and sampling orchestration.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        model: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        try:
            self.model_config = model_config or ModelConfig()
        except Exception:
            self.model_config = ModelConfig(d_model=768, vocab_size=50257)

        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device) if device else torch.device("cpu")

        if model is not None:
            self.model = model
        else:
            try:
                self.model = FullTransformerModel(self.model_config)
            except Exception as e:
                logger.debug(f"Could not build FullTransformerModel: {e}")
                self.model = None

    def _prepare_model(self) -> None:
        """
        Internal lifecycle helper to prepare model state before inference execution.
        """
        if self.model is not None:
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()

    def generate_tokens(
        self,
        prompt_tokens: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop_sequences: Optional[List[List[int]]] = None,
    ) -> List[int]:
        """
        Generate a list of token IDs autoregressively from input token sequence.
        """
        self._prepare_model()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_tokens == 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        tokens = list(prompt_tokens)
        if not tokens:
            raise ValueError("prompt_tokens must contain at least one token")
        if self.model is None:
            raise RuntimeError("No model is available for generation")

        sampler = Sampler(
            temperature=temp,
            top_k=self.inference_config.top_k if top_k is None else top_k,
            top_p=self.inference_config.top_p if top_p is None else top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        rng = None
        if self.inference_config.seed is not None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(self.inference_config.seed)
        stop_ids = stop_token_ids or self.inference_config.stop_token_ids or [EOS_TOKEN_ID]
        kv_cache = (
            KVCache(num_layers=getattr(self.model, "num_layers"))
            if self.inference_config.use_kv_cache and hasattr(self.model, "num_layers")
            else None
        )

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
                if kv_cache is not None:
                    model_input = input_tensor if kv_cache.get_seq_length() == 0 else input_tensor[:, -1:]
                    logits = self.model(model_input, kv_cache=kv_cache)
                else:
                    logits = self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]
                next_token = sampler.sample(
                    next_token_logits,
                    generated_tokens=tokens,
                    generator=rng,
                ).item()

                tokens.append(int(next_token))

                if next_token in stop_ids or any(
                    sequence and tokens[-len(sequence):] == sequence
                    for sequence in (stop_sequences or [])
                ):
                    break

        return tokens

    def generate(
        self,
        prompt: str,
        tokenizer: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        End-to-end string-to-string text generation.
        """
        # Tokenize prompt
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = [ord(c) if ord(c) < self.model_config.vocab_size else UNK_TOKEN_ID for c in prompt]

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Decode generated tokens
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)
        return prompt + "".join(chr(token) for token in output_ids[len(prompt_ids):] if 0 <= token <= 0x10FFFF)
