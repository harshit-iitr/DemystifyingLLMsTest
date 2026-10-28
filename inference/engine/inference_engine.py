"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from config.constants import EOS_TOKEN_ID
from inference.generation.kv_cache import KVCache
from inference.sampling.sampler import Sampler
from utils.logging_utils import get_logger

logger = get_logger("inference_engine")


class FullTransformerModel(nn.Module):
    """Fallback modular transformer wrapper combining embeddings, blocks, and lm_head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.n_layers
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead

        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        self.apply(self._initialize_weights)
        if config.tie_weights:
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        kv_cache = kwargs.get("kv_cache")
        offset = kv_cache.get_seq_length(0) if kv_cache is not None else 0
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        if attention_mask is not None:
            if attention_mask.shape != (input_ids.size(0), offset + input_ids.size(1)):
                raise ValueError("attention_mask must cover cached and current tokens")
            if position_ids is None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
                position_ids = position_ids[:, -input_ids.size(1):]
            padding_mask = torch.zeros(
                attention_mask.shape, device=input_ids.device,
                dtype=self.embeddings.token_embedding.embedding.weight.dtype,
            ).masked_fill(attention_mask.to(input_ids.device).eq(0), float("-inf"))
            padding_mask = padding_mask[:, None, None, :]
            mask = kwargs.get("mask")
            kwargs["mask"] = padding_mask if mask is None else mask + padding_mask
        h = self.embeddings(input_ids, offset=offset, position_ids=position_ids)
        h = self.blocks(h, position_ids=position_ids, **kwargs)
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
        self.model_config = model_config if model_config is not None else ModelConfig()

        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device) if device else torch.device("cpu")

        self.model = model if model is not None else FullTransformerModel(self.model_config)
        if hasattr(self.model, "to"):
            self.model.to(self.device)

    def _prepare_model(self) -> None:
        """
        Internal lifecycle helper to prepare model state before inference execution.
        """
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()

    def generate_tokens(
        self,
        prompt_tokens: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None
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

        if self.model is None:
            raise RuntimeError("Inference model is unavailable")
        temp = temperature if temperature is not None else self.inference_config.temperature
        sampler = Sampler(
            temperature=temp,
            top_k=top_k if top_k is not None else self.inference_config.top_k,
            top_p=top_p if top_p is not None else self.inference_config.top_p,
            repetition_penalty=(
                repetition_penalty
                if repetition_penalty is not None
                else self.inference_config.repetition_penalty
            ),
        )
        tokens = list(prompt_tokens)
        if not tokens:
            raise ValueError("prompt_tokens must contain at least one token")

        stop_token_ids = self.inference_config.stop_token_ids
        if self.inference_config.beam_width > 1:
            from inference.generation.beam_search import BeamSearch
            search = BeamSearch(
                beam_width=self.inference_config.beam_width,
                max_length=max_tokens,
                stop_token_ids=stop_token_ids if self.inference_config.early_stopping else [],
                context_length=self.model_config.max_seq_len,
            )
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            return search.search(self.model, input_ids)[0].tolist()
        use_cache = (
            self.inference_config.use_kv_cache
            and isinstance(self.model, FullTransformerModel)
            and len(tokens) + max_tokens <= self.model_config.max_seq_len
        )
        kv_cache = KVCache(self.model_config.n_layers) if use_cache else None
        generator = None
        if self.inference_config.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.inference_config.seed)

        # Autoregressive generation loop
        with torch.inference_mode():
            for step in range(max_tokens):
                if kv_cache is None:
                    context = tokens[-self.model_config.max_seq_len:]
                elif step == 0:
                    context = tokens
                else:
                    context = tokens[-1:]
                input_tensor = torch.tensor([context], dtype=torch.long, device=self.device)
                logits = self.model(input_tensor, kv_cache=kv_cache) if kv_cache is not None else self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]
                sampled = sampler.sample(
                    next_token_logits,
                    generated_tokens=tokens,
                    generator=generator,
                )
                next_token = int(sampled.item())
                tokens.append(next_token)

                if self.inference_config.early_stopping and next_token in stop_token_ids:
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
        self._prepare_model()

        # Tokenize prompt
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            raise ValueError("A tokenizer with an encode method is required for text generation")
        prompt_ids = tokenizer.encode(prompt)

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Decode generated tokens
        if tokenizer is None or not hasattr(tokenizer, "decode"):
            raise ValueError("A tokenizer with a decode method is required for text generation")
        return tokenizer.decode(output_ids)
