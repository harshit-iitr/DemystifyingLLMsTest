"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from inference.generation import KVCache
from inference.sampling import Sampler
from utils.logging_utils import get_logger

logger = get_logger("inference_engine")


class FullTransformerModel(nn.Module):
    """Fallback modular transformer wrapper combining embeddings, blocks, and lm_head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead

        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        self.num_layers = config.n_layers
        if config.tie_weights:
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def forward(self, input_ids: torch.Tensor, offset: int = 0, **kwargs) -> torch.Tensor:
        h = self.embeddings(input_ids, offset=offset)
        h = self.blocks(h, **kwargs)
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
        self.model_config = model_config or ModelConfig()

        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device) if device else torch.device("cpu")

        if model is not None:
            self.model = model
        else:
            self.model = FullTransformerModel(self.model_config)
        if isinstance(self.model, nn.Module):
            self.model.to(self.device)

    def _prepare_model(self) -> None:
        """
        Internal lifecycle helper to prepare model state before inference execution.
        """
        if isinstance(self.model, nn.Module):
            self.model.eval()

    def generate_tokens(
        self,
        prompt_tokens: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> List[int]:
        """
        Generate a list of token IDs autoregressively from input token sequence.
        """
        self._prepare_model()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_tokens}")
        if max_tokens == 0:
            return list(prompt_tokens)
        if not prompt_tokens:
            raise ValueError("prompt_tokens must contain at least one token")
        if self.model is None or not callable(self.model):
            raise RuntimeError("No callable model is available for generation")

        temp = temperature if temperature is not None else self.inference_config.temperature
        sampling_top_k = top_k if top_k is not None else self.inference_config.top_k
        sampling_top_p = top_p if top_p is not None else self.inference_config.top_p
        tokens = list(prompt_tokens)
        sampler = Sampler(
            temperature=temp,
            top_k=sampling_top_k,
            top_p=sampling_top_p,
            repetition_penalty=self.inference_config.repetition_penalty
        )
        generator = None
        if self.inference_config.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.inference_config.seed)
        cache = (
            KVCache(num_layers=self.model.num_layers)
            if self.inference_config.use_kv_cache and isinstance(self.model, FullTransformerModel)
            else None
        )

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                if cache is not None:
                    cached_length = cache.get_seq_length()
                    if cached_length >= self.model_config.max_seq_len:
                        cache.reset()
                        cached_length = 0
                    model_tokens = tokens[cached_length:] if cached_length else tokens[-self.model_config.max_seq_len:]
                    input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)
                    logits = self.model(input_tensor, kv_cache=cache, offset=cached_length)
                else:
                    model_tokens = tokens[-self.model_config.max_seq_len:]
                    input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)
                    logits = self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]
                next_token = sampler.sample(
                    next_token_logits,
                    generated_tokens=tokens,
                    generator=generator
                ).item()

                tokens.append(int(next_token))

                if next_token in self.inference_config.stop_token_ids:
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
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = [ord(c) for c in prompt]

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
