"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn
import inspect

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from config.constants import BOS_TOKEN_ID
from inference.generation.kv_cache import KVCache
from inference.sampling.sampler import Sampler
from inference.sampling.repetition_penalty import apply_repetition_penalty


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
        if config.tie_weights:
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)
        self.num_layers = config.n_layers

    def forward(self, input_ids: torch.Tensor, kv_cache: Optional[KVCache] = None, **kwargs) -> torch.Tensor:
        offset = 0
        if kv_cache is not None:
            offset = kv_cache.get_seq_length()
        h = self.embeddings(input_ids, offset=offset)
        h = self.blocks(h, kv_cache=kv_cache, **kwargs)
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

        self.inference_config = inference_config if inference_config is not None else InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device) if device else torch.device("cpu")

        if model is not None:
            self.model = model
        else:
            self.model = FullTransformerModel(self.model_config)
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(self.device)

    def _prepare_model(self) -> None:
        """
        Internal lifecycle helper to prepare model state before inference execution.
        """
        if self.model is not None and hasattr(self.model, "train"):
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
            raise ValueError("max_new_tokens must be non-negative")
        if max_tokens == 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        if temp < 0:
            raise ValueError("temperature must be non-negative")
        selected_top_k = self.inference_config.top_k if top_k is None else top_k
        selected_top_p = self.inference_config.top_p if top_p is None else top_p
        if selected_top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 <= selected_top_p <= 1.0:
            raise ValueError("top_p must be in [0.0, 1.0]")
        tokens = list(prompt_tokens)
        if not tokens:
            tokens = [BOS_TOKEN_ID]
        try:
            forward_parameters = inspect.signature(self.model.forward).parameters
        except (AttributeError, TypeError, ValueError):
            forward_parameters = {}
        accepts_cache = "kv_cache" in forward_parameters
        use_cache = self.inference_config.use_kv_cache and accepts_cache
        cache = None
        if use_cache:
            num_layers = getattr(self.model, "num_layers", self.model_config.n_layers)
            cache = KVCache(num_layers=num_layers)
        sampler = Sampler(
            temperature=max(temp, torch.finfo(torch.float32).tiny),
            top_k=selected_top_k,
            top_p=selected_top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        generator = None
        if self.inference_config.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.inference_config.seed)

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                input_tokens = tokens if cache is None or cache.get_seq_length() == 0 else tokens[-1:]
                input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=self.device)

                if self.model is not None and callable(self.model):
                    logits = self.model(input_tensor, kv_cache=cache) if cache is not None else self.model(input_tensor)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                    if temp > 0:
                        next_token = sampler.sample(
                            next_token_logits,
                            generated_tokens=tokens,
                            generator=generator,
                        ).item()
                    else:
                        if self.inference_config.repetition_penalty != 1.0:
                            next_token_logits = apply_repetition_penalty(
                                next_token_logits,
                                tokens,
                                penalty=self.inference_config.repetition_penalty,
                            )
                        next_token = torch.argmax(next_token_logits, dim=-1).item()
                else:
                    raise RuntimeError("No callable model is available for generation")

                tokens.append(int(next_token))

                if self.inference_config.early_stopping and next_token in self.inference_config.stop_token_ids:
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
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)
        return "".join(chr(token) for token in output_ids if 0 <= token <= 0x10FFFF)
