"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn
import inspect

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
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

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        cache = kwargs.get("kv_cache")
        offset = cache.get_seq_length(0) if cache is not None else 0
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
        if self.model is not None and hasattr(self.model, "to"):
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
        top_p: Optional[float] = None
    ) -> List[int]:
        """
        Generate a list of token IDs autoregressively from input token sequence.
        """
        was_training = self.model.training if self.model is not None and hasattr(self.model, "training") else None
        self._failed = False
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_tokens}")
        if max_tokens == 0:
            if was_training is not None and self.model is not None:
                self.model.train(was_training)
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        tokens = list(prompt_tokens)
        if not tokens:
            raise ValueError("prompt_tokens must contain at least one token")

        # Autoregressive generation loop
        from inference.sampling.sampler import Sampler
        sampler = Sampler(
            temperature=temp,
            top_k=top_k if top_k is not None else self.inference_config.top_k,
            top_p=top_p if top_p is not None else self.inference_config.top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        from inference.generation.kv_cache import KVCache
        use_cache = bool(self.inference_config.use_kv_cache and self.model is not None)
        if use_cache:
            try:
                parameters = inspect.signature(self.model.forward).parameters.values()
                supports_cache = any(p.name == "kv_cache" or p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
            except (AttributeError, TypeError, ValueError):
                supports_cache = False
            use_cache = supports_cache
        kv_cache = KVCache(getattr(self.model, "num_layers", self.model_config.n_layers)) if use_cache else None
        with torch.inference_mode():
            for step in range(max_tokens):
                model_tokens = tokens if kv_cache is None or step == 0 else tokens[-1:]
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)

                if self.model is not None and callable(self.model):
                    logits = self.model(input_tensor, kv_cache=kv_cache) if kv_cache is not None else self.model(input_tensor)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    next_token_logits = logits[:, -1, :]
                    next_token = sampler.sample(next_token_logits, generated_tokens=tokens).item()
                else:
                    self._failed = True
                    raise RuntimeError("No callable model is available for token generation")

                tokens.append(int(next_token))

                if next_token in self.inference_config.stop_token_ids:
                    break

        if was_training is not None and self.model is not None:
            self.model.train(was_training)
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
        self._failed = False

        # Tokenize prompt
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            prompt_ids = tokenizer.encode(prompt)
        else:
            raise ValueError("A tokenizer with an encode method is required")

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Decode generated tokens
        if not getattr(self, "_failed", False) and tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)

        raise RuntimeError("Unable to decode generated token IDs with the provided tokenizer")
