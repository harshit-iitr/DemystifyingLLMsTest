"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import inspect
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
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
        if config.tie_weights:
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        offset = kwargs.pop("offset", 0)
        cache = kwargs.get("kv_cache")
        if cache is not None and cache.num_layers and cache.get_seq_length(0):
            offset = cache.get_seq_length(0)
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
            self.model = self.model.to(self.device)

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
        self._prepare_model()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_tokens}")
        if max_tokens == 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        k = top_k if top_k is not None else self.inference_config.top_k
        p = top_p if top_p is not None else self.inference_config.top_p
        if temp < 0:
            raise ValueError(f"temperature must be non-negative, got {temp}")
        if k < 0:
            raise ValueError(f"top_k must be non-negative, got {k}")
        if not 0 <= p <= 1:
            raise ValueError(f"top_p must be between 0 and 1, got {p}")
        tokens = list(prompt_tokens)
        if not tokens:
            raise ValueError("prompt_tokens must contain at least one token")

        use_cache = self.inference_config.use_kv_cache
        cache = None
        if use_cache and self.model is not None:
            try:
                parameters = inspect.signature(self.model.forward).parameters.values()
                supports_cache = any(
                    parameter.name == "kv_cache" or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            except (AttributeError, TypeError, ValueError):
                supports_cache = False
            if supports_cache:
                from inference.generation.kv_cache import KVCache
                num_layers = getattr(self.model, "num_layers", self.model_config.n_layers)
                cache = KVCache(num_layers=num_layers)

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                cached_length = cache.get_seq_length(0) if cache is not None else 0
                model_tokens = tokens[cached_length:] if cached_length else tokens
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)

                if self.model is not None and callable(self.model):
                    logits = self.model(input_tensor, kv_cache=cache) if cache is not None else self.model(input_tensor)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                    from inference.sampling import Sampler
                    sampler = Sampler(
                        temperature=temp if temp > 0 else 1.0,
                        top_k=k,
                        top_p=p,
                        repetition_penalty=self.inference_config.repetition_penalty,
                    )
                    if temp > 0:
                        next_token = sampler.sample(next_token_logits, tokens).item()
                    else:
                        processed = next_token_logits
                        if self.inference_config.repetition_penalty != 1.0:
                            from inference.sampling.repetition_penalty import apply_repetition_penalty
                            processed = apply_repetition_penalty(
                                processed, tokens, self.inference_config.repetition_penalty
                            )
                        if k > 0:
                            from inference.sampling.top_k import apply_top_k
                            processed = apply_top_k(processed, k)
                        if p < 1.0:
                            from inference.sampling.top_p import apply_top_p
                            processed = apply_top_p(processed, p)
                        next_token = torch.argmax(processed, dim=-1).item()
                else:
                    raise RuntimeError("InferenceEngine has no callable model")

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
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)
        return "".join(chr(token_id) for token_id in output_ids if 0 <= token_id <= 0x10FFFF)
