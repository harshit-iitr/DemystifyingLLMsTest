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
        kv_cache = kwargs.get("kv_cache")
        offset = kv_cache.get_seq_length(0) if kv_cache is not None else 0
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
        top_p: Optional[float] = None
    ) -> List[int]:
        """
        Generate a list of token IDs autoregressively from input token sequence.
        """
        self._prepare_model()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens <= 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        if temp < 0:
            raise ValueError("temperature must be non-negative")
        top_k = self.inference_config.top_k if top_k is None else top_k
        top_p = self.inference_config.top_p if top_p is None else top_p
        tokens = list(prompt_tokens)
        if self.inference_config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.inference_config.seed)
        else:
            generator = None

        from inference.sampling import Sampler
        sampler = Sampler(
            temperature=max(temp, 1e-8), top_k=top_k, top_p=top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        stop_ids = set(self.inference_config.stop_token_ids)
        if not stop_ids:
            from config.constants import EOS_TOKEN_ID
            stop_ids.add(EOS_TOKEN_ID)

        try:
            parameters = inspect.signature(self.model.forward).parameters.values()
            supports_cache = any(
                parameter.name == "kv_cache" or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (AttributeError, TypeError, ValueError):
            supports_cache = isinstance(self.model, FullTransformerModel)
        use_cache = self.inference_config.use_kv_cache and supports_cache
        if use_cache:
            from inference.generation import KVCache
            kv_cache = KVCache(getattr(self.model, "num_layers", self.model_config.n_layers))
        else:
            kv_cache = None

        with torch.no_grad():
            for step in range(max_tokens):
                current_tokens = tokens[-1:] if use_cache and step > 0 else tokens
                input_tensor = torch.tensor([current_tokens], dtype=torch.long, device=self.device)
                if self.model is None or not callable(self.model):
                    raise RuntimeError("No callable model is available for token generation")
                logits = self.model(input_tensor, kv_cache=kv_cache) if use_cache else self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]

                if temp == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1).item()
                else:
                    next_token = sampler.sample(next_token_logits, generated_tokens=torch.tensor([tokens], device=self.device), generator=generator).item()
                tokens.append(int(next_token))
                if next_token in stop_ids:
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
        if tokenizer is None or not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise ValueError("A tokenizer with encode() and decode() is required")
        prompt_ids = tokenizer.encode(prompt)

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        return tokenizer.decode(output_ids)
