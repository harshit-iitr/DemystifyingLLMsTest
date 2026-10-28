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
        self.num_layers = config.n_layers
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead

        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        if config.tie_weights:
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cache = kv_cache
        offset = cache.get_seq_length(0) if cache is not None else 0
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.clamp_min(0)[:, -input_ids.size(1):]
        h = self.embeddings(input_ids, offset=offset, position_ids=position_ids)
        h = self.blocks(h, mask=attention_mask, kv_cache=cache)
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
        if self.model is not None and hasattr(self.model, "train"):
            self.model.eval()
            self.model.to(self.device)

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
        if self.inference_config.seed is not None:
            torch.manual_seed(self.inference_config.seed)
        top_k = self.inference_config.top_k if top_k is None else top_k
        top_p = self.inference_config.top_p if top_p is None else top_p
        tokens = list(prompt_tokens)
        if self.inference_config.beam_width > 1:
            if self.model is None:
                raise RuntimeError("No model is available for beam search")
            from inference.generation.beam_search import BeamSearch
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
            beam_search = BeamSearch(
                beam_width=self.inference_config.beam_width,
                max_length=max_tokens,
                eos_token_id=self.inference_config.stop_token_ids[0] if self.inference_config.stop_token_ids else None,
                early_stopping=self.inference_config.early_stopping,
            )
            with torch.no_grad():
                result = beam_search.search(self.model, input_tensor)
            return result[0].tolist()

        from inference.sampling.sampler import Sampler
        from inference.generation.kv_cache import KVCache
        sampler = Sampler(
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        accepts_cache = False
        if self.model is not None:
            try:
                parameters = inspect.signature(self.model.forward).parameters
                accepts_cache = "kv_cache" in parameters
            except (AttributeError, TypeError, ValueError):
                accepts_cache = False
        kv_cache = (
            KVCache(getattr(self.model, "num_layers", self.model_config.n_layers))
            if self.inference_config.use_kv_cache and accepts_cache else None
        )

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                current_ids = tokens if kv_cache is None or kv_cache.get_seq_length(0) == 0 else tokens[-1:]
                input_tensor = torch.tensor([current_ids], dtype=torch.long, device=self.device)

                if self.model is None or not callable(self.model):
                    raise RuntimeError("No callable model is available for inference")
                logits = self.model(input_tensor, kv_cache=kv_cache) if kv_cache is not None else self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                sampled = sampler.sample(next_token_logits, tokens)
                next_token = sampled.item()

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
            try:
                prompt_ids = tokenizer.encode(prompt)
            except Exception:
                prompt_ids = [ord(c) for c in prompt]
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
            try:
                return tokenizer.decode(output_ids)
            except Exception:
                raise

        # Fallback decode: append dummy synthesized continuation
        raise RuntimeError("A tokenizer with decode() is required to generate text")
