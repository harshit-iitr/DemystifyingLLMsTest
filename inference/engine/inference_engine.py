"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import inspect
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from inference.generation.autoregressive import autoregressive_generate
from inference.generation.beam_search import BeamSearch
from inference.sampling.sampler import Sampler
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
        self.apply(self._initialize_weights)
        if config.tie_weights:
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif hasattr(module, "weight") and hasattr(module, "bias"):
            weight = getattr(module, "weight")
            bias = getattr(module, "bias")
            if isinstance(weight, nn.Parameter) and isinstance(bias, nn.Parameter):
                nn.init.ones_(weight)
                nn.init.zeros_(bias)

    def forward(self, input_ids: torch.Tensor, kv_cache: Optional[Any] = None, **kwargs) -> torch.Tensor:
        offset = kv_cache.get_seq_length() if kv_cache is not None else 0
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
        top_p: Optional[float] = None
    ) -> List[int]:
        """
        Generate a list of token IDs autoregressively from input token sequence.
        """
        self._prepare_model()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_tokens}")
        if max_tokens <= 0:
            return list(prompt_tokens)
        if self.model is None or not callable(self.model):
            raise RuntimeError("No callable model is available for generation")
        if not prompt_tokens:
            raise ValueError("prompt_tokens must contain at least one token")

        input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        stop_ids = self.inference_config.stop_token_ids
        if self.inference_config.beam_width > 1:
            search = BeamSearch(
                beam_width=self.inference_config.beam_width,
                max_length=max_tokens,
                eos_token_id=stop_ids or None,
                early_stopping=self.inference_config.early_stopping,
            )
            with torch.no_grad():
                output = search.search(self.model, input_tensor)
            return output[0].tolist()

        temperature_value = temperature if temperature is not None else self.inference_config.temperature
        sampler = Sampler(
            temperature=temperature_value,
            top_k=self.inference_config.top_k if top_k is None else top_k,
            top_p=self.inference_config.top_p if top_p is None else top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        generator = None
        if self.inference_config.seed is not None and temperature_value > 0.0:
            try:
                generator = torch.Generator(device=self.device.type)
                generator.manual_seed(self.inference_config.seed)
            except (RuntimeError, TypeError):
                generator = None
        history = list(prompt_tokens)

        def sample_next(logits: torch.Tensor) -> torch.Tensor:
            next_ids = sampler.sample(logits, generated_tokens=history, generator=generator)
            history.append(int(next_ids[0].item()))
            return next_ids

        model_forward = getattr(self.model, "forward", self.model)
        try:
            forward_parameters = inspect.signature(model_forward).parameters.values()
            supports_cache = any(parameter.name == "kv_cache" for parameter in forward_parameters)
        except (TypeError, ValueError):
            supports_cache = False
        with torch.no_grad():
            output = autoregressive_generate(
                model=self.model,
                input_ids=input_tensor,
                max_new_tokens=max_tokens,
                eos_token_id=stop_ids or None,
                sampler_fn=sample_next,
                use_cache=self.inference_config.use_kv_cache and supports_cache,
            )
        return output[0].tolist()

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
        if tokenizer is None or not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise ValueError("A tokenizer with encode and decode methods is required for text generation")
        prompt_ids = tokenizer.encode(prompt)

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        return tokenizer.decode(output_ids)
