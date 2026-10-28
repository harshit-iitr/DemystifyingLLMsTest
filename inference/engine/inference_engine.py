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
        if config.tie_weights:
            from model.heads.weight_tying import tie_weights
            tie_weights(self.lm_head, self.embeddings.token_embedding)
        self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        kv_cache = kwargs.get("kv_cache")
        offset = kwargs.pop("position_offset", kv_cache.get_seq_length() if kv_cache is not None else 0)
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
        self._prepare_model()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if max_tokens == 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        sampler_top_k = top_k if top_k is not None else self.inference_config.top_k
        sampler_top_p = top_p if top_p is not None else self.inference_config.top_p
        tokens = list(prompt_tokens)
        if not tokens:
            raise ValueError("prompt_tokens must contain at least one token")
        if self.model is None or not callable(self.model):
            raise RuntimeError("No callable model is available for inference")

        if self.inference_config.beam_width > 1:
            from inference.generation.beam_search import BeamSearch
            eos_id = self.inference_config.stop_token_ids[0] if self.inference_config.stop_token_ids else None
            prompt_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
            output = BeamSearch(
                beam_width=self.inference_config.beam_width,
                max_length=max_tokens,
                eos_token_id=eos_id,
            ).search(self.model, prompt_tensor)
            return output[0].tolist()

        forward = getattr(self.model, "forward", None)
        supports_cache = False
        if forward is not None:
            try:
                parameters = inspect.signature(forward).parameters.values()
                supports_cache = any(parameter.name == "kv_cache" for parameter in parameters)
            except (TypeError, ValueError):
                supports_cache = isinstance(self.model, FullTransformerModel)
        use_cache = self.inference_config.use_kv_cache and (
            isinstance(self.model, FullTransformerModel) or supports_cache
        )
        from inference.generation.kv_cache import KVCache
        cache = KVCache(getattr(getattr(self.model, "config", None), "n_layers", 12)) if use_cache else None
        cache_initialized = False
        generator = None
        if self.inference_config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.inference_config.seed)

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

                if cache is not None:
                    model_input = input_tensor if not cache_initialized else input_tensor[:, -1:]
                    logits = self.model(model_input, kv_cache=cache)
                    cache_initialized = True
                else:
                    logits = self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]

                from inference.sampling import Sampler
                sampler = Sampler(
                    temperature=temp,
                    top_k=sampler_top_k,
                    top_p=sampler_top_p,
                    repetition_penalty=self.inference_config.repetition_penalty,
                    generator=generator,
                )
                next_token = sampler.sample(next_token_logits, generated_tokens=tokens).item()

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
        self._failed = False

        # Tokenize prompt
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                prompt_ids = tokenizer.encode(prompt)
            except Exception:
                prompt_ids = [ord(c) for c in prompt]
        else:
            prompt_ids = [ord(c) for c in prompt]
        from tokenizer.encoding import fit_token_ids_to_vocab
        model_vocab_size = getattr(
            getattr(self.model, "config", None), "vocab_size", self.model_config.vocab_size
        )
        prompt_ids = fit_token_ids_to_vocab(prompt_ids, model_vocab_size)

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Decode generated tokens
        if not getattr(self, "_failed", False) and tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                return tokenizer.decode(output_ids[len(prompt_ids):])
            except Exception:
                pass

        # Character fallback is only meaningful for codepoint-based token IDs.
        return "".join(chr(token) for token in output_ids[len(prompt_ids):] if 0 <= token <= 0x10FFFF)
