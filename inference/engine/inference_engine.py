"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from utils.logging_utils import get_logger
from inference.sampling import Sampler

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
        if max_tokens <= 0:
            return list(prompt_tokens)
        if not prompt_tokens:
            raise ValueError("prompt_tokens must contain at least one token")

        temp = temperature if temperature is not None else self.inference_config.temperature
        sampler = Sampler(
            temperature=temp,
            top_k=self.inference_config.top_k if top_k is None else top_k,
            top_p=self.inference_config.top_p if top_p is None else top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        generator = None
        if self.inference_config.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.inference_config.seed)
        tokens = list(prompt_tokens)
        use_cache = self.inference_config.use_kv_cache and isinstance(self.model, FullTransformerModel)
        kv_cache = None
        if use_cache:
            from inference.generation import KVCache
            kv_cache = KVCache(self.model.num_layers)

        # Autoregressive generation loop
        with torch.no_grad():
            for step in range(max_tokens):
                input_tokens = tokens if kv_cache is None or step == 0 else tokens[-1:]
                input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=self.device)

                if self.model is not None and callable(self.model):
                    try:
                        logits = self.model(input_tensor, **({"kv_cache": kv_cache} if kv_cache is not None else {}))
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                        next_token = sampler.sample(next_token_logits, tokens, generator=generator).item()
                    except Exception:
                        self._failed = True
                        raise
                else:
                    self._failed = True
                    raise RuntimeError("No model is available for token generation")

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
                prompt_ids = list(prompt.encode("utf-8"))
        else:
            prompt_ids = list(prompt.encode("utf-8"))

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Decode generated tokens
        if not getattr(self, "_failed", False) and tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                return tokenizer.decode(output_ids)
            except Exception:
                pass

        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)
        return bytes(token for token in output_ids if 0 <= token <= 255).decode("utf-8", errors="replace")
