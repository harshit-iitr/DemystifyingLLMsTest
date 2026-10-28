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
        if getattr(config, "tie_weights", False):
            from model.heads import tie_weights
            tie_weights(self.lm_head, self.embeddings.token_embedding)

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
        self.device = torch.device(device) if device else torch.device("cpu")

        if model is not None:
            self.model = model
        else:
            try:
                self.model = FullTransformerModel(self.model_config)
            except Exception as e:
                logger.debug(f"Could not build FullTransformerModel: {e}")
                self.model = None
        if isinstance(self.model, nn.Module):
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

        temp = temperature if temperature is not None else self.inference_config.temperature
        top_k = top_k if top_k is not None else self.inference_config.top_k
        top_p = top_p if top_p is not None else self.inference_config.top_p
        tokens = list(prompt_tokens)
        kv_cache = None
        forward = getattr(self.model, "forward", None)
        supports_cache = False
        if callable(forward):
            try:
                parameters = inspect.signature(forward).parameters.values()
                supports_cache = any(
                    p.name == "kv_cache" or p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in parameters
                )
            except (TypeError, ValueError):
                supports_cache = isinstance(self.model, FullTransformerModel)
        if self.inference_config.use_kv_cache and supports_cache:
            from inference.generation.kv_cache import KVCache
            n_layers = getattr(self.model, "num_layers", getattr(self.model_config, "n_layers", 12))
            kv_cache = KVCache(num_layers=n_layers)

        # Autoregressive generation loop
        with torch.no_grad():
            from inference.sampling import Sampler
            sampler = Sampler(temperature=temp, top_k=top_k, top_p=top_p, repetition_penalty=self.inference_config.repetition_penalty)
            for _ in range(max_tokens):
                model_tokens = tokens if kv_cache is None or kv_cache.get_seq_length() == 0 else tokens[-1:]
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)

                if self.model is not None and callable(self.model):
                    try:
                        logits = self.model(input_tensor, kv_cache=kv_cache) if kv_cache is not None else self.model(input_tensor)
                        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                        next_token = sampler.sample(next_token_logits, generated_tokens=tokens).item()
                    except Exception:
                        self._failed = True
                        raise
                else:
                    self._failed = True
                    raise RuntimeError("Inference model is unavailable")

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

        if tokenizer is None or not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise ValueError("generate() requires a tokenizer with encode() and decode()")
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("Tokenization produced an empty prompt")

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        if self._failed:
            raise RuntimeError("Text generation failed; inspect the model error above")
        if tokenizer is None or not hasattr(tokenizer, "decode"):
            raise ValueError("A tokenizer with decode() is required to return generated text")
        return tokenizer.decode(output_ids[len(prompt_ids):])
