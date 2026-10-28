"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from inference.generation.kv_cache import KVCache
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
        if getattr(config, "tie_weights", False):
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        kv_cache = kwargs.get("kv_cache")
        offset = kv_cache.get_seq_length() if kv_cache is not None else 0
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
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> List[int]:
        """
        Generate a list of token IDs autoregressively from input token sequence.
        """
        self._prepare_model()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens <= 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        k = top_k if top_k is not None else self.inference_config.top_k
        p = top_p if top_p is not None else self.inference_config.top_p
        penalty = repetition_penalty if repetition_penalty is not None else self.inference_config.repetition_penalty
        tokens = list(prompt_tokens)
        if not tokens:
            return []
        if self.model is None or not callable(self.model):
            raise RuntimeError("InferenceEngine has no callable model")
        if temp < 0 or not torch.isfinite(torch.tensor(float(temp))).item():
            raise ValueError(f"temperature must be non-negative and finite, got {temp}")
        if penalty <= 0 or not torch.isfinite(torch.tensor(float(penalty))).item():
            raise ValueError(f"repetition_penalty must be positive and finite, got {penalty}")
        if p is not None and (not torch.isfinite(torch.tensor(float(p))).item() or p < 0 or p > 1):
            raise ValueError(f"top_p must be in [0, 1], got {p}")

        generator = None
        if self.inference_config.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.inference_config.seed)

        # Autoregressive generation loop
        supports_cache = self.inference_config.use_kv_cache and isinstance(self.model, FullTransformerModel)
        kv_cache = KVCache(self.model_config.n_layers) if supports_cache else None
        with torch.no_grad():
            for _ in range(max_tokens):
                model_tokens = tokens[-1:] if kv_cache is not None and kv_cache.get_seq_length() > 0 else tokens
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)

                logits = self.model(input_tensor, kv_cache=kv_cache) if kv_cache is not None else self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                    raise ValueError("model must return logits with shape (batch_size, seq_len, vocab_size)")
                next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                if penalty != 1.0:
                    from inference.sampling.repetition_penalty import apply_repetition_penalty
                    next_token_logits = apply_repetition_penalty(
                        next_token_logits, tokens, penalty=penalty
                    )
                if k is not None and k > 0:
                    from inference.sampling.top_k import apply_top_k
                    next_token_logits = apply_top_k(next_token_logits, k)
                if p is not None and p < 1.0:
                    from inference.sampling.top_p import apply_top_p
                    next_token_logits = apply_top_p(next_token_logits, p)

                if temp > 0:
                    probs = torch.softmax(next_token_logits / temp, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1, generator=generator).item()
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).item()

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

        # Decode the actual generated sequence; never substitute synthetic output.
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)
        return "".join(chr(token) for token in output_ids if 0 <= token <= 0x10FFFF)
