"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from config.constants import EOS_TOKEN_ID
from inference.sampling import Sampler
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
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def forward(self, input_ids: torch.Tensor, kv_cache: Optional[Any] = None, **kwargs) -> torch.Tensor:
        # With a KV cache, new tokens sit after the cached positions
        offset = kv_cache.get_seq_length(0) if kv_cache is not None else 0
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
                logger.warning(f"Could not build FullTransformerModel: {e}")
                self.model = None

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

        if self.model is None or not callable(self.model):
            raise RuntimeError("InferenceEngine has no usable model; provide one or fix model construction.")

        cfg = self.inference_config
        sampler = Sampler(
            temperature=temperature if temperature is not None else cfg.temperature,
            top_k=top_k if top_k is not None else cfg.top_k,
            top_p=top_p if top_p is not None else cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
        )
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        stop_ids = set(cfg.stop_token_ids) | {EOS_TOKEN_ID}
        tokens = list(prompt_tokens)
        max_len = getattr(self.model_config, "max_seq_len", None)

        with torch.no_grad():
            for _ in range(max_tokens):
                # Respect the model's context window
                context = tokens[-max_len:] if max_len else tokens
                input_tensor = torch.tensor([context], dtype=torch.long, device=self.device)

                logits = self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                next_token = int(sampler.sample(next_token_logits, generated_tokens=tokens).item())
                tokens.append(next_token)

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

        # Tokenize prompt
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = [ord(c) for c in prompt]

        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Decode generated tokens
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)
        return "".join(chr(t) if 0 <= t < 0x110000 else "?" for t in output_ids)
