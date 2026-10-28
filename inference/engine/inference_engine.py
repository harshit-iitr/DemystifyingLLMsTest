"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from config.constants import EOS_TOKEN_ID
from inference.generation.kv_cache import KVCache
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

    def forward(self, input_ids: torch.Tensor, kv_cache: Optional[Any] = None, **kwargs) -> torch.Tensor:
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
            self.model = model.to(self.device) if hasattr(model, "to") else model
        else:
            try:
                self.model = FullTransformerModel(self.model_config).to(self.device)
            except Exception as e:
                logger.debug(f"Could not build FullTransformerModel: {e}")
                self.model = None

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
        self.last_finish_reason = "length"

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens <= 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        top_k = self.inference_config.top_k if top_k is None else top_k
        top_p = self.inference_config.top_p if top_p is None else top_p
        repetition_penalty = self.inference_config.repetition_penalty
        tokens = list(prompt_tokens)
        use_cache = self.inference_config.use_kv_cache and hasattr(self.model, "config")
        kv_cache = KVCache(num_layers=self.model.config.n_layers) if use_cache else None

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                if len(tokens) > self.model_config.max_seq_len:
                    raise ValueError(
                        f"Prompt plus generated tokens exceed max_seq_len {self.model_config.max_seq_len}"
                    )
                model_tokens = tokens if kv_cache is None or kv_cache.get_seq_length() == 0 else tokens[-1:]
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)

                if self.model is not None and callable(self.model):
                    try:
                        logits = self.model(input_tensor, kv_cache=kv_cache) if kv_cache is not None else self.model(input_tensor)
                        next_token_logits = logits[:, -1, :]  # (1, vocab_size)
                        from inference.sampling import Sampler
                        sampler = Sampler(temperature=temp, top_k=top_k, top_p=top_p,
                                          repetition_penalty=repetition_penalty)
                        sampled = sampler.sample(next_token_logits, tokens)
                        next_token = sampled.item()
                    except Exception as exc:
                        self._failed = True
                        raise RuntimeError("Model inference failed while generating tokens") from exc
                else:
                    self._failed = True
                    raise RuntimeError("Cannot generate tokens because no model is available")

                tokens.append(int(next_token))

                if next_token in self.inference_config.stop_token_ids:
                    self.last_finish_reason = "stop"
                    break
                if next_token == EOS_TOKEN_ID:
                    self.last_finish_reason = "stop"
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

        if self._failed:
            raise RuntimeError("Generation failed; no decoded model output is available")
        raise RuntimeError("A tokenizer with a decode method is required to return generated text")
