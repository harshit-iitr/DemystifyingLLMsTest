"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from utils.logging_utils import get_logger
from inference.sampling.sampler import Sampler
from inference.generation.kv_cache import KVCache
from inference.generation.beam_search import BeamSearch

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
        self.num_layers = len(self.blocks)
        self.lm_head = LMHead(config=config)
        if config.tie_weights:
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        offset = kwargs.get("kv_cache").get_seq_length() if kwargs.get("kv_cache") is not None else 0
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
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError(f"max_new_tokens must be a non-negative integer, got {max_tokens!r}")
        if max_tokens == 0:
            return list(prompt_tokens)
        if self.model is None or not callable(self.model):
            raise RuntimeError("No inference model is available")
        if not prompt_tokens:
            raise ValueError("prompt_tokens must contain at least one token")

        if self.inference_config.beam_width > 1:
            input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            eos_token_id = (
                self.inference_config.stop_token_ids[0]
                if self.inference_config.stop_token_ids else None
            )
            with torch.no_grad():
                output = BeamSearch(
                    beam_width=self.inference_config.beam_width,
                    max_length=max_tokens,
                    eos_token_id=eos_token_id,
                ).search(self.model, input_tensor)
            return [int(token_id) for token_id in output[0].tolist()]

        temp = temperature if temperature is not None else self.inference_config.temperature
        sampler = Sampler(
            temperature=temp,
            top_k=top_k if top_k is not None else self.inference_config.top_k,
            top_p=top_p if top_p is not None else self.inference_config.top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        if self.inference_config.seed is not None:
            torch.manual_seed(self.inference_config.seed)
        tokens = list(prompt_tokens)
        cache = None
        if self.inference_config.use_kv_cache and isinstance(self.model, FullTransformerModel):
            cache = KVCache(num_layers=self.model.num_layers)

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                model_tokens = tokens if cache is None or cache.get_seq_length() == 0 else tokens[-1:]
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)

                if self.model is not None and callable(self.model):
                    try:
                        logits = self.model(input_tensor, kv_cache=cache) if cache is not None else self.model(input_tensor)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                        next_token = sampler.sample(next_token_logits, generated_tokens=tokens).item()
                    except Exception as exc:
                        raise RuntimeError("Model inference failed") from exc
                else:
                    raise RuntimeError("No inference model is available")

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
        return " ".join(str(token_id) for token_id in output_ids)
