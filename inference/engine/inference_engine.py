"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
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
        if config.tie_weights:
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)
        self.num_layers = config.n_layers

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
        self.model_config = model_config if model_config is not None else ModelConfig()

        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device) if device else torch.device("cpu")

        self.model = model if model is not None else FullTransformerModel(self.model_config)

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
        if max_tokens == 0:
            return list(prompt_tokens)
        if not prompt_tokens:
            raise ValueError("prompt_tokens must contain at least one token")

        temp = temperature if temperature is not None else self.inference_config.temperature
        if temp < 0:
            raise ValueError(f"temperature must be non-negative, got {temp}")
        if self.inference_config.seed is not None:
            torch.manual_seed(self.inference_config.seed)
        tokens = list(prompt_tokens)

        if self.inference_config.beam_width > 1:
            from inference.generation.beam_search import BeamSearch
            prompt_window = tokens[-self.model_config.max_seq_len:]
            input_tensor = torch.tensor([prompt_window], dtype=torch.long, device=self.device)
            result = BeamSearch(
                beam_width=self.inference_config.beam_width,
                max_length=max_tokens,
                eos_token_ids=self.inference_config.stop_token_ids,
                max_context_length=self.model_config.max_seq_len,
            ).search(self.model, input_tensor)
            prefix = tokens[:-len(prompt_window)] if prompt_window else tokens
            return prefix + result[0].tolist()

        use_cache = self.inference_config.use_kv_cache and isinstance(self.model, FullTransformerModel)
        kv_cache = None
        if use_cache:
            from inference.generation.kv_cache import KVCache
            kv_cache = KVCache(num_layers=self.model_config.n_layers)

        from inference.sampling import Sampler
        sampler = Sampler(
            temperature=temp,
            top_k=self.inference_config.top_k if top_k is None else top_k,
            top_p=self.inference_config.top_p if top_p is None else top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                model_tokens = tokens
                if kv_cache is not None and kv_cache.get_seq_length() > 0:
                    if kv_cache.get_seq_length() >= self.model_config.max_seq_len:
                        kv_cache.reset()
                        model_tokens = tokens[-self.model_config.max_seq_len:]
                    else:
                        model_tokens = tokens[-1:]
                elif kv_cache is None:
                    model_tokens = tokens[-self.model_config.max_seq_len:]
                else:
                    model_tokens = tokens[-self.model_config.max_seq_len:]
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)

                if self.model is None or not callable(self.model):
                    raise RuntimeError("InferenceEngine has no callable model")
                logits = self.model(input_tensor, kv_cache=kv_cache) if kv_cache is not None else self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]
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

        # Decode generated tokens
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)

        return "".join(
            chr(token) for token in output_ids
            if 0 <= token <= 0x10FFFF and not 0xD800 <= token <= 0xDFFF
        )
