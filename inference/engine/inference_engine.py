"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

import inspect
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
        if getattr(config, "tie_weights", False):
            from model.heads import tie_weights
            tie_weights(self.lm_head, self.embeddings.token_embedding)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        kv_cache = kwargs.get("kv_cache")
        mask = kwargs.get("mask")
        if mask is None:
            mask = kwargs.get("attention_mask")
        offset = kv_cache.get_seq_length(0) if kv_cache is not None else 0
        h = self.embeddings(input_ids, offset=offset)
        h = self.blocks(h, mask=mask, kv_cache=kv_cache)
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
        self.model_config = model_config or ModelConfig()

        self.inference_config = inference_config or InferenceConfig()
        self.device = torch.device(device) if device else torch.device("cpu")

        if model is not None:
            self.model = model
        else:
            self.model = FullTransformerModel(self.model_config)

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
        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens <= 0:
            return list(prompt_tokens)
        if not prompt_tokens:
            raise ValueError("prompt_tokens must contain at least one token")
        if self.model is None:
            raise RuntimeError("No inference model is available")

        self._prepare_model()
        tokens = torch.tensor([list(prompt_tokens)], dtype=torch.long, device=self.device)

        if self.inference_config.beam_width > 1:
            from inference.generation import BeamSearch
            eos_token_id = self.inference_config.stop_token_ids[0] if self.inference_config.stop_token_ids else None
            beam_search = BeamSearch(
                beam_width=self.inference_config.beam_width,
                max_length=max_tokens,
                eos_token_id=eos_token_id,
            )
            with torch.no_grad():
                return beam_search.search(self.model, tokens)[0].tolist()

        temp = temperature if temperature is not None else self.inference_config.temperature
        sample_top_k = top_k if top_k is not None else self.inference_config.top_k
        sample_top_p = top_p if top_p is not None else self.inference_config.top_p
        from inference.sampling import Sampler
        sampler = Sampler(
            temperature=temp,
            top_k=sample_top_k,
            top_p=sample_top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )

        supports_cache = False
        try:
            parameters = inspect.signature(self.model.forward).parameters.values()
            supports_cache = any(
                parameter.name == "kv_cache" or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (AttributeError, TypeError, ValueError):
            pass
        use_cache = self.inference_config.use_kv_cache and supports_cache
        kv_cache = None
        if use_cache:
            from inference.generation import KVCache
            num_layers = getattr(self.model_config, "n_layers", 12)
            kv_cache = KVCache(num_layers=num_layers)

        if self.inference_config.seed is not None:
            torch.manual_seed(self.inference_config.seed)

        # Autoregressive generation loop
        with torch.no_grad():
            for step in range(max_tokens):
                model_input = tokens if not use_cache or step == 0 else tokens[:, -1:]
                logits = self.model(model_input, kv_cache=kv_cache) if use_cache else self.model(model_input)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_logits = logits[:, -1, :]
                next_ids = sampler.sample(next_logits, generated_tokens=tokens)
                next_token = next_ids.reshape(-1, 1)
                tokens = torch.cat((tokens, next_token), dim=-1)

                if self.inference_config.stop_token_ids and all(
                    int(token_id) in self.inference_config.stop_token_ids
                    for token_id in next_token.reshape(-1).tolist()
                ):
                    break

        return tokens[0].tolist()

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
            prompt_ids = list(prompt.encode("utf-8"))

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Decode generated tokens
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)

        return bytes(int(token_id) % 256 for token_id in output_ids).decode("utf-8", errors="replace")
