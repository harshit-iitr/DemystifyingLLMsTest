"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import math
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from inference.generation.kv_cache import KVCache


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
        self._initialize_weights(config.initializer_range)
        if getattr(config, "tie_weights", False):
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def _initialize_weights(self, std: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

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
        self.model_config = model_config if model_config is not None else ModelConfig()

        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device) if device else torch.device("cpu")
        self.generator = torch.Generator(device=self.device)
        if self.inference_config.seed is not None:
            self.generator.manual_seed(self.inference_config.seed)

        self.model = model if model is not None else FullTransformerModel(self.model_config)

    def _prepare_model(self) -> None:
        """
        Internal lifecycle helper to prepare model state before inference execution.
        """
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(self.device)

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
            raise ValueError("max_new_tokens must be non-negative")
        if max_tokens == 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        top_k = top_k if top_k is not None else self.inference_config.top_k
        top_p = top_p if top_p is not None else self.inference_config.top_p
        if not math.isfinite(temp) or temp < 0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not math.isfinite(top_p) or not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be in [0.0, 1.0]")
        tokens = list(prompt_tokens)
        if not tokens:
            from config.constants import BOS_TOKEN_ID
            tokens = [BOS_TOKEN_ID]

        if self.inference_config.beam_width > 1:
            from inference.generation import BeamSearch
            from config.constants import EOS_TOKEN_ID
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
            with torch.no_grad():
                result = BeamSearch(
                    beam_width=self.inference_config.beam_width,
                    max_length=max_tokens,
                    eos_token_id=EOS_TOKEN_ID,
                    early_stopping=self.inference_config.early_stopping,
                ).search(self.model, input_tensor)
            return result[0].tolist()

        # Autoregressive generation loop
        supports_cache = (
            self.inference_config.use_kv_cache
            and isinstance(self.model, FullTransformerModel)
        )
        kv_cache = KVCache(self.model.blocks.n_layers) if supports_cache else None
        from inference.sampling import Sampler
        from config.constants import EOS_TOKEN_ID
        stop_ids = self.inference_config.stop_token_ids or [EOS_TOKEN_ID]
        with torch.no_grad():
            for _ in range(max_tokens):
                model_tokens = tokens if kv_cache is None or kv_cache.get_seq_length(0) == 0 else tokens[-1:]
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)

                if self.model is None or not callable(self.model):
                    raise RuntimeError("No model is configured for inference")
                logits = self.model(input_tensor, kv_cache=kv_cache) if kv_cache is not None else self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]

                sampler = Sampler(
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=self.inference_config.repetition_penalty,
                )
                next_token = sampler.sample(
                    next_token_logits,
                    generated_tokens=tokens,
                    generator=self.generator,
                ).item()

                tokens.append(int(next_token))

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

        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)
        from config.constants import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID
        special_ids = {BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID}
        return "".join(
            chr(token) for token in output_ids
            if token not in special_ids and 0 <= token <= 0x10FFFF and not 0xD800 <= token <= 0xDFFF
        )
