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

    def _prepare_model(self) -> None:
        """
        Internal lifecycle helper to prepare model state before inference execution.
        """
        if self.model is not None:
            if hasattr(self.model, "eval"):
                self.model.eval()
            if hasattr(self.model, "to"):
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
        if self.model is not None:
            if hasattr(self.model, "eval"):
                self.model.eval()
            if hasattr(self.model, "to"):
                self.model.to(self.device)

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens <= 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        top_k = self.inference_config.top_k if top_k is None else top_k
        top_p = self.inference_config.top_p if top_p is None else top_p
        tokens = list(prompt_tokens)
        if self.inference_config.seed is not None:
            torch.manual_seed(self.inference_config.seed)

        forward = getattr(self.model, "forward", None)
        supports_cache = False
        if forward is not None:
            try:
                parameters = inspect.signature(forward).parameters.values()
                supports_cache = any(
                    parameter.name == "kv_cache" or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                pass
        use_cache = self.inference_config.use_kv_cache and supports_cache
        cache = None
        if use_cache:
            from inference.generation import KVCache
            config = getattr(self.model, "config", self.model_config)
            cache = KVCache(num_layers=getattr(config, "n_layers", 12))

        # Autoregressive generation loop
        with torch.no_grad():
            for step in range(max_tokens):
                model_tokens = tokens if cache is None or step == 0 else tokens[-1:]
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)

                if self.model is not None and callable(self.model):
                    try:
                        logits = self.model(input_tensor, **({"kv_cache": cache} if cache is not None else {}))
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                        if temp <= 0:
                            next_token = torch.argmax(next_token_logits, dim=-1).item()
                        else:
                            from inference.sampling import Sampler
                            sampler = Sampler(
                                temperature=temp,
                                top_k=top_k,
                                top_p=top_p,
                                repetition_penalty=self.inference_config.repetition_penalty,
                            )
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

        # Tokenize prompt
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = [ord(c) for c in prompt]
        if not prompt_ids:
            prompt_ids = [self.model_config.bos_token_id if hasattr(self.model_config, "bos_token_id") else 50256]

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Decode generated tokens
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)
        return " ".join(str(token_id) for token_id in output_ids)
