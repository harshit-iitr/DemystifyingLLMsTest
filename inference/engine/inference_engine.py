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
        from model.heads import LMHead, tie_weights

        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        self.num_layers = config.n_layers
        if config.tie_weights:
            tie_weights(self.lm_head, self.embeddings.token_embedding)

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
        if max_tokens <= 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        selected_top_k = top_k if top_k is not None else self.inference_config.top_k
        selected_top_p = top_p if top_p is not None else self.inference_config.top_p
        tokens = list(prompt_tokens)

        if not tokens:
            raise ValueError("prompt_tokens must contain at least one token")
        if self.model is None or not callable(self.model):
            raise RuntimeError("No callable model is available for generation")

        from inference.sampling import Sampler
        sampler = Sampler(temperature=temp, top_k=selected_top_k, top_p=selected_top_p,
                          repetition_penalty=self.inference_config.repetition_penalty)

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

                logits = self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]
                if temp == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1).item()
                else:
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

        # When no tokenizer is provided, preserve the actual generated IDs in a
        # readable form rather than fabricating a placeholder continuation.
        return "".join(
            chr(token_id) if 0 <= token_id <= 0x10FFFF else f"<token_{token_id}>"
            for token_id in output_ids
        )
