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
        self.num_layers = config.n_layers
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead

        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        if config.tie_weights:
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def forward(self, input_ids: torch.Tensor, position_offset: int = 0, **kwargs) -> torch.Tensor:
        h = self.embeddings(input_ids, offset=position_offset)
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
        self.model_config = model_config or ModelConfig()

        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
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
        if self.model is not None and hasattr(self.model, "train"):
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
        self._prepare_model()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens <= 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        k = top_k if top_k is not None else self.inference_config.top_k
        p = top_p if top_p is not None else self.inference_config.top_p
        tokens = list(prompt_tokens)
        if not tokens:
            raise ValueError("prompt_tokens must contain at least one token")
        unk_id = 1 if self.model_config.vocab_size > 1 else 0
        tokens = [
            token_id if 0 <= token_id < self.model_config.vocab_size else unk_id
            for token_id in tokens
        ]

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

                if self.model is not None and callable(self.model):
                    logits = self.model(input_tensor)
                    next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                    from inference.sampling.top_k import apply_top_k
                    from inference.sampling.top_p import apply_top_p
                    if k > 0:
                        next_token_logits = apply_top_k(next_token_logits, k)
                    if p < 1.0:
                        next_token_logits = apply_top_p(next_token_logits, p)

                    if temp > 0:
                        probs = torch.softmax(next_token_logits / temp, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1).item()
                else:
                    raise RuntimeError("No model is available for token generation")

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
        if tokenizer is None or not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise RuntimeError("A tokenizer with encode() and decode() is required")
        prompt_ids = tokenizer.encode(prompt)

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Decode generated tokens
        return tokenizer.decode(output_ids)
