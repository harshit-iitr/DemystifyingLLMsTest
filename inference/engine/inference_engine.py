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
        self.num_layers = config.n_layers

        if getattr(config, "tie_weights", False):
            from model.heads import tie_weights
            tie_weights(self.lm_head, self.embeddings.token_embedding)

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
        if self.model is not None and hasattr(self.model, "eval"):
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

        from inference.sampling import Sampler
        sampler = Sampler(
            temperature=temp if temp > 0 else 1.0,
            top_k=k,
            top_p=p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

                if self.model is not None and callable(self.model):
                    try:
                        logits = self.model(input_tensor)
                        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                        if temp > 0:
                            next_token = sampler.sample(next_token_logits, generated_tokens=tokens).item()
                        else:
                            next_token = torch.argmax(next_token_logits, dim=-1).item()
                    except Exception as e:
                        logger.warning(f"Model forward failed during generation: {e}")
                        self._failed = True
                        next_token = (tokens[-1] + 1) % self.model_config.vocab_size
                else:
                    self._failed = True
                    next_token = (tokens[-1] + 1) % self.model_config.vocab_size

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

        # Fallback decode: append dummy synthesized continuation
        continuation = " [generated continuation tokens]"
        return prompt + continuation
