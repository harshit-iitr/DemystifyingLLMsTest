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
        if getattr(config, "tie_weights", False):
            from model.heads.weight_tying import tie_weights
            tie_weights(self.lm_head, self.embeddings.token_embedding)

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
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> List[int]:
        """
        Generate a list of token IDs autoregressively from input token sequence.
        """
        self._prepare_model()

        if isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = prompt_tokens.squeeze().tolist()
            if isinstance(prompt_tokens, int):
                prompt_tokens = [prompt_tokens]

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens <= 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        top_k = top_k if top_k is not None else self.inference_config.top_k
        top_p = top_p if top_p is not None else self.inference_config.top_p
        rep_penalty = kwargs.get("repetition_penalty", self.inference_config.repetition_penalty)

        stop_ids = list(self.inference_config.stop_token_ids or [])
        if "stop_token_ids" in kwargs and kwargs["stop_token_ids"] is not None:
            extra = kwargs["stop_token_ids"]
            stop_ids.extend(extra if isinstance(extra, (list, tuple, set)) else [extra])
        if "eos_token_id" in kwargs and kwargs["eos_token_id"] is not None:
            stop_ids.append(kwargs["eos_token_id"])

        from inference.sampling import Sampler
        sampler = Sampler(
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_penalty,
        )
        tokens = list(prompt_tokens)
        if not tokens:
            from config.constants import BOS_TOKEN_ID
            tokens = [BOS_TOKEN_ID]

        # Autoregressive generation loop
        with torch.no_grad():
            cache = None
            if self.inference_config.use_kv_cache:
                from inference.generation import KVCache
                cache = KVCache(num_layers=self.model_config.n_layers)
            for _ in range(max_tokens):
                model_tokens = tokens if cache is None or cache.get_seq_length(0) == 0 else tokens[-1:]
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)

                if self.model is not None and callable(self.model):
                    try:
                        if cache is not None:
                            try:
                                logits = self.model(input_tensor, kv_cache=cache)
                            except TypeError as error:
                                if "kv_cache" not in str(error):
                                    raise
                                cache = None
                                input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
                                logits = self.model(input_tensor)
                        else:
                            logits = self.model(input_tensor)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        next_token_logits = logits[:, -1, :]  # (1, vocab_size)
                        sampled = sampler.sample(next_token_logits, generated_tokens=tokens)
                        next_token = sampled[0].item()
                    except Exception:
                        self._failed = True
                        raise
                else:
                    self._failed = True
                    raise RuntimeError("No callable model is available for generation")

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
        self._prepare_model()
        self._failed = False

        # Tokenize prompt
        vocab_sz = getattr(self.model_config, "vocab_size", 50257)
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                prompt_ids = tokenizer.encode(prompt)
            except Exception:
                prompt_ids = [ord(c) % vocab_sz for c in prompt]
        else:
            prompt_ids = [ord(c) % vocab_sz for c in prompt]

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
        return "".join(chr(token) for token in output_ids if 0 <= token < 0x110000)
