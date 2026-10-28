"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union
import inspect
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from utils.logging_utils import get_logger
from inference.generation import autoregressive_generate
from inference.sampling import Sampler
from config.constants import EOS_TOKEN_ID

logger = get_logger("serving_pipeline")


class ServingPipeline:
    """
    End-to-end request processor: text -> tokenizer -> model -> decoding -> response.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.config = config or ModelConfig()
        except Exception:
            self.config = ModelConfig(d_model=768, vocab_size=50257)
        self.inference_config = inference_config or InferenceConfig()
        self.device = torch.device(device) if device else torch.device("cpu")

    def process(self, text: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Execute full inference pipeline for an incoming prompt.

        Args:
            text: Input prompt string.
            max_new_tokens: Optional token generation count override.

        Returns:
            Decoded completion string.
        """
        # Step 1: Tokenization
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = list(text.encode("utf-8"))

        if self.model is None or not callable(self.model):
            raise RuntimeError("A callable model is required to serve completions")

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        sampler = Sampler(
            temperature=self.inference_config.temperature,
            top_k=self.inference_config.top_k,
            top_p=self.inference_config.top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        self.model.eval()
        forward_method = getattr(self.model, "forward", self.model)
        try:
            forward_parameters = inspect.signature(forward_method).parameters
            supports_cache = "kv_cache" in forward_parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_parameters.values()
            )
        except (TypeError, ValueError):
            supports_cache = False
        with torch.no_grad():
            output_ids = autoregressive_generate(
                self.model,
                input_ids,
                max_new_tokens=(max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens),
                eos_token_id=EOS_TOKEN_ID,
                sampler_fn=(lambda logits: sampler.sample(logits, tokens)) if self.inference_config.temperature > 0 else None,
                use_cache=self.inference_config.use_kv_cache and supports_cache,
            )

        result = output_ids[0].tolist()
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(result)
        return bytes(token for token in result if 0 <= token <= 255).decode("utf-8", errors="replace")
