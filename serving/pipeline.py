"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union, List
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from utils.logging_utils import get_logger

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
            tokens = [ord(c) for c in text]

        if self.tokenizer is None or not hasattr(self.tokenizer, "decode"):
            raise ValueError("A tokenizer with encode() and decode() is required")
        if self.model is None or not callable(self.model):
            raise RuntimeError("No callable model is available")

        input_ids = torch.as_tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        if max_new_tokens is None:
            max_new_tokens = self.inference_config.max_new_tokens
        if max_new_tokens > 0:
            from inference.generation.autoregressive import autoregressive_generate
            output_ids = autoregressive_generate(
                self.model, input_ids, max_new_tokens=max_new_tokens,
                eos_token_id=getattr(getattr(self.tokenizer, "vocab", None), "eos_token_id", None),
                use_cache=self.inference_config.use_kv_cache,
            )
        else:
            output_ids = input_ids
        return self.tokenizer.decode(output_ids[0].tolist())
