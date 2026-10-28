"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union, List
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from inference.engine.inference_engine import InferenceEngine
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
        if self.model is not None and hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

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
        if self.tokenizer is None or not hasattr(self.tokenizer, "encode") or not hasattr(self.tokenizer, "decode"):
            raise RuntimeError("Serving requires a tokenizer with encode and decode methods")
        tokens = self.tokenizer.encode(text)

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        if self.model is None or not callable(self.model):
            raise RuntimeError("A model is required to produce a serving response")
        engine = InferenceEngine(
            model_config=self.config,
            inference_config=self.inference_config,
            model=self.model,
            device=self.device,
        )
        generated_tokens = engine.generate_tokens(
            prompt_tokens=tokens,
            max_new_tokens=max_new_tokens,
        )
        continuation = generated_tokens[len(tokens):]
        return self.tokenizer.decode(continuation)
