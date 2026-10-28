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
        self.config = config if config is not None else ModelConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.device = torch.device(device) if device else torch.device("cpu")
        self.engine = InferenceEngine(
            model_config=self.config,
            inference_config=self.inference_config,
            model=self.model,
            device=self.device,
        )

    def process(self, text: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Execute full inference pipeline for an incoming prompt.

        Args:
            text: Input prompt string.
            max_new_tokens: Optional token generation count override.

        Returns:
            Decoded completion string.
        """
        if self.tokenizer is None or not hasattr(self.tokenizer, "encode") or not hasattr(self.tokenizer, "decode"):
            raise ValueError("A tokenizer with encode and decode methods is required for serving")
        tokens = self.tokenizer.encode(text)
        output_tokens = self.engine.generate_tokens(tokens, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_tokens)
