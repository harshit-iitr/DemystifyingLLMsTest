"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union
import torch

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
        self.config = config or ModelConfig()
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
        if self.tokenizer is None or not hasattr(self.tokenizer, "encode") or not hasattr(self.tokenizer, "decode"):
            raise ValueError("A tokenizer with encode and decode methods is required")
        if self.model is None:
            raise RuntimeError("No model is configured for the serving pipeline")

        engine = InferenceEngine(
            model_config=self.config,
            inference_config=self.inference_config,
            model=self.model,
            device=self.device
        )
        return engine.generate(
            text,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens
        )
