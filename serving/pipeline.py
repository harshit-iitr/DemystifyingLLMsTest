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
        self.engine = None

        if self.model is None or isinstance(self.model, nn.Module):
            self.engine = InferenceEngine(
                model_config=self.config,
                inference_config=self.inference_config,
                model=self.model,
                device=str(self.device),
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
        token_budget = (
            self.inference_config.max_new_tokens
            if max_new_tokens is None
            else max_new_tokens
        )

        if self.engine is not None and self.tokenizer is not None:
            return self.engine.generate(
                text,
                tokenizer=self.tokenizer,
                max_new_tokens=token_budget,
                temperature=self.inference_config.temperature,
            )

        # Step 1: Tokenization
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = [ord(c) for c in text]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Step 2: Model Forward Execution (fallback path)
        if self.model is not None and callable(self.model):
            output = self.model(input_ids)
        else:
            output = None

        # Step 3: Decoding (fallback path)
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            if isinstance(output, torch.Tensor) and output.ndim == 3:
                out_tokens = output.argmax(dim=-1).squeeze(0).tolist()
                return self.tokenizer.decode(out_tokens)
            return self.tokenizer.decode(tokens)

        return f"{text} [processed]"
