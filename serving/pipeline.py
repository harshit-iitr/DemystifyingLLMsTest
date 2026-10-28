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
        self.config = config or getattr(model, "config", None) or ModelConfig()
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
        if self.model is None:
            raise RuntimeError("A model is required for serving")
        if max_new_tokens is not None:
            from inference.engine import InferenceEngine
            engine = InferenceEngine(model_config=self.config, inference_config=self.inference_config,
                                     model=self.model, device=self.device)
            return engine.generate(text, tokenizer=self.tokenizer, max_new_tokens=max_new_tokens)

        # Step 1: Tokenization
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = [ord(c) for c in text]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Step 2: Model Forward Execution
        if self.model is not None and callable(self.model):
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            with torch.no_grad():
                output = self.model(input_ids)
                if isinstance(output, tuple):
                    output = output[0]
        else:
            output = None

        # Step 3: Decoding
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            if isinstance(output, torch.Tensor):
                out_tokens = output.argmax(dim=-1).squeeze(0).tolist()
                return self.tokenizer.decode(out_tokens)
            raise TypeError("Model must return vocabulary logits")

        if isinstance(output, torch.Tensor):
            return "".join(chr(i) for i in output.argmax(dim=-1).squeeze(0).tolist())
        raise TypeError("Model must return vocabulary logits")
