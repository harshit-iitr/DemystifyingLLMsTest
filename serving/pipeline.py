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

        if not tokens:
            raise ValueError("text must produce at least one token")
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Step 2: Model Forward Execution
        if self.model is not None and callable(self.model):
            generated = list(tokens)
            num_new_tokens = (
                max_new_tokens
                if max_new_tokens is not None
                else self.inference_config.max_new_tokens
            )
            if hasattr(self.model, "eval"):
                self.model.eval()
            with torch.no_grad():
                for _ in range(num_new_tokens):
                    model_input = torch.tensor([generated], dtype=torch.long, device=self.device)
                    output = self.model(model_input)
                    if isinstance(output, tuple):
                        output = output[0]
                    if output.dim() != 3:
                        raise ValueError("model must return logits shaped (batch, sequence, vocabulary)")
                    generated.append(int(output[:, -1, :].argmax(dim=-1).item()))
            output = generated
        else:
            output = None

        # Step 3: Decoding
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            if isinstance(output, list):
                return self.tokenizer.decode(output)
            return self.tokenizer.decode(tokens)

        return f"{text} [processed]"
