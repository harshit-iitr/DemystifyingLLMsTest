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

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Step 2: Autoregressive generation
        n_new = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        out_tokens = list(tokens)
        if self.model is not None and callable(self.model) and n_new > 0 and len(tokens) > 0:
            if hasattr(self.model, "eval"):
                self.model.eval()
            stop_ids = set(self.inference_config.stop_token_ids or [])
            with torch.no_grad():
                for _ in range(n_new):
                    input_ids = torch.tensor([out_tokens], dtype=torch.long, device=self.device)
                    logits = self.model(input_ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                    out_tokens.append(next_token)
                    if next_token in stop_ids:
                        break

        # Step 3: Decoding
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(out_tokens)

        if self.model is None:
            return f"{text} [processed]"
        return "".join(chr(t) for t in out_tokens if 0 <= t < 0x110000)
