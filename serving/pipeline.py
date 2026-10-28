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
        if self.model is None:
            raise RuntimeError("A model is required to process a completion")
        if self.tokenizer is None or not hasattr(self.tokenizer, "encode") or not hasattr(self.tokenizer, "decode"):
            raise RuntimeError("A tokenizer with encode() and decode() is required")

        token_ids = self.tokenizer.encode(text)
        vocab_size = getattr(self.config, "vocab_size", None)
        if vocab_size is not None:
            vocab = getattr(self.tokenizer, "vocab", None)
            unknown_id = getattr(vocab, "unk_token_id", 1)
            if not 0 <= unknown_id < vocab_size:
                unknown_id = 0
            token_ids = [
                token_id if 0 <= token_id < vocab_size else unknown_id
                for token_id in token_ids
            ]
        input_ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        self.model.to(self.device) if hasattr(self.model, "to") else None
        if hasattr(self.model, "eval"):
            self.model.eval()

        with torch.no_grad():
            output = self.model(input_ids)
            if max_new_tokens is not None and max_new_tokens > 0:
                generated = input_ids
                for _ in range(max_new_tokens):
                    logits = output[0] if isinstance(output, tuple) else output
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat((generated, next_token), dim=-1)
                    output = self.model(generated)
                output_ids = generated[0].tolist()
            else:
                logits = output[0] if isinstance(output, tuple) else output
                output_ids = logits.argmax(dim=-1)[0].tolist()
        return self.tokenizer.decode(output_ids)
