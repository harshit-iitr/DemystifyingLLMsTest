"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union, List
import torch
import torch.nn as nn
from inference.generation.autoregressive import autoregressive_generate

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from config.constants import EOS_TOKEN_ID, UNK_TOKEN_ID
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
        self.inference_config.validate()
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
            raise RuntimeError("No model is configured for serving")

        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = [ord(c) if ord(c) < self.config.vocab_size else UNK_TOKEN_ID for c in text]

        if not tokens:
            raise ValueError("The prompt must contain at least one token")

        if hasattr(self.model, "generate"):
            return self.model.generate(
                text,
                tokenizer=self.tokenizer,
                max_new_tokens=(
                    max_new_tokens
                    if max_new_tokens is not None
                    else self.inference_config.max_new_tokens
                ),
            )

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        stop_ids = self.inference_config.stop_token_ids or [EOS_TOKEN_ID]
        eos_token_id = stop_ids[0]
        with torch.no_grad():
            output_ids = autoregressive_generate(
                self.model,
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=eos_token_id,
                use_cache=(
                    self.inference_config.use_kv_cache
                    and hasattr(self.model, "num_layers")
                ),
            )

        output_tokens = output_ids.squeeze(0).tolist()
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(output_tokens)
        return "".join(chr(token) for token in output_tokens if 0 <= token <= 0x10FFFF)
