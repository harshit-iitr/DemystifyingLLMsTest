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

        # Step 2: Autoregressive generation (greedy) via the shared generation loop
        out_tokens = None
        if self.model is not None and callable(self.model):
            from inference.generation.autoregressive import autoregressive_generate

            n_new = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
            max_ctx = getattr(getattr(self.model, "config", None), "max_seq_len", None) or self.config.max_seq_len
            n_new = max(0, min(n_new, max_ctx - len(tokens)))

            vocab = getattr(self.tokenizer, "vocab", None)
            eos_token_id = getattr(vocab, "eos_token_id", None)

            if hasattr(self.model, "eval"):
                self.model.eval()
            with torch.no_grad():
                output = autoregressive_generate(
                    self.model,
                    input_ids,
                    max_new_tokens=n_new,
                    eos_token_id=eos_token_id,
                )
            out_tokens = output[0].tolist()

        # Step 3: Decoding (prompt + generated continuation)
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(out_tokens if out_tokens is not None else tokens)

        if out_tokens is not None:
            return "".join(chr(t) for t in out_tokens)

        return f"{text} [processed]"
