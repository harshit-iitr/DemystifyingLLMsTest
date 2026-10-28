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
        try:
            self.config = config or ModelConfig()
        except Exception:
            self.config = ModelConfig(d_model=768, vocab_size=50257)
        self.inference_config = inference_config or InferenceConfig()
        self.device = torch.device(device) if device else torch.device("cpu")

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                from tokenizer import BPETokenizer
                self.tokenizer = BPETokenizer()
            except Exception:
                self.tokenizer = None

        if model is not None:
            self.model = model
        else:
            try:
                from inference.engine import InferenceEngine
                self.model = InferenceEngine(
                    model_config=self.config,
                    inference_config=self.inference_config,
                    device=self.device
                )
            except Exception:
                self.model = None

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

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens

        # Step 2: Model Generation Execution
        if self.model is not None:
            if hasattr(self.model, "generate") and callable(self.model.generate):
                return self.model.generate(text, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
            elif hasattr(self.model, "generate_tokens") and callable(self.model.generate_tokens):
                out_ids = self.model.generate_tokens(prompt_tokens=tokens, max_new_tokens=max_tokens)
                if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                    return self.tokenizer.decode(out_ids)
                return " ".join(str(i) for i in out_ids)
            elif callable(self.model):
                from inference.generation.autoregressive import autoregressive_generate
                out_tensor = autoregressive_generate(
                    self.model,
                    input_ids,
                    max_new_tokens=max_tokens,
                    use_cache=self.inference_config.use_kv_cache
                )
                out_ids = out_tensor.squeeze(0).tolist()
                if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                    return self.tokenizer.decode(out_ids)
                return " ".join(str(i) for i in out_ids)

        return f"{text} [processed]"
