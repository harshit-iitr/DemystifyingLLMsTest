"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union, List
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from config.constants import EOS_TOKEN_ID
from inference.generation import autoregressive_generate
from inference.sampling import Sampler
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

        if self.model is None or not callable(self.model):
            return f"{text} [processed]"

        # Step 2: Autoregressive generation (model expects a (batch, seq) LongTensor)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        max_new = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        cfg = self.inference_config
        sampler = Sampler(
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
        )
        if hasattr(self.model, "eval"):
            self.model.eval()
        with torch.no_grad():
            output = autoregressive_generate(
                self.model,
                input_ids,
                max_new_tokens=max_new,
                eos_token_id=EOS_TOKEN_ID,
                sampler_fn=lambda logits: sampler.sample(logits),
            )

        # Step 3: Decode only the newly generated tokens
        new_tokens = output[0, input_ids.size(1):].tolist()
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(new_tokens)
        return "".join(chr(t) for t in new_tokens if 0 <= t < 0x110000)
