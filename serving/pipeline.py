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
        token_limit = self.inference_config.max_new_tokens if max_new_tokens is None else max_new_tokens
        if not isinstance(token_limit, int) or isinstance(token_limit, bool) or token_limit < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")

        # Step 1: Tokenization
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = [ord(c) for c in text]
        from tokenizer.encoding import fit_token_ids_to_vocab
        model_vocab_size = getattr(
            getattr(self.model, "config", None), "vocab_size", self.config.vocab_size
        )
        tokens = fit_token_ids_to_vocab(tokens, model_vocab_size)

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Step 2: Model Forward Execution
        if self.model is None or not callable(self.model):
            raise RuntimeError("No callable model is available for inference")

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        output = None
        with torch.no_grad():
            for _ in range(token_limit):
                output = self.model(input_ids)
                if isinstance(output, tuple):
                    output = output[0]
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat((input_ids, next_token), dim=-1)
                if next_token.item() in self.inference_config.stop_token_ids:
                    break

        # Step 3: Decoding
        out_tokens = input_ids.squeeze(0).tolist()[len(tokens):]
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            try:
                return self.tokenizer.decode(out_tokens)
            except Exception:
                logger.debug("Tokenizer decoding failed; falling back to character decoding", exc_info=True)
        return "".join(chr(token) for token in out_tokens if 0 <= token <= 0x10FFFF)
