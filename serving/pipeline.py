"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union, List
import torch
import torch.nn as nn
import inspect

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
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(self.device)

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
            raise RuntimeError("Serving pipeline requires a callable model")

        from inference.generation import autoregressive_generate
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        use_cache = self.inference_config.use_kv_cache
        if use_cache:
            try:
                parameters = inspect.signature(self.model.forward if isinstance(self.model, nn.Module) else self.model).parameters
                accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
                use_cache = "kv_cache" in parameters or accepts_kwargs
            except (TypeError, ValueError):
                use_cache = False

        with torch.no_grad():
            output_ids = autoregressive_generate(
                self.model,
                input_ids,
                max_new_tokens=(max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens),
                eos_token_id=(self.inference_config.stop_token_ids[0] if self.inference_config.stop_token_ids else None),
                use_cache=use_cache,
            )
        completion_tokens = output_ids[0, input_ids.size(1):].tolist()
        output_tokens = output_ids[0].tolist()
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(completion_tokens)
        return " ".join(str(token) for token in completion_tokens)
