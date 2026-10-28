"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union, List
import inspect
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
            tokens = [50256]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        if self.model is None or not callable(self.model):
            raise RuntimeError("No inference model is available")

        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        from inference.generation import autoregressive_generate
        from inference.sampling import Sampler

        generation_limit = max_new_tokens
        if generation_limit is None:
            generation_limit = self.inference_config.max_new_tokens
        sampler = None
        if self.inference_config.temperature > 0:
            sampler = Sampler(
                temperature=self.inference_config.temperature,
                top_k=self.inference_config.top_k,
                top_p=self.inference_config.top_p,
                repetition_penalty=self.inference_config.repetition_penalty,
            )

        def sample_next(logits: torch.Tensor) -> torch.Tensor:
            return sampler.sample(logits) if sampler is not None else torch.argmax(logits, dim=-1)

        model_forward = getattr(self.model, "forward", None)
        use_cache = self.inference_config.use_kv_cache
        if model_forward is None:
            use_cache = False
        else:
            try:
                parameters = inspect.signature(model_forward).parameters.values()
                use_cache = use_cache and any(
                    parameter.name == "kv_cache" or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                use_cache = False

        with torch.no_grad():
            output_ids = autoregressive_generate(
                self.model,
                input_ids,
                max_new_tokens=generation_limit,
                eos_token_id=(self.inference_config.stop_token_ids[0]
                              if self.inference_config.stop_token_ids else None),
                sampler_fn=sample_next,
                use_cache=use_cache,
            )

        output_tokens = output_ids[0, input_ids.size(1):].tolist()
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(output_tokens)
        return " ".join(map(str, output_tokens))
