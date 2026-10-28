"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union
import torch

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
        if self.tokenizer is None or not hasattr(self.tokenizer, "encode") or not hasattr(self.tokenizer, "decode"):
            raise ValueError("ServingPipeline requires a tokenizer with encode and decode methods")

        # Step 1: Tokenization
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Step 2: Model Forward Execution
        if self.model is None or not callable(self.model):
            raise RuntimeError("ServingPipeline has no callable model")
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        from inference.generation.autoregressive import autoregressive_generate
        from inference.sampling.sampler import Sampler

        temperature = self.inference_config.temperature
        sampler = Sampler(
            temperature=temperature if temperature > 0 else 1.0,
            top_k=self.inference_config.top_k,
            top_p=self.inference_config.top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        sampler_fn = sampler.sample if temperature > 0 else None
        token_limit = (
            max_new_tokens if max_new_tokens is not None
            else self.inference_config.max_new_tokens
        )
        output_ids = autoregressive_generate(
            self.model,
            input_ids,
            max_new_tokens=token_limit,
            eos_token_id=self.inference_config.stop_token_ids[0]
            if self.inference_config.stop_token_ids else None,
            sampler_fn=sampler_fn,
            use_cache=self.inference_config.use_kv_cache,
        )
        return self.tokenizer.decode(output_ids[0].tolist())
