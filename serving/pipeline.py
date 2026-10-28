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
        self.inference_config.validate()
        self.device = torch.device(device) if device else torch.device("cpu")
        if isinstance(self.model, nn.Module):
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
        if self.model is None or not callable(self.model):
            raise RuntimeError("ServingPipeline requires a callable model")
        if self.tokenizer is None or not hasattr(self.tokenizer, "encode") or not hasattr(self.tokenizer, "decode"):
            raise ValueError("ServingPipeline requires a tokenizer with encode() and decode()")

        tokens = self.tokenizer.encode(text)
        if not tokens:
            raise ValueError("Tokenization produced an empty prompt")
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        from inference.generation import autoregressive_generate
        from inference.sampling import Sampler
        sampler = Sampler(
            temperature=self.inference_config.temperature,
            top_k=self.inference_config.top_k,
            top_p=self.inference_config.top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        sampling_history = list(tokens)

        def sample_next(logits: torch.Tensor) -> torch.Tensor:
            sampled = sampler.sample(logits, generated_tokens=sampling_history)
            sampling_history.append(int(sampled.reshape(-1)[0].item()))
            return sampled

        was_training = self.model.training if isinstance(self.model, nn.Module) else None
        try:
            output_ids = autoregressive_generate(
                self.model,
                input_ids,
                max_new_tokens=(max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens),
                eos_token_id=self.inference_config.stop_token_ids,
                sampler_fn=sample_next,
                use_cache=False,
            )
        finally:
            if was_training:
                self.model.train()
        continuation = output_ids[0, len(tokens):].tolist()
        return self.tokenizer.decode(continuation)
