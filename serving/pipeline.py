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
        if isinstance(self.model, nn.Module):
            self.model.to(self.device).eval()

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
        attention_mask = torch.ones_like(input_ids)

        # Step 2: Autoregressive model execution
        output_ids = input_ids
        token_limit = self.inference_config.max_new_tokens if max_new_tokens is None else max_new_tokens
        if token_limit < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.model is None or not callable(self.model):
            raise RuntimeError("No callable model is available for serving")
        from inference.sampling.sampler import Sampler
        sampler = Sampler(
            temperature=self.inference_config.temperature,
            top_k=self.inference_config.top_k,
            top_p=self.inference_config.top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        if self.inference_config.seed is not None:
            torch.manual_seed(self.inference_config.seed)
        generated_tokens = []
        try:
            forward_parameters = inspect.signature(self.model.forward).parameters
        except (AttributeError, TypeError, ValueError):
            forward_parameters = {}
        if self.model is not None and callable(self.model):
            with torch.no_grad():
                for _ in range(token_limit):
                    if "attention_mask" in forward_parameters:
                        output = self.model(output_ids, attention_mask=attention_mask)
                    else:
                        output = self.model(output_ids)
                    if isinstance(output, tuple):
                        output = output[0]
                    next_id = sampler.sample(output[:, -1, :], output_ids[0]).view(1, 1)
                    output_ids = torch.cat((output_ids, next_id), dim=-1)
                    attention_mask = torch.cat((attention_mask, torch.ones_like(next_id)), dim=-1)
                    generated_tokens.append(int(next_id.item()))
                    if next_id.item() in self.inference_config.stop_token_ids:
                        break

        # Step 3: Decoding
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(generated_tokens)

        raise RuntimeError("No callable model is available for serving")
