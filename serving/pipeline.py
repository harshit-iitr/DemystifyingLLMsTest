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
            raise ValueError("A tokenizer with encode/decode methods is required")
        if self.model is None or not callable(self.model):
            raise RuntimeError("A callable model is required for serving")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        tokens = self.tokenizer.encode(text)

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Step 2: Model Forward Execution
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        from inference.generation.autoregressive import autoregressive_generate
        from inference.sampling import Sampler
        import inspect
        sampler = Sampler(
            temperature=self.inference_config.temperature,
            top_k=self.inference_config.top_k,
            top_p=self.inference_config.top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        history = list(tokens)
        generator = None
        if self.inference_config.seed is not None and self.inference_config.temperature > 0:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.inference_config.seed)

        def sample_next(logits):
            next_ids = sampler.sample(logits, history, generator=generator)
            history.extend(next_ids.detach().cpu().tolist())
            return next_ids

        forward = getattr(self.model, "forward", self.model)
        try:
            signature = inspect.signature(forward)
            supports_cache = "kv_cache" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            supports_cache = False

        with torch.no_grad():
            output_ids = autoregressive_generate(
                self.model, input_ids,
                max_new_tokens=(self.inference_config.max_new_tokens if max_new_tokens is None else max_new_tokens),
                eos_token_id=getattr(getattr(self.tokenizer, "vocab", None), "eos_token_id", None),
                sampler_fn=sample_next,
                use_cache=self.inference_config.use_kv_cache and supports_cache,
                stop_token_ids=self.inference_config.stop_token_ids,
            )

        # Step 3: Decoding
        return self.tokenizer.decode(output_ids.squeeze(0).tolist())

