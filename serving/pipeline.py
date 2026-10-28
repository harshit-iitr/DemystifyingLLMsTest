"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union, List
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from utils.logging_utils import get_logger
from inference.generation.autoregressive import autoregressive_generate
from inference.sampling.sampler import Sampler

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
        if self.model is None or not callable(self.model):
            raise RuntimeError("No inference model is available")

        # Step 1: Tokenization
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = [ord(c) for c in text]

        input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        # Step 2: Model Forward Execution
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        with torch.no_grad():
            max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
            if self.inference_config.seed is not None:
                torch.manual_seed(self.inference_config.seed)
            sampler = Sampler(
                temperature=self.inference_config.temperature,
                top_k=self.inference_config.top_k,
                top_p=self.inference_config.top_p,
                repetition_penalty=self.inference_config.repetition_penalty,
            )
            output = autoregressive_generate(
                self.model,
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=self.inference_config.stop_token_ids[0] if self.inference_config.stop_token_ids else None,
                sampler_fn=lambda logits: sampler.sample(logits),
                use_cache=self.inference_config.use_kv_cache and hasattr(self.model, "blocks"),
            )

        # Step 3: Decoding
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            if isinstance(output, torch.Tensor):
                out_tokens = output.squeeze(0).tolist()
                # Generation returns prompt + continuation; this method promises
                # the completion only.
                if out_tokens[:len(tokens)] == tokens:
                    out_tokens = out_tokens[len(tokens):]
                return self.tokenizer.decode(out_tokens)
            return self.tokenizer.decode([])

        return str(output)
