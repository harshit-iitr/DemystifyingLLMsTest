"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union, List
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from config.constants import BOS_TOKEN_ID


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
        self.config = config if config is not None else ModelConfig()
        self.inference_config = inference_config if inference_config is not None else InferenceConfig()
        self.inference_config.validate()
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
        if not tokens:
            tokens = [BOS_TOKEN_ID]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Step 2: Model Forward Execution
        token_count = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if token_count <= 0:
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                return self.tokenizer.decode(tokens)
            return text

        if self.model is not None and callable(self.model):
            self.model.eval()
            with torch.no_grad():
                from inference.generation.autoregressive import autoregressive_generate
                sampler = None
                if self.inference_config.temperature > 0:
                    from inference.sampling.sampler import Sampler
                    sampler = Sampler(
                        temperature=self.inference_config.temperature,
                        top_k=self.inference_config.top_k,
                        top_p=self.inference_config.top_p,
                        repetition_penalty=self.inference_config.repetition_penalty,
                    )
                    generator = None
                    if self.inference_config.seed is not None:
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(self.inference_config.seed)

                    def sample_next(logits: torch.Tensor, generated_tokens: torch.Tensor) -> torch.Tensor:
                        return sampler.sample(logits, generated_tokens=generated_tokens, generator=generator)

                generated = autoregressive_generate(
                    self.model,
                    input_ids,
                    max_new_tokens=token_count,
                    eos_token_id=(self.inference_config.stop_token_ids[0]
                                  if self.inference_config.stop_token_ids else None),
                    sampler_fn=sample_next if sampler is not None else None,
                    use_cache=self.inference_config.use_kv_cache,
                )
                completion = generated[0, input_ids.size(1):].tolist()
                if completion and self.inference_config.stop_token_ids and completion[-1] in self.inference_config.stop_token_ids:
                    completion.pop()
                if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                    return self.tokenizer.decode(completion)
                return "".join(chr(int(token)) for token in completion if 0 <= int(token) <= 0x10FFFF)
        else:
            raise RuntimeError("No model is available to generate a completion")
