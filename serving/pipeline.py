"""Serving pipeline sharing the real inference engine."""
from typing import Optional, Any, Union
import torch
from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from inference.engine import InferenceEngine
from tokenizer import BPETokenizer


class ServingPipeline:
    def __init__(self, model: Optional[Any] = None, tokenizer: Optional[Any] = None,
                 config: Optional[ModelConfig] = None, inference_config: Optional[InferenceConfig] = None,
                 device: Optional[Union[str, torch.device]] = None):
        self.tokenizer = tokenizer if tokenizer is not None else BPETokenizer()
        self.engine = InferenceEngine(config, inference_config, model, device)
        self.model = self.engine.model
        self.config = self.engine.model_config
        self.inference_config = self.engine.inference_config
        self.device = self.engine.device

    def process(self, text: str, max_new_tokens: Optional[int] = None) -> str:
        return self.engine.generate(text, tokenizer=self.tokenizer, max_new_tokens=max_new_tokens)
