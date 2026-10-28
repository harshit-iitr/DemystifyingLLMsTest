"""Transformer model and end-to-end inference engine."""
from typing import Optional, List, Union, Any
import torch
import torch.nn as nn
from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from inference.generation import autoregressive_generate
from inference.sampling import Sampler


class FullTransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead, tie_weights
        self.config = config
        self.num_layers = config.n_layers
        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        if config.tie_weights:
            tie_weights(self.lm_head, self.embeddings.token_embedding)

    def forward(self, input_ids: torch.Tensor, kv_cache=None, **kwargs) -> torch.Tensor:
        offset = kv_cache.get_seq_length() if kv_cache is not None else 0
        hidden = self.embeddings(input_ids, offset=offset)
        hidden = self.blocks(hidden, kv_cache=kv_cache, **kwargs)
        return self.lm_head(hidden)


class InferenceEngine:
    def __init__(self, model_config: Optional[ModelConfig] = None,
                 inference_config: Optional[InferenceConfig] = None,
                 model: Optional[Any] = None,
                 device: Optional[Union[str, torch.device]] = None):
        self.model_config = model_config or getattr(model, "config", None) or ModelConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device or 'cpu')
        if self.inference_config.seed is not None:
            torch.manual_seed(self.inference_config.seed)
        self.model = model if model is not None else FullTransformerModel(self.model_config)
        self._prepare_model()

    def _prepare_model(self) -> None:
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def generate_tokens(self, prompt_tokens: List[int], max_new_tokens: Optional[int] = None,
                        temperature: Optional[float] = None, top_k: Optional[int] = None,
                        top_p: Optional[float] = None) -> List[int]:
        self._prepare_model()
        cfg = self.inference_config
        count = cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        sampler = Sampler(temperature=cfg.temperature if temperature is None else temperature,
                          top_k=cfg.top_k if top_k is None else top_k,
                          top_p=cfg.top_p if top_p is None else top_p,
                          repetition_penalty=cfg.repetition_penalty)
        history = list(prompt_tokens)
        def sample(scores):
            token = sampler.sample(scores, history)
            history.append(int(token.item()))
            return token
        # Stop IDs are evaluated after appending the generated token.
        stop_ids = set(cfg.stop_token_ids) if cfg.early_stopping else set()
        inputs = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        output = autoregressive_generate(self.model, inputs, count, sampler_fn=sample,
                                         use_cache=cfg.use_kv_cache, stop_token_ids=stop_ids)
        return output[0].tolist()

    def generate(self, prompt: str, tokenizer: Optional[Any] = None,
                 max_new_tokens: Optional[int] = None, **kwargs) -> str:
        ids = tokenizer.encode(prompt) if tokenizer is not None else [ord(c) for c in prompt]
        if not ids:
            ids = [self.model_config.vocab_size - 1]
        output = self.generate_tokens(ids, max_new_tokens=max_new_tokens, **kwargs)
        if tokenizer is not None:
            return tokenizer.decode(output)
        return ''.join(chr(i) for i in output)
