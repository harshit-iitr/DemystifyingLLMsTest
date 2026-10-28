"""Transformer construction and end-to-end generation without synthetic fallbacks."""
from typing import Optional, List, Union, Any
import torch
import torch.nn as nn
from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from config.constants import BOS_TOKEN_ID, EOS_TOKEN_ID
from inference.sampling import Sampler
from inference.generation.kv_cache import KVCache, supports_kv_cache


class FullTransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.n_layers
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead
        from model.heads.weight_tying import tie_weights
        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        if config.tie_weights:
            tie_weights(self.lm_head, self.embeddings.token_embedding)

    def forward(self, input_ids: torch.Tensor, mask=None, kv_cache=None,
                attention_mask=None, position_ids=None, **kwargs):
        offset = kv_cache.get_seq_length() if kv_cache is not None else 0
        if attention_mask is not None:
            expected_shape = (input_ids.size(0), offset + input_ids.size(1))
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError("attention_mask must cover the complete cached and current sequence")
            allowed = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            if position_ids is None:
                position_ids = (allowed.long().cumsum(-1) - 1).clamp_min(0)[:, offset:]
            bias = torch.zeros(expected_shape, device=input_ids.device, dtype=self.embeddings.token_embedding.embedding.weight.dtype)
            bias.masked_fill_(~allowed, float("-inf"))
            bias = bias[:, None, None, :]
            if mask is not None and mask.dtype == torch.bool:
                mask = torch.zeros_like(mask, dtype=bias.dtype).masked_fill(~mask, float("-inf"))
            mask = bias if mask is None else mask + bias
        h = self.embeddings(input_ids, offset=offset, position_ids=position_ids)
        h = self.blocks(h, mask=mask, kv_cache=kv_cache)
        return self.lm_head(h)


class InferenceEngine:
    def __init__(self, model_config: Optional[ModelConfig] = None,
                 inference_config: Optional[InferenceConfig] = None,
                 model: Optional[Any] = None, device: Optional[Union[str, torch.device]] = None):
        self.model_config = model_config or getattr(model, "config", None) or ModelConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device or "cpu")
        if self.inference_config.seed is not None:
            torch.manual_seed(self.inference_config.seed)
        self.model = model if model is not None else FullTransformerModel(self.model_config)
        self._prepare_model()

    def _prepare_model(self):
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        elif hasattr(self.model, "train"):
            self.model.train(False)

    @torch.no_grad()
    def generate_tokens(self, prompt_tokens: List[int], max_new_tokens: Optional[int] = None,
                        temperature: Optional[float] = None, top_k: Optional[int] = None,
                        top_p: Optional[float] = None,
                        eos_token_id: Optional[int] = None) -> List[int]:
        self._prepare_model()
        cfg = self.inference_config
        count = cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        if type(count) is not int or count < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        tokens = list(prompt_tokens)
        if count == 0:
            return tokens
        if not tokens:
            tokens = [getattr(self.model_config, "bos_token_id", min(BOS_TOKEN_ID, self.model_config.vocab_size - 1))]
        if len(tokens) > self.model_config.max_seq_len:
            raise ValueError("Prompt exceeds the model context window")
        count = min(count, self.model_config.max_seq_len - len(tokens))
        if count == 0:
            return tokens
        sampler = Sampler(cfg.temperature if temperature is None else temperature,
                          cfg.top_k if top_k is None else top_k,
                          cfg.top_p if top_p is None else top_p, cfg.repetition_penalty)
        eos = getattr(self.model_config, "eos_token_id", min(EOS_TOKEN_ID, self.model_config.vocab_size - 1)) if eos_token_id is None else eos_token_id
        stop_ids = set(cfg.stop_token_ids) | {eos}
        if cfg.beam_width > 1:
            from inference.generation.beam_search import BeamSearch
            return BeamSearch(cfg.beam_width, count,
                              eos if cfg.early_stopping else None,
                              stop_token_ids=stop_ids if cfg.early_stopping else []).search(
                self.model, torch.tensor([tokens], device=self.device)).squeeze(0).tolist()
        # Honor a model's public cache capability while retaining ordinary
        # single-input forward calls for models without cache support.
        num_layers = getattr(self.model, "num_layers", self.model_config.n_layers)
        cache = KVCache(num_layers) if cfg.use_kv_cache and num_layers > 0 and supports_kv_cache(self.model) else None
        for step in range(count):
            if len(tokens) > self.model_config.max_seq_len:
                raise ValueError("Generation exceeded the model context window")
            model_tokens = tokens[-1:] if cache is not None and cache.get_seq_length() else tokens
            inputs = torch.tensor([model_tokens], dtype=torch.long, device=self.device)
            logits = self.model(inputs, kv_cache=cache) if cache is not None else self.model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_id = int(sampler.sample(logits[:, -1, :], tokens).item())
            tokens.append(next_id)
            if cfg.early_stopping and next_id in stop_ids:
                break
        return tokens

    def generate(self, prompt: str, tokenizer: Optional[Any] = None,
                 max_new_tokens: Optional[int] = None, **kwargs) -> str:
        if tokenizer is None:
            from tokenizer import BPETokenizer
            tokenizer = BPETokenizer()
        if hasattr(tokenizer, "bind_to_model"):
            tokenizer.bind_to_model(self.model_config.vocab_size)
        ids = tokenizer.encode(prompt)
        vocab = getattr(tokenizer, "vocab", None)
        if "eos_token_id" not in kwargs and vocab is not None:
            kwargs["eos_token_id"] = getattr(vocab, "eos_token_id", EOS_TOKEN_ID)
        output = self.generate_tokens(ids, max_new_tokens=max_new_tokens, **kwargs)
        return tokenizer.decode(output)
