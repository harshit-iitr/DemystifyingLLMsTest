"""
Inference engine for the transformer model.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig

from model.embeddings.embedding_layer import EmbeddingLayer
from model.blocks.block_stack import BlockStack
from model.heads.lm_head import LMHead
from model.heads.weight_tying import tie_weights

from inference.sampling.sampler import Sampler


class FullTransformerModel(nn.Module):
    """
    Full transformer model:
        1. Token and positional embeddings
        2. Transformer block stack
        3. Language modeling head
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()

        self.config = config if config is not None else ModelConfig()
        self.num_layers = self.config.n_layers

        self.embeddings = EmbeddingLayer(config=self.config)
        self.blocks = BlockStack(config=self.config)
        self.lm_head = LMHead(config=self.config)
        if getattr(self.config, "tie_weights", False):
            tie_weights(self.lm_head, self.embeddings.token_embedding)

    def forward(
        self,
        input_ids: torch.Tensor,
        offset: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length).

        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size).
        """
        hidden_states = self.embeddings(input_ids, offset=offset)
        hidden_states = self.blocks(hidden_states, **kwargs)

        return self.lm_head(hidden_states)


class InferenceEngine:
    """
    Autoregressive inference engine for transformer models.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        model: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ):
        self.model_config = (
            model_config if model_config is not None else ModelConfig()
        )

        self.inference_config = (
            inference_config
            if inference_config is not None
            else InferenceConfig()
        )

        self.inference_config.validate()

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.model = (
            model
            if model is not None
            else FullTransformerModel(config=self.model_config)
        )

        self._prepare_model()

    def _prepare_model(self) -> None:
        """Move the model to the configured device and enable evaluation mode."""
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        if hasattr(self.model, "eval"):
            self.model.eval()

    def _create_sampler(self, temperature: float) -> Sampler:
        """Create a sampler using the configured decoding parameters."""
        return Sampler(
            temperature=temperature,
            top_k=self.inference_config.top_k,
            top_p=self.inference_config.top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )

    @torch.inference_mode()
    def generate_tokens(
        self,
        input_ids: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[int]:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Encoded prompt token IDs.
            max_new_tokens: Maximum number of additional tokens to generate.
            temperature: Sampling temperature; zero means greedy decoding.

        Returns:
            Prompt token IDs followed by generated token IDs.
        """
        if not input_ids:
            raise ValueError("input_ids must contain at least one token.")

        if max_new_tokens is None:
            max_new_tokens = self.inference_config.max_new_tokens

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

        if temperature is None:
            temperature = self.inference_config.temperature

        if temperature < 0:
            raise ValueError("temperature must be non-negative.")

        generated = [int(token_id) for token_id in input_ids]

        if max_new_tokens == 0:
            return generated

        self._prepare_model()

        # Set a reproducible seed when one is configured.
        seed = self.inference_config.seed
        if seed is not None:
            torch.manual_seed(seed)

            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        sampler = self._create_sampler(temperature) if temperature > 0 else None
        
        stop_token_ids = set(self.inference_config.stop_token_ids)

        # This implementation deliberately does not use KV caching yet.
        # Each iteration sends the full generated sequence to the model.
        # KV caching can be integrated once cache updates and positional
        # offsets are verified together.

        for _ in range(max_new_tokens):
            input_tensor = torch.tensor(
                [generated],
                dtype=torch.long,
                device=self.device,
            )

            logits = self.model(input_tensor)

            if not isinstance(logits, torch.Tensor):
                raise TypeError(
                    "The model must return a torch.Tensor containing logits."
                )

            if logits.ndim != 3:
                raise ValueError(
                    "Expected model logits with shape "
                    "(batch_size, sequence_length, vocab_size); "
                    f"received {tuple(logits.shape)}."
                )

            if logits.size(0) != 1 or logits.size(1) != len(generated):
                raise ValueError(
                    "Unexpected logits shape for the supplied input sequence."
                )

            # Select logits corresponding to the final prompt/generated token.
            next_token_logits = logits[:, -1, :]

            # Apply repetition penalty, top-k, top-p, and temperature
            # through the repository's Sampler.
            if temperature == 0:
                # Greedy decoding: choose the highest logit directly.
                next_token = torch.argmax(
                    next_token_logits,
                    dim=-1,
                )
            else:
                next_token = sampler.sample(
                    next_token_logits,
                    generated_tokens=generated,
                )

            next_token_id = int(next_token.reshape(-1)[0].item())

            # Append the new token after the existing sequence.
            generated.append(next_token_id)

            if next_token_id in stop_token_ids:
                break

        return generated

    def generate(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Encode a prompt, generate a continuation, and decode the result.
        """
        if tokenizer is None:
            raise ValueError("A tokenizer must be provided.")

        if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise TypeError(
                "The tokenizer must provide encode() and decode() methods."
            )

        prompt_ids = tokenizer.encode(prompt)

        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.reshape(-1).tolist()

        prompt_ids = [int(token_id) for token_id in prompt_ids]

        if not prompt_ids:
            raise ValueError(
                "The tokenizer returned no tokens for the supplied prompt."
            )

        output_ids = self.generate_tokens(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return tokenizer.decode(output_ids)