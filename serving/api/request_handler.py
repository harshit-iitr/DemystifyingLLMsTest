"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from config.constants import EOS_TOKEN_ID, UNK_TOKEN_ID
from utils.logging_utils import get_logger

logger = get_logger("request_handler")


class RequestHandler:
    """
    Handles endpoint request validation, inference invocation, and response formatting.
    """

    def __init__(self, engine: Optional[Any] = None, tokenizer: Optional[Any] = None):
        self.engine = engine
        self.tokenizer = tokenizer

    def handle_completion(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process OpenAI-compatible /v1/completions request.

        Args:
            request_body: Dictionary containing 'prompt', optional 'max_tokens', 'temperature', etc.

        Returns:
            Formatted response dictionary.
        """
        prompt = request_body.get("prompt", "")
        max_tokens = request_body.get("max_tokens", 16)
        temperature = request_body.get("temperature", 0.8)
        top_k = request_body.get("top_k")
        top_p = request_body.get("top_p")
        stop = request_body.get("stop")
        stop_token_ids = request_body.get("stop_token_ids")
        stop_strings: List[str] = []
        stop_sequences: List[List[int]] = []

        if stop_token_ids is None:
            if isinstance(stop, int):
                stop_token_ids = [stop]
            elif isinstance(stop, list) and all(isinstance(token, int) for token in stop):
                stop_token_ids = stop
            elif isinstance(stop, str):
                stop_strings = [stop]
            elif isinstance(stop, list) and all(isinstance(item, str) for item in stop):
                stop_strings = stop
        stop_token_ids = stop_token_ids or [EOS_TOKEN_ID]

        for stop_text in stop_strings:
            if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                stop_ids = self.tokenizer.encode(stop_text, add_bos=False)
            else:
                vocab_size = getattr(getattr(self.engine, "model_config", None), "vocab_size", 50257)
                stop_ids = [ord(char) if ord(char) < vocab_size else UNK_TOKEN_ID for char in stop_text]
            if stop_ids:
                stop_sequences.append(stop_ids)

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            vocab_size = getattr(getattr(self.engine, "model_config", None), "vocab_size", 50257)
            prompt_tokens = [ord(c) if ord(c) < vocab_size else UNK_TOKEN_ID for c in prompt]

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_token_ids=stop_token_ids,
                stop_sequences=stop_sequences,
            )
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                full_text = self.tokenizer.decode(generated_tokens)
                response_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            else:
                response_text = "".join(chr(token) for token in generated_tokens[len(prompt_tokens):] if 0 <= token <= 0x10FFFF)
        elif self.engine is not None and hasattr(self.engine, "generate"):
            full_text = self.engine.generate(
                prompt,
                tokenizer=self.tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_token_ids=stop_token_ids,
                stop_sequences=stop_sequences,
            )
            response_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                generated_tokens = self.tokenizer.encode(full_text)
            else:
                generated_tokens = [ord(char) for char in full_text]
        else:
            raise RuntimeError("No inference engine is configured")

        stop_markers = stop_strings
        stopped_by_text = False
        if stop_markers:
            stop_positions = [response_text.find(marker) for marker in stop_markers if marker in response_text]
            if stop_positions:
                stopped_by_text = True
                response_text = response_text[:min(stop_positions)]

        configured_stop_ids = getattr(
            getattr(self.engine, "inference_config", None),
            "stop_token_ids",
            [EOS_TOKEN_ID],
        )
        stopped_by_sequence = any(
            sequence and generated_tokens[-len(sequence):] == sequence
            for sequence in stop_sequences
        )
        finish_reason = "stop" if (
            stopped_by_text
            or stopped_by_sequence
            or (generated_tokens and generated_tokens[-1] in (stop_token_ids or configured_stop_ids))
        ) else "length"

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "broken-transformer",
            "text": response_text,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": max(0, len(generated_tokens) - len(prompt_tokens)),
                "total_tokens": max(len(prompt_tokens), len(generated_tokens))
            }
        }

    def handle_health(self) -> Dict[str, Any]:
        """Process /health endpoint check."""
        return {
            "status": "ok",
            "service": "broken-transformer-server",
            "timestamp": time.time()
        }

    def handle_models(self) -> Dict[str, Any]:
        """Process /v1/models endpoint check."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "broken-transformer",
                    "object": "model",
                    "owned_by": "openai-dsg-iitr",
                    "ready": True
                }
            ]
        }
