"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
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

        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError("temperature must be a positive number")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
            raise ValueError("top_k must be a non-negative integer")
        if top_p is not None and (not isinstance(top_p, (int, float)) or not 0 <= top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")

        # Tokenize prompt
        if self.tokenizer is None or not hasattr(self.tokenizer, "encode") or not hasattr(self.tokenizer, "decode"):
            raise RuntimeError("A tokenizer with encode and decode methods is required")
        prompt_tokens = self.tokenizer.encode(prompt)

        # Generate continuation
        generated_text = None
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        elif self.engine is not None and hasattr(self.engine, "generate"):
            generated_text = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
            if isinstance(generated_text, str) and generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            if not isinstance(generated_text, str):
                raise TypeError("engine.generate must return a string")
            generated_tokens = prompt_tokens
        else:
            raise RuntimeError("No inference engine is available")

        continuation_ids = generated_tokens[len(prompt_tokens):]
        if generated_text is not None:
            response_text = generated_text
            try:
                completion_count = len(self.tokenizer.encode(generated_text, add_bos=False))
            except TypeError:
                completion_count = len(self.tokenizer.encode(generated_text))
        else:
            response_text = self.tokenizer.decode(continuation_ids)
            completion_count = len(continuation_ids)

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
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": completion_count,
                "total_tokens": len(prompt_tokens) + completion_count
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
