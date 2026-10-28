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
        if not isinstance(request_body, dict):
            raise ValueError("request body must be a JSON object")
        prompt = request_body.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        max_tokens = request_body.get("max_tokens", 16)
        temperature = request_body.get("temperature", 0.8)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("temperature must be a non-negative number")

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = [ord(c) for c in prompt]

        # Generate continuation
        completion_tokens = []
        generate_only = False
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generation_args = {
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            }
            for name in ("top_k", "top_p"):
                if name in request_body:
                    generation_args[name] = request_body[name]
            generated_tokens = self.engine.generate_tokens(**generation_args)
        elif self.engine is not None and hasattr(self.engine, "generate"):
            generate_only = True
            raw_gen = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
            if isinstance(raw_gen, str) and raw_gen.startswith(prompt):
                raw_gen = raw_gen[len(prompt):]
            response_text = raw_gen
            generated_tokens = prompt_tokens
        else:
            raise RuntimeError("No inference engine is available to handle completion")

        if not generate_only:
            has_prompt_prefix = (
                len(generated_tokens) >= len(prompt_tokens)
                and list(generated_tokens[:len(prompt_tokens)]) == list(prompt_tokens)
            )
            completion_tokens = generated_tokens[len(prompt_tokens):] if has_prompt_prefix else generated_tokens
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                response_text = self.tokenizer.decode(completion_tokens)
            else:
                response_text = "".join(chr(token) for token in completion_tokens if 0 <= token <= 0x10FFFF)
        completion_count = len(completion_tokens)
        if generate_only:
            if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                try:
                    completion_count = len(self.tokenizer.encode(response_text, add_bos=False))
                except TypeError:
                    completion_count = len(self.tokenizer.encode(response_text))

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
