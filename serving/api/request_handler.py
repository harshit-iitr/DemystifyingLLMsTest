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

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = list(prompt.encode("utf-8"))

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
        elif self.engine is not None and hasattr(self.engine, "generate"):
            full_text = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens, temperature=temperature)
            response_text = full_text[len(prompt):] if isinstance(full_text, str) and full_text.startswith(prompt) else full_text
            if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                try:
                    continuation_tokens = self.tokenizer.encode(response_text, add_bos=False)
                except TypeError:
                    continuation_tokens = self.tokenizer.encode(response_text)
            else:
                continuation_tokens = list(response_text.encode("utf-8")) if isinstance(response_text, str) else []
            generated_tokens = prompt_tokens + continuation_tokens
        else:
            raise RuntimeError("An inference engine is required to handle completion requests")

        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                response_text = self.tokenizer.decode(generated_tokens[len(prompt_tokens):])
            else:
                response_text = bytes(token for token in generated_tokens[len(prompt_tokens):] if 0 <= token <= 255).decode("utf-8", errors="replace")
        completion_count = max(0, len(generated_tokens) - len(prompt_tokens))

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
