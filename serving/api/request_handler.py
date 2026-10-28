"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from config.constants import EOS_TOKEN_ID
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
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if not isinstance(temperature, (int, float)) or temperature < 0.0:
            raise ValueError("temperature must be non-negative")

        # Tokenize prompt
        if self.tokenizer is None or not hasattr(self.tokenizer, "encode"):
            raise ValueError("A tokenizer with an encode method is required")
        prompt_tokens = self.tokenizer.encode(prompt)

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=request_body.get("top_k"),
                top_p=request_body.get("top_p"),
                repetition_penalty=request_body.get("repetition_penalty"),
            )
            if not hasattr(self.tokenizer, "decode"):
                raise ValueError("A tokenizer with a decode method is required")
            completion_ids = generated_tokens[len(prompt_tokens):]
            response_text = self.tokenizer.decode(completion_ids)
            completion_count = len(completion_ids)
            configured_stops = getattr(
                getattr(self.engine, "inference_config", None),
                "stop_token_ids",
                [EOS_TOKEN_ID],
            )
            stop_ids = configured_stops or [EOS_TOKEN_ID]
            finish_reason = "stop" if any(token in stop_ids for token in completion_ids) else "length"
        elif self.engine is not None and hasattr(self.engine, "generate"):
            full_text = self.engine.generate(
                prompt,
                tokenizer=self.tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=request_body.get("top_k"),
                top_p=request_body.get("top_p"),
                repetition_penalty=request_body.get("repetition_penalty"),
            )
            response_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            completion_count = max(
                0, len(self.tokenizer.encode(full_text)) - len(prompt_tokens)
            )
            finish_reason = "length"
        else:
            raise RuntimeError("No inference engine is available")

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
                    "ready": self.engine is not None
                }
            ]
        }
