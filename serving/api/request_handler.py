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

        if self.tokenizer is None or not hasattr(self.tokenizer, "encode") or not hasattr(self.tokenizer, "decode"):
            raise ValueError("A tokenizer with encode and decode methods is required for completion requests")
        prompt_tokens = self.tokenizer.encode(prompt)

        completion_text: str
        completion_count: int
        finish_reason = "length"

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=request_body.get("top_k"),
                top_p=request_body.get("top_p"),
            )
            completion_ids = generated_tokens[len(prompt_tokens):]
            completion_text = self.tokenizer.decode(completion_ids)
            completion_count = len(completion_ids)
            stop_ids = getattr(getattr(self.engine, "inference_config", None), "stop_token_ids", [])
            if completion_ids and completion_ids[-1] in stop_ids:
                finish_reason = "stop"
        elif self.engine is not None and hasattr(self.engine, "generate"):
            raw_text = self.engine.generate(
                prompt,
                tokenizer=self.tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=request_body.get("top_k"),
                top_p=request_body.get("top_p"),
            )
            completion_text = raw_text[len(prompt):] if raw_text.startswith(prompt) else raw_text
            try:
                completion_count = len(self.tokenizer.encode(completion_text, add_bos=False))
            except TypeError:
                completion_count = len(self.tokenizer.encode(completion_text))
        else:
            raise RuntimeError("No inference engine is available to handle the completion request")

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "broken-transformer",
            "text": completion_text,
            "choices": [
                {
                    "text": completion_text,
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
                    "ready": True
                }
            ]
        }
