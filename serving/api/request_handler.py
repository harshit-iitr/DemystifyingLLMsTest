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
            raise ValueError("Request body must be a JSON object")
        prompt = request_body.get("prompt", "")
        max_tokens = request_body.get("max_tokens", 16)
        temperature = request_body.get("temperature", 0.8)
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        from config.inference_config import InferenceConfig
        InferenceConfig(max_new_tokens=max_tokens, temperature=temperature,
                        top_k=request_body.get("top_k", 0),
                        top_p=request_body.get("top_p", 1.0)).validate()
        if self.engine is None:
            raise RuntimeError("No inference engine configured")
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer configured")
        prompt_tokens = self.tokenizer.encode(prompt)
        if hasattr(self.engine, "generate_tokens"):
            options = {name: request_body[name] for name in ("top_k", "top_p") if name in request_body}
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens, max_new_tokens=max_tokens,
                temperature=temperature, **options)
            response_text = self.tokenizer.decode(generated_tokens[len(prompt_tokens):])
        elif hasattr(self.engine, "generate"):
            # Preserve the public generate-only engine contract using its actual
            # text output, without inventing token IDs or a continuation.
            raw_text = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
            if not isinstance(raw_text, str):
                raise TypeError("Engine.generate must return text")
            response_text = raw_text[len(prompt):] if raw_text.startswith(prompt) else raw_text
            completion_ids = self.tokenizer.encode(response_text)
            vocab = getattr(self.tokenizer, "vocab", None)
            bos = getattr(vocab, "bos_token_id", None)
            if bos is not None and completion_ids and completion_ids[0] == bos:
                completion_ids = completion_ids[1:]
            generated_tokens = list(prompt_tokens) + list(completion_ids)
        else:
            raise TypeError("Engine must implement generate_tokens or generate")

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
                    "finish_reason": "length" if len(generated_tokens) - len(prompt_tokens) == max_tokens else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(generated_tokens) - len(prompt_tokens),
                "total_tokens": len(generated_tokens)
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
