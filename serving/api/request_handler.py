"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
import math
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
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        top_k = request_body.get("top_k")
        top_p = request_body.get("top_p")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0):
            raise ValueError("top_k must be a non-negative integer")
        if top_p is not None and (
            not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0.0 <= top_p <= 1.0
        ):
            raise ValueError("top_p must be in [0.0, 1.0]")

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = [ord(c) for c in prompt]

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if hasattr(generated_tokens, "tolist"):
                generated_tokens = generated_tokens.tolist()
            if generated_tokens and isinstance(generated_tokens[0], list):
                if len(generated_tokens) != 1:
                    raise ValueError("completion endpoint supports a single prompt at a time")
                generated_tokens = generated_tokens[0]
            completion_tokens = generated_tokens[len(prompt_tokens):]
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                response_text = self.tokenizer.decode(completion_tokens)
            else:
                response_text = " ".join(str(token) for token in completion_tokens)
        elif self.engine is not None and hasattr(self.engine, "generate"):
            raw_gen = self.engine.generate(
                prompt,
                tokenizer=self.tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if not isinstance(raw_gen, str):
                raise TypeError("engine.generate must return text")
            if raw_gen.startswith(prompt):
                response_text = raw_gen[len(prompt):]
                full_text = raw_gen
            else:
                response_text = raw_gen
                full_text = prompt + raw_gen
            if self.tokenizer is not None and hasattr(self.tokenizer, "encode") and isinstance(raw_gen, str):
                generated_tokens = self.tokenizer.encode(full_text)
            else:
                generated_tokens = prompt_tokens + [ord(char) for char in response_text]
            completion_tokens = generated_tokens[len(prompt_tokens):]
        else:
            raise RuntimeError("No inference engine is configured")

        from config.constants import EOS_TOKEN_ID
        stop_ids = getattr(getattr(self.engine, "inference_config", None), "stop_token_ids", []) or [EOS_TOKEN_ID]
        finish_reason = "stop" if any(token in stop_ids for token in completion_tokens) else "length"

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
