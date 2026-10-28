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
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                from tokenizer import BPETokenizer
                self.tokenizer = BPETokenizer()
            except Exception:
                self.tokenizer = None

        if engine is not None:
            self.engine = engine
        else:
            try:
                from inference.engine import InferenceEngine
                self.engine = InferenceEngine()
            except Exception:
                self.engine = None

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
            prompt_tokens = [ord(c) for c in prompt]

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                try:
                    completion_tokens = generated_tokens[len(prompt_tokens):]
                    response_text = self.tokenizer.decode(completion_tokens if len(completion_tokens) > 0 else generated_tokens)
                except Exception:
                    response_text = str(generated_tokens)
            else:
                response_text = str(generated_tokens)
        elif self.engine is not None and hasattr(self.engine, "generate"):
            raw_gen = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
            if isinstance(raw_gen, str):
                response_text = raw_gen
                if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                    generated_tokens = self.tokenizer.encode(raw_gen)
                else:
                    generated_tokens = [ord(c) for c in raw_gen]
            else:
                generated_tokens = raw_gen
                response_text = self.tokenizer.decode(generated_tokens) if self.tokenizer else str(generated_tokens)
        else:
            generated_tokens = prompt_tokens + [prompt_tokens[-1] + 1 if prompt_tokens else 42]
            response_text = str(generated_tokens)

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
