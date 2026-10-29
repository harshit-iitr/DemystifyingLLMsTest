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
        sampling_args = {
            name: request_body[name]
            for name in ("top_k", "top_p", "repetition_penalty")
            if name in request_body
        }

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = [ord(c) for c in prompt]

        completion_text = None
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **sampling_args,
            )
            completion_tokens = generated_tokens[len(prompt_tokens):]
        elif self.engine is not None and hasattr(self.engine, "generate"):
            raw_gen = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
            if isinstance(raw_gen, str):
                completion_text = raw_gen[len(prompt):] if raw_gen.startswith(prompt) else raw_gen
                if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                    try:
                        completion_tokens = self.tokenizer.encode(completion_text, add_bos=False)
                    except TypeError:
                        completion_tokens = self.tokenizer.encode(completion_text)
                else:
                    completion_tokens = [ord(char) for char in completion_text]
                generated_tokens = prompt_tokens + completion_tokens
            else:
                generated_tokens = list(raw_gen)
                completion_tokens = generated_tokens[len(prompt_tokens):]
        else:
            completion_tokens = []
            generated_tokens = list(prompt_tokens)

        if completion_text is None:
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                completion_text = self.tokenizer.decode(completion_tokens)
            else:
                completion_text = "".join(chr(token) for token in completion_tokens if 0 <= token <= 0x10FFFF)

        finish_reason = "stop" if any(token in completion_tokens for token in getattr(
            getattr(self.engine, "inference_config", None), "stop_token_ids", []
        )) else "length"

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
