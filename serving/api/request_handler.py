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
        completion_count = 0

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
                top_k=request_body.get("top_k"),
                top_p=request_body.get("top_p"),
            )
            if hasattr(generated_tokens, "detach"):
                generated_tokens = generated_tokens.detach().reshape(-1).tolist()
            else:
                generated_tokens = list(generated_tokens)
        elif self.engine is not None and hasattr(self.engine, "generate"):
            raw_gen = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
            response_text = raw_gen[len(prompt):] if isinstance(raw_gen, str) and raw_gen.startswith(prompt) else raw_gen
            generated_tokens = []
            if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                try:
                    completion_count = len(self.tokenizer.encode(response_text, add_bos=False))
                except TypeError:
                    completion_count = len(self.tokenizer.encode(response_text))
        else:
            raise RuntimeError("No inference engine is available")

        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            if generated_tokens[:len(prompt_tokens)] == prompt_tokens:
                completion_tokens = generated_tokens[len(prompt_tokens):]
            else:
                completion_tokens = generated_tokens
            stop_ids = set(getattr(
                getattr(self.engine, "inference_config", None),
                "stop_token_ids",
                [],
            ))
            vocab = getattr(self.tokenizer, "vocab", None)
            if vocab is not None and hasattr(vocab, "eos_token_id"):
                stop_ids.add(vocab.eos_token_id)
            finish_reason = "length"
            completion_count = len(completion_tokens)
            for index, token_id in enumerate(completion_tokens):
                if token_id in stop_ids:
                    completion_tokens = completion_tokens[:index]
                    finish_reason = "stop"
                    break
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                response_text = self.tokenizer.decode(completion_tokens)
            else:
                response_text = " ".join(str(token) for token in completion_tokens)
        else:
            finish_reason = "length"

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
                    "ready": True
                }
            ]
        }
