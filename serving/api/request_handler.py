"""
HTTP API request parsing, validation, and execution handler.
"""

import inspect
import time
import uuid
from typing import Dict, Any, Optional, List
import torch
from utils.logging_utils import get_logger

logger = get_logger("request_handler")


def _call_supported(method: Any, kwargs: Dict[str, Any]) -> Any:
    """Call an engine method with the keyword arguments its signature accepts."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(**kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return method(**kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return method(**accepted)


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
            prompt_tokens = [ord(c) for c in prompt]

        if self.engine is None:
            raise RuntimeError("No inference engine is configured")
        if hasattr(self.engine, "generate_tokens"):
            generation_args = {
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            }
            if "top_k" in request_body:
                generation_args["top_k"] = request_body["top_k"]
            if "top_p" in request_body:
                generation_args["top_p"] = request_body["top_p"]
            generated_tokens = _call_supported(self.engine.generate_tokens, generation_args)
            if isinstance(generated_tokens, torch.Tensor):
                generated_tokens = generated_tokens.tolist()
            if generated_tokens and isinstance(generated_tokens[0], list):
                generated_tokens = generated_tokens[0]
            if self.tokenizer is None or not hasattr(self.tokenizer, "decode"):
                raise ValueError("A tokenizer with decode() is required")
            has_prompt_prefix = generated_tokens[:len(prompt_tokens)] == prompt_tokens
            completion_tokens = generated_tokens[len(prompt_tokens):] if has_prompt_prefix else generated_tokens
            response_text = self.tokenizer.decode(completion_tokens)
            completion_count = len(completion_tokens)
            stop_ids = set(getattr(getattr(self.engine, "inference_config", None), "stop_token_ids", []))
            if not stop_ids:
                from config.constants import EOS_TOKEN_ID
                stop_ids.add(EOS_TOKEN_ID)
            finish_reason = "stop" if completion_tokens and completion_tokens[-1] in stop_ids else "length"
        elif hasattr(self.engine, "generate"):
            response_text = _call_supported(
                self.engine.generate,
                {"prompt": prompt, "tokenizer": self.tokenizer,
                 "max_new_tokens": max_tokens, "temperature": temperature},
            )
            if isinstance(response_text, str) and response_text.startswith(prompt):
                response_text = response_text[len(prompt):]
            completion_count = len(self.tokenizer.encode(response_text)) if self.tokenizer is not None and hasattr(self.tokenizer, "encode") else max_tokens
            finish_reason = "length"
        else:
            raise RuntimeError("The configured inference engine cannot generate text")

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
