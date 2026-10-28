"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
import math
import inspect
from typing import Dict, Any, Optional, List
from utils.logging_utils import get_logger

logger = get_logger("request_handler")


def _call_with_supported_kwargs(method: Any, kwargs: Dict[str, Any]) -> Any:
    """Call an engine method with the options its signature supports."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return method(**kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return method(**kwargs)
    accepted = {parameter.name for parameter in parameters}
    return method(**{key: value for key, value in kwargs.items() if key in accepted})


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
        sampling_options = {}
        if "top_k" in request_body:
            top_k = request_body["top_k"]
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
                raise ValueError("top_k must be a non-negative integer")
            sampling_options["top_k"] = top_k
        if "top_p" in request_body:
            top_p = request_body["top_p"]
            if not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or not math.isfinite(top_p) or not 0.0 <= top_p <= 1.0:
                raise ValueError("top_p must be in [0.0, 1.0]")
            sampling_options["top_p"] = top_p
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a non-negative number")

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = [ord(c) for c in prompt]
        model = getattr(self.engine, "model", None)
        model_config = getattr(model, "config", None) or getattr(self.engine, "model_config", None)
        if model_config is not None and hasattr(model_config, "vocab_size"):
            from tokenizer.encoding import fit_token_ids_to_vocab
            prompt_tokens = fit_token_ids_to_vocab(prompt_tokens, model_config.vocab_size)

        # Generate continuation
        generated_text = None
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generation_args = {
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                **sampling_options,
            }
            generated_tokens = _call_with_supported_kwargs(
                self.engine.generate_tokens, generation_args
            )
        elif self.engine is not None and hasattr(self.engine, "generate"):
            raw_gen = _call_with_supported_kwargs(
                self.engine.generate,
                {
                    "prompt": prompt,
                    "tokenizer": self.tokenizer,
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    **sampling_options,
                },
            )
            generated_text = raw_gen
            generated_tokens = prompt_tokens[:]
            if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                try:
                    completion_tokens = self.tokenizer.encode(raw_gen, add_bos=False)
                except TypeError:
                    completion_tokens = self.tokenizer.encode(raw_gen)
                    bos_token_id = getattr(getattr(self.tokenizer, "vocab", None), "bos_token_id", None)
                    if completion_tokens and completion_tokens[0] == bos_token_id:
                        completion_tokens = completion_tokens[1:]
                generated_tokens.extend(completion_tokens)
            elif isinstance(raw_gen, str):
                generated_tokens.extend(ord(character) for character in raw_gen)
        else:
            raise RuntimeError("No inference engine is available")

        if generated_text is not None:
            response_text = generated_text
        elif self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            response_text = self.tokenizer.decode(generated_tokens[len(prompt_tokens):])
        else:
            response_text = "".join(chr(t) for t in generated_tokens[len(prompt_tokens):] if 0 <= t <= 0x10FFFF)

        stop_ids = getattr(getattr(self.engine, "inference_config", None), "stop_token_ids", [])
        finish_reason = "stop" if generated_tokens and generated_tokens[-1] in stop_ids else "length"
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
