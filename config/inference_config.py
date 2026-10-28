"""
Inference and text generation configuration.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
import math


@dataclass
class InferenceConfig:
    max_new_tokens: int = 20
    use_kv_cache: bool = True
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_token_ids: List[int] = field(default_factory=lambda: [50256])
    early_stopping: bool = True
    beam_width: int = 1         # 1 indicates greedy / stochastic sampling (non-beam)
    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate inference parameters."""
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool) or self.max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {self.max_new_tokens}")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0.0
        ):
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(self.top_p)
            or not (0.0 <= self.top_p <= 1.0)
        ):
            raise ValueError(f"top_p must be in [0.0, 1.0], got {self.top_p}")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if (
            isinstance(self.repetition_penalty, bool)
            or not isinstance(self.repetition_penalty, (int, float))
            or not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0.0
        ):
            raise ValueError(f"repetition_penalty must be positive, got {self.repetition_penalty}")
        if not isinstance(self.beam_width, int) or isinstance(self.beam_width, bool) or self.beam_width < 1:
            raise ValueError(f"beam_width must be at least 1, got {self.beam_width}")
        if not isinstance(self.stop_token_ids, list) or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in self.stop_token_ids
        ):
            raise ValueError("stop_token_ids must be a list of non-negative integers")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError(f"seed must be an integer or None, got {self.seed!r}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceConfig":
        """Load configuration from dictionary."""
        valid_keys = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_keys)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "InferenceConfig":
        """Instantiate configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))

