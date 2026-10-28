from dataclasses import asdict, dataclass, field
from typing import Any, Dict
import json


@dataclass
class ModelConfig:
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    vocab_size: int = 50256
    max_seq_len: int = 1024
    ffn_multiplier: int = 4
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_rotary: bool = False
    tie_weights: bool = True

    d_head: int = field(init=False)
    d_ffn: int = field(init=False)

    def __post_init__(self):
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        self.d_head = self.d_model // self.n_heads
        self.d_ffn = self.d_model * self.ffn_multiplier

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        fields = cls.__dataclass_fields__
        clean_data = {
            key: value
            for key, value in data.items()
            if key in fields and key not in {"d_head", "d_ffn"}
        }
        return cls(**clean_data)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelConfig":
        return cls.from_dict(json.loads(json_str))
