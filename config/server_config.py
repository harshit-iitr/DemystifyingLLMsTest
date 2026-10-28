"""
API server configuration.

Status: Clean
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import json


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    timeout: int = 30
    max_batch_size: int = 16
    max_queue_size: int = 64
    log_level: str = "INFO"
    cors_allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    keep_alive_timeout: int = 5

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must not be empty")
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if self.timeout <= 0 or self.keep_alive_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_batch_size <= 0 or self.max_queue_size <= 0:
            raise ValueError("batch and queue sizes must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerConfig":
        """Instantiate configuration from dictionary."""
        valid_keys = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_keys)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ServerConfig":
        """Instantiate configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))

