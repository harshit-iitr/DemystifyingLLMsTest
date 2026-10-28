"""Vocabulary manager."""
import json
from pathlib import Path
from typing import Dict, List, Optional
from config.constants import DEFAULT_VOCAB_PATH
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    vocab_size = 50257

    def __init__(
        self,
        vocab_dict: Optional[Dict[str, int]] = None,
        vocab_path: Optional[str] = None,
    ):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        if vocab_dict is None:
            path = Path(vocab_path or DEFAULT_VOCAB_PATH)
            if path.is_file():
                with path.open("r", encoding="utf-8") as vocab_file:
                    vocab_dict = json.load(vocab_file)

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)

        # Add special tokens when their preferred IDs are available. A supplied
        # vocabulary remains authoritative when it already uses those IDs.
        self._ensure_special_token(PAD_TOKEN, 0)
        self._ensure_special_token(UNK_TOKEN, 1)
        self._ensure_special_token(BOS_TOKEN, 50256)
        self._ensure_special_token(EOS_TOKEN, 50256)

        self.pad_token_id = self._token_to_id.get(PAD_TOKEN)
        self.unk_token_id = self._token_to_id.get(UNK_TOKEN)
        self.bos_token_id = self._token_to_id.get(BOS_TOKEN)
        self.eos_token_id = self._token_to_id.get(EOS_TOKEN)
        self._id_to_token = {
            token_id: token for token, token_id in self._token_to_id.items()
        }

    def _ensure_special_token(self, token: str, preferred_id: int) -> None:
        if token in self._token_to_id:
            return
        if preferred_id not in self._id_to_token:
            self.add_token(token, preferred_id)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = 0
            while idx in self._id_to_token:
                idx += 1
        else:
            idx = int(idx)
        if idx < 0:
            raise ValueError(f"token ID must be non-negative, got {idx}")
        existing_token = self._id_to_token.get(idx)
        if existing_token is not None and existing_token != token:
            raise ValueError(f"token ID {idx} is already assigned to {existing_token!r}")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        return self.add_token(token)

    def id_to_token(self, idx: int) -> str:
        idx = int(idx)
        return self._id_to_token.get(idx, "")

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
