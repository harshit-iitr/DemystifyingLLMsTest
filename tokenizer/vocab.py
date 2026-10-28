"""Vocabulary manager."""
from typing import Dict, List, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    _shared_vocab: Dict[str, int] = {}
    _shared_id_to_token: Dict[int, str] = {}

    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)
        else:
            for token, idx in Vocabulary._shared_vocab.items():
                self.add_token(token, idx)

        self.pad_token_id = self._ensure_special(PAD_TOKEN, 0)
        self.unk_token_id = self._ensure_special(UNK_TOKEN, 1)
        self.bos_token_id = self._ensure_special(BOS_TOKEN, 50256)
        self.eos_token_id = self.bos_token_id

    def _ensure_special(self, token: str, preferred_id: int) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        idx = preferred_id if preferred_id not in self._id_to_token else None
        return self.add_token(token, idx)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if idx is not None and idx < 0:
            raise ValueError("token ID must be non-negative")
        if token in self._token_to_id:
            if idx is not None and self._token_to_id[token] != idx:
                raise ValueError(f"token {token!r} already has ID {self._token_to_id[token]}")
            return self._token_to_id[token]
        if idx is None:
            idx = 0
            while idx in self._id_to_token:
                idx += 1
        if idx in self._id_to_token:
            raise ValueError(f"token ID {idx} is already assigned to {self._id_to_token[idx]!r}")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        return self.add_token(token)

    def id_to_token(self, idx: int) -> str:
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        return f"<token_{idx}>"

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
