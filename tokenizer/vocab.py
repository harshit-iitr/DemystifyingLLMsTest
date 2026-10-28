"""Vocabulary manager."""
from typing import Dict, List, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)

        if PAD_TOKEN not in self._token_to_id:
            self.add_token(PAD_TOKEN, self._first_free_id(0))
        if UNK_TOKEN not in self._token_to_id:
            self.add_token(UNK_TOKEN, self._first_free_id(1))
        if BOS_TOKEN not in self._token_to_id:
            preferred_id = 50256 if 50256 not in self._id_to_token else self._first_free_id(2)
            self.add_token(BOS_TOKEN, preferred_id)

        self.pad_token_id = self._token_to_id[PAD_TOKEN]
        self.unk_token_id = self._token_to_id[UNK_TOKEN]
        self.bos_token_id = self._token_to_id[BOS_TOKEN]
        self.eos_token_id = self.bos_token_id

    def _first_free_id(self, start: int = 0) -> int:
        idx = start
        while idx in self._id_to_token:
            idx += 1
        return idx

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if not isinstance(token, str):
            raise TypeError(f"token must be a string, got {type(token).__name__}")
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = self._first_free_id()
        if idx < 0:
            raise ValueError(f"token ID must be non-negative, got {idx}")
        if idx in self._id_to_token and self._id_to_token[idx] != token:
            raise ValueError(f"Token ID {idx} is already assigned to {self._id_to_token[idx]!r}")
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
