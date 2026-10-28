"""Vocabulary manager."""
from typing import Dict, List, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        
        self.pad_token_id: int = 0
        self.unk_token_id: int = 1
        self.bos_token_id: int = 50256
        self.eos_token_id: int = 50256

        self.add_token(PAD_TOKEN, 0)
        self.add_token(UNK_TOKEN, 1)
        self.add_token(EOS_TOKEN, 50256)

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        if token in self._token_to_id:
            existing_idx = self._token_to_id[token]
            if idx is not None and idx != existing_idx:
                raise ValueError(f"Token {token!r} is already assigned to ID {existing_idx}")
            return existing_idx
        if idx is None:
            idx = 2
            while idx in self._id_to_token:
                idx += 1
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            raise ValueError("token ID must be a non-negative integer")
        if idx in self._id_to_token:
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
        return max(self._id_to_token, default=-1) + 1

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
