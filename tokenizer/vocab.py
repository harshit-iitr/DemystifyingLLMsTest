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
        self.add_token(BOS_TOKEN, self.bos_token_id)
        self.add_token(EOS_TOKEN, self.eos_token_id)

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = len(self._token_to_id)
        if idx < 0:
            raise ValueError(f"Token ID must be non-negative, got {idx}")
        existing_token = self._id_to_token.get(idx)
        if existing_token is not None and existing_token != token:
            raise ValueError(f"Token ID {idx} is already assigned to {existing_token!r}")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        return self.unk_token_id

    def id_to_token(self, idx: int) -> str:
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        return f"<token_{idx}>"

    def __len__(self) -> int:
        return max(self._id_to_token, default=-1) + 1

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
