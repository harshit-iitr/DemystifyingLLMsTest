"""Vocabulary manager."""
from typing import Dict, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)

        for token, preferred_id in ((PAD_TOKEN, 0), (UNK_TOKEN, 1), (BOS_TOKEN, 50256)):
            if token not in self._token_to_id:
                token_id = preferred_id if preferred_id not in self._id_to_token else max(self._id_to_token, default=-1) + 1
                self.add_token(token, token_id)

        self.pad_token_id = self._token_to_id[PAD_TOKEN]
        self.unk_token_id = self._token_to_id[UNK_TOKEN]
        self.bos_token_id = self._token_to_id[BOS_TOKEN]
        self.eos_token_id = self._token_to_id[EOS_TOKEN]

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = max(self._id_to_token, default=-1) + 1
        if idx < 0:
            raise ValueError("Token IDs must be non-negative")
        existing_token = self._id_to_token.get(idx)
        if existing_token is not None and existing_token != token:
            raise ValueError(f"Token ID {idx} is already assigned to {existing_token!r}")
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
