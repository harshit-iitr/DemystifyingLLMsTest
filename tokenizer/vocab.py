"""Vocabulary manager."""
from typing import Dict, List, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._dynamic = vocab_dict is None
        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)
            self.pad_token_id = self._token_to_id.get(PAD_TOKEN, 0)
            self.unk_token_id = self._token_to_id.get(UNK_TOKEN, self.pad_token_id)
            self.bos_token_id = self._token_to_id.get(BOS_TOKEN)
            self.eos_token_id = self._token_to_id.get(EOS_TOKEN, self.bos_token_id)
        else:
            self.pad_token_id = self.add_token(PAD_TOKEN, 0)
            self.unk_token_id = self.add_token(UNK_TOKEN, 1)
            self.bos_token_id = self.add_token(BOS_TOKEN, 50256)
            self.eos_token_id = self.bos_token_id

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = 0
            while idx in self._id_to_token:
                idx += 1
        if idx in self._id_to_token and self._id_to_token[idx] != token:
            raise ValueError(f"Token id {idx} is already assigned to {self._id_to_token[idx]!r}")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if not self._dynamic:
            return self.unk_token_id
        return self.add_token(token)

    def id_to_token(self, idx: int) -> str:
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        return f"<token_{idx}>"

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
