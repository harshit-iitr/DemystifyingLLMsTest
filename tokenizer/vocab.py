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
        self._add_missing_special(PAD_TOKEN, preferred_id=0)
        self._add_missing_special(UNK_TOKEN, preferred_id=1)
        self._add_missing_special(BOS_TOKEN, preferred_id=50256)
        self.pad_token_id = self._token_to_id[PAD_TOKEN]
        self.unk_token_id = self._token_to_id[UNK_TOKEN]
        self.bos_token_id = self._token_to_id[BOS_TOKEN]
        self.eos_token_id = self._token_to_id.get(EOS_TOKEN, self.bos_token_id)

    def _add_missing_special(self, token: str, preferred_id: int) -> None:
        if token in self._token_to_id:
            return
        idx = preferred_id if preferred_id not in self._id_to_token else max(self._id_to_token, default=-1) + 1
        self.add_token(token, idx)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = max(self._id_to_token, default=-1) + 1
        if idx in self._id_to_token and self._id_to_token[idx] != token:
            raise ValueError(f"Token id {idx} is already assigned")
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
