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
            self.pad_token_id = self._token_to_id.get(PAD_TOKEN)
            if self.pad_token_id is None:
                self.pad_token_id = self.add_token(PAD_TOKEN)
            self.unk_token_id = self._token_to_id.get(UNK_TOKEN)
            if self.unk_token_id is None:
                self.unk_token_id = self.add_token(UNK_TOKEN)
            self.bos_token_id = self._token_to_id.get(BOS_TOKEN)
            if self.bos_token_id is None:
                self.bos_token_id = self.add_token(BOS_TOKEN, 50256 if 50256 not in self._id_to_token else None)
        else:
            self.pad_token_id = self.add_token(PAD_TOKEN, 0)
            self.unk_token_id = self.add_token(UNK_TOKEN, 1)
            self.bos_token_id = self.add_token(BOS_TOKEN, 50256)

        self.eos_token_id = self.add_token(EOS_TOKEN, 50256)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = 0
            while idx in self._id_to_token:
                idx += 1
        elif idx in self._id_to_token:
            raise ValueError(f"Token ID {idx} is already assigned")
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
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
