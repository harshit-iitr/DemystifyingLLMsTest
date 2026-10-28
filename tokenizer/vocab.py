"""Vocabulary manager."""
from typing import Dict, List, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        
        self.pad_token_id: int = self.add_token(PAD_TOKEN, 0)
        self.unk_token_id: int = self.add_token(UNK_TOKEN, 1)
        # GPT-2 uses the same end-of-text token for both BOS and EOS.
        self.bos_token_id: int = self.add_token(BOS_TOKEN, 50256)
        self.eos_token_id: int = self.add_token(EOS_TOKEN, 50256)

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = 0
            while idx in self._id_to_token:
                idx += 1
        if idx in self._id_to_token:
            raise ValueError(f"Token ID {idx} is already assigned to '{self._id_to_token[idx]}'")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        return self.add_token(token)

    def id_to_token(self, idx: int) -> Optional[str]:
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        return None

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
