"""Vocabulary manager."""
from typing import Dict, List, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
from config.constants import PAD_TOKEN_ID, UNK_TOKEN_ID, BOS_TOKEN_ID


class Vocabulary:
    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)

        if PAD_TOKEN not in self._token_to_id:
            pad_id = PAD_TOKEN_ID if PAD_TOKEN_ID not in self._id_to_token else None
            self.add_token(PAD_TOKEN, pad_id)
        if UNK_TOKEN not in self._token_to_id:
            unk_id = UNK_TOKEN_ID if UNK_TOKEN_ID not in self._id_to_token else None
            self.add_token(UNK_TOKEN, unk_id)
        if BOS_TOKEN not in self._token_to_id:
            bos_id = BOS_TOKEN_ID if vocab_dict is None and BOS_TOKEN_ID not in self._id_to_token else None
            self.add_token(BOS_TOKEN, bos_id)

        self.pad_token_id = self._token_to_id[PAD_TOKEN]
        self.unk_token_id = self._token_to_id[UNK_TOKEN]
        self.bos_token_id = self._token_to_id[BOS_TOKEN]
        self.eos_token_id = self._token_to_id[EOS_TOKEN]

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = 0
            while idx in self._id_to_token:
                idx += 1
        if not isinstance(idx, int) or idx < 0:
            raise ValueError(f"Token ID must be a non-negative integer, got {idx!r}")
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
        return self._id_to_token.get(idx, UNK_TOKEN)

    def __len__(self) -> int:
        return len(self._token_to_id)

    @property
    def vocab_size(self) -> int:
        """Return the minimum embedding table size that contains every token ID."""
        return max(self._id_to_token, default=-1) + 1

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
