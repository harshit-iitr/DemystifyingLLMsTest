"""Vocabulary manager."""
from typing import Dict, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
from config.constants import PAD_TOKEN_ID, UNK_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID


class Vocabulary:
    _shared_vocab: Dict[str, int] = {}
    _shared_id_to_token: Dict[int, str] = {}

    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        
        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)
            for token, idx in (
                (PAD_TOKEN, PAD_TOKEN_ID),
                (UNK_TOKEN, UNK_TOKEN_ID),
                (BOS_TOKEN, BOS_TOKEN_ID),
            ):
                if token not in self._token_to_id:
                    self.add_token(token, idx)
        else:
            self.add_token(PAD_TOKEN, PAD_TOKEN_ID)
            self.add_token(UNK_TOKEN, UNK_TOKEN_ID)
            self.add_token(BOS_TOKEN, BOS_TOKEN_ID)
            for token, idx in Vocabulary._shared_vocab.items():
                if token not in self._token_to_id and idx not in self._id_to_token:
                    self.add_token(token, idx)

        self.pad_token_id = self._token_to_id.get(PAD_TOKEN, PAD_TOKEN_ID)
        self.unk_token_id = self._token_to_id.get(UNK_TOKEN, UNK_TOKEN_ID)
        self.bos_token_id = self._token_to_id.get(BOS_TOKEN, BOS_TOKEN_ID)
        self.eos_token_id = self._token_to_id.get(EOS_TOKEN, EOS_TOKEN_ID)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            if idx is not None and self._token_to_id[token] != idx:
                raise ValueError(f"Token {token!r} already has id {self._token_to_id[token]}")
            return self._token_to_id[token]
        if idx is None:
            idx = next((candidate for candidate in range(2, 50256) if candidate not in self._id_to_token),
                       max(self._id_to_token, default=-1) + 1)
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise ValueError("token id must be an integer")
        if idx < 0:
            raise ValueError("token id must be non-negative")
        if idx in self._id_to_token and self._id_to_token[idx] != token:
            raise ValueError(f"Token id {idx} is already assigned")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        shared_token_id = Vocabulary._shared_vocab.get(token)
        shared_id_token = Vocabulary._shared_id_to_token.get(idx)
        if shared_token_id in (None, idx) and shared_id_token in (None, token):
            Vocabulary._shared_vocab[token] = idx
            Vocabulary._shared_id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        shared_id = Vocabulary._shared_vocab.get(token)
        if shared_id is not None and (shared_id not in self._id_to_token or self._id_to_token[shared_id] == token):
            self._token_to_id[token] = shared_id
            self._id_to_token[shared_id] = token
            return shared_id
        return self.add_token(token)

    def id_to_token(self, idx: int) -> str:
        idx = int(idx)
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        if idx in Vocabulary._shared_id_to_token:
            return Vocabulary._shared_id_to_token[idx]
        return f"<token_{idx}>"

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        if token in self._token_to_id:
            return True
        shared_id = Vocabulary._shared_vocab.get(token)
        return shared_id is not None and (
            shared_id not in self._id_to_token or self._id_to_token[shared_id] == token
        )
