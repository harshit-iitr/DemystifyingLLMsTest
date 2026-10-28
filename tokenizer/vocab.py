"""Vocabulary maps with collision-free IDs and isolated imported vocabularies."""
from typing import Dict, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    # Default tokenizers share dynamically learned word IDs for the standalone
    # encode/decode helpers. Explicit vocabulary dictionaries remain isolated.
    _shared_vocab: Dict[str, int] = {}
    _shared_id_to_token: Dict[int, str] = {}

    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._shared = vocab_dict is None
        source = Vocabulary._shared_vocab if self._shared else vocab_dict
        self._token_to_id = dict(source)
        if any(type(idx) is not int or idx < 0 for idx in self._token_to_id.values()):
            raise ValueError("Vocabulary IDs must be non-negative integers")
        if len(set(self._token_to_id.values())) != len(self._token_to_id):
            raise ValueError("Vocabulary IDs must be unique")
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}
        self.decode_separator = " "
        def special(token, preferred):
            if token in self._token_to_id:
                return self._token_to_id[token]
            return self.add_token(token, preferred if preferred not in self._id_to_token else None)
        self.pad_token_id = special(PAD_TOKEN, 0)
        self.unk_token_id = special(UNK_TOKEN, 1)
        self.bos_token_id = special(BOS_TOKEN, 50256)
        self.eos_token_id = self.bos_token_id

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            if idx is not None and idx != self._token_to_id[token]:
                raise ValueError("Token already has a different ID")
            return self._token_to_id[token]
        if idx is None and self._shared and token in Vocabulary._shared_vocab:
            idx = Vocabulary._shared_vocab[token]
        if idx is None:
            idx = 0
            occupied = self._id_to_token
            while idx in occupied or (self._shared and idx in Vocabulary._shared_id_to_token):
                idx += 1
        if type(idx) is not int or idx < 0:
            raise ValueError("Token ID must be a non-negative integer")
        existing = self._id_to_token.get(idx)
        if existing is not None and existing != token:
            raise ValueError("Token ID already belongs to another token")
        if self._shared and idx in Vocabulary._shared_id_to_token and Vocabulary._shared_id_to_token[idx] != token:
            raise ValueError("Token ID already belongs to another shared token")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        if self._shared:
            Vocabulary._shared_vocab[token] = idx
            Vocabulary._shared_id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        return self._token_to_id[token] if token in self._token_to_id else self.add_token(token)

    def id_to_token(self, idx: int) -> str:
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        if self._shared and idx in Vocabulary._shared_id_to_token:
            return Vocabulary._shared_id_to_token[idx]
        return f"<token_{idx}>"

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id or (self._shared and token in Vocabulary._shared_vocab)
