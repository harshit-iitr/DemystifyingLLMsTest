"""Vocabulary manager."""
from typing import Dict, Optional
from config.constants import PAD_TOKEN_ID, UNK_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        
        self.pad_token_id = PAD_TOKEN_ID
        self.unk_token_id = UNK_TOKEN_ID
        self.bos_token_id = BOS_TOKEN_ID
        self.eos_token_id = EOS_TOKEN_ID

        self.add_token(PAD_TOKEN, self.pad_token_id)
        self.add_token(UNK_TOKEN, self.unk_token_id)
        self.add_token(EOS_TOKEN, self.eos_token_id)

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
        if idx < 0:
            raise ValueError("token IDs must be non-negative")
        if idx in self._id_to_token and self._id_to_token[idx] != token:
            raise ValueError(f"token ID {idx} is already assigned to {self._id_to_token[idx]!r}")
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


# The convenience encode/decode functions share one vocabulary. Explicitly
# constructed Vocabulary and BPETokenizer instances remain independent.
_default_vocab = None


def get_default_vocab() -> Vocabulary:
    global _default_vocab
    if _default_vocab is None:
        _default_vocab = Vocabulary()
    return _default_vocab
