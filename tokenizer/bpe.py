"""Module implementation."""
from typing import List, Dict, Tuple, Optional
import re
from .vocab import Vocabulary
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN


class BPETokenizer:
    def __init__(self, vocab: Optional[Vocabulary] = None, merges: Optional[List[Tuple[str, str]]] = None):
        self.vocab = vocab if vocab is not None else Vocabulary()
        self.merges: List[Tuple[str, str]] = merges or []
        self.vocab.decode_separator = "" if self.merges else " "

    @property
    def vocab_size(self) -> int:
        return self.vocab.max_size or (max(self.vocab._id_to_token, default=-1) + 1)

    def _get_stats(self, pieces: List[str]) -> Dict[Tuple[str, str], int]:
        pairs: Dict[Tuple[str, str], int] = {}
        for i in range(len(pieces) - 1):
            pair = (pieces[i], pieces[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def _merge_word(self, pieces: List[str], pair: Tuple[str, str]) -> List[str]:
        new_pieces = []
        i = 0
        while i < len(pieces):
            if i < len(pieces) - 1 and pieces[i] == pair[0] and pieces[i + 1] == pair[1]:
                new_pieces.append(pair[0] + pair[1])
                i += 2
            else:
                new_pieces.append(pieces[i])
                i += 1
        return new_pieces

    def bpe(self, token: str) -> List[str]:
        if not self.merges or token in self.vocab:
            return [token]
        pieces = list(token)
        if len(pieces) <= 1:
            return pieces

        ranks = {pair: rank for rank, pair in reversed(list(enumerate(self.merges)))}
        while len(pieces) > 1:
            pairs = self._get_stats(pieces)
            candidates = [pair for pair in pairs if pair in ranks]
            if not candidates:
                break
            pair = min(candidates, key=ranks.get)
            pieces = self._merge_word(pieces, pair)

        return pieces

    def tokenize(self, text: str) -> List[str]:
        """Return word pieces, preserving the original public token-list API."""
        return [piece for word in text.split(" ") if word for piece in self.bpe(word)]

    def _encoding_pieces(self, text: str) -> List[str]:
        if not self.merges:
            return self.tokenize(text)
        # Encoding needs word boundaries to reconstruct merged pieces. These
        # whitespace tokens are internal and do not change tokenize() results.
        return [token for piece in re.findall(r"\s+|\S+", text)
                for token in ([piece] if piece.isspace() else self.bpe(piece))]

    def bind_to_model(self, vocab_size: int) -> None:
        """Bind a dynamic tokenizer to a compact model's explicit ID space.

        Imported vocabulary IDs are preserved and validated. Only the default
        dynamic word vocabulary is rebuilt with in-range special-token IDs.
        """
        if self.vocab._shared and self.vocab_size > vocab_size:
            old = self.vocab
            bounded = Vocabulary({}, max_size=vocab_size)
            # Start a fresh dynamic ID space: unrelated words learned by other
            # default tokenizers must not consume this compact model's capacity.
            bounded.decode_separator = old.decode_separator
            self.vocab = bounded
        else:
            if any(idx >= vocab_size for idx in self.vocab._id_to_token):
                raise ValueError("Tokenizer vocabulary contains IDs outside the model vocabulary")
            self.vocab.max_size = vocab_size

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        from .encoding import encode
        return encode(text, tokenizer=self, add_bos=add_bos)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        from .decoding import decode
        return decode(ids, vocab=self.vocab, skip_special_tokens=skip_special_tokens)
