"""Module implementation."""
from typing import List, Dict, Tuple, Optional
from .vocab import Vocabulary


class BPETokenizer:
    def __init__(self, vocab: Optional[Vocabulary] = None, merges: Optional[List[Tuple[str, str]]] = None):
        self.vocab = vocab if vocab is not None else Vocabulary()
        self.merges: List[Tuple[str, str]] = list(merges) if merges is not None else []
        self.vocab_size = 50257

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
        if token in self.vocab:
            return [token]
        pieces = list(token)
        if len(pieces) <= 1:
            return pieces

        ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        while len(pieces) > 1:
            available = [
                (ranks[(pieces[i], pieces[i + 1])], (pieces[i], pieces[i + 1]))
                for i in range(len(pieces) - 1)
                if (pieces[i], pieces[i + 1]) in ranks
            ]
            if not available:
                break
            _, best_pair = min(available, key=lambda item: item[0])
            pieces = self._merge_word(pieces, best_pair)

        return pieces

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        tokens = self.bpe(text)
        return tokens

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        from .encoding import encode
        return encode(text, tokenizer=self, add_bos=add_bos)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        from .decoding import decode
        return decode(ids, vocab=self.vocab, skip_special_tokens=skip_special_tokens)
