"""Text encoding."""
from typing import List, Optional
from .bpe import BPETokenizer
from .vocab import Vocabulary


def encode(text: str, tokenizer: Optional[BPETokenizer] = None, vocab: Optional[Vocabulary] = None, add_bos: bool = True) -> List[int]:
    """
    Encode input string into token IDs.
    """
    if tokenizer is None:
        tokenizer = BPETokenizer(vocab=vocab)
    vocab = tokenizer.vocab

    tokens = tokenizer.tokenize(text)
    token_ids = [vocab.token_to_id(t) for t in tokens]

    if add_bos and vocab.bos_token_id is not None:
        token_ids = [vocab.bos_token_id] + token_ids

    tokenizer.vocab_size = max(vocab._id_to_token, default=-1) + 1

    return token_ids
