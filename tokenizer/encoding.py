"""Text encoding."""
from typing import List, Optional

from .bpe import BPETokenizer
from .vocab import Vocabulary
from .special_tokens import BOS_TOKEN


def encode(
    text: str,
    tokenizer: Optional[BPETokenizer] = None,
    vocab: Optional[Vocabulary] = None,
    add_bos: bool = True,
) -> List[int]:
    """
    Encode text into token IDs, optionally prepending a BOS token.
    """
    if tokenizer is None:
        tokenizer = BPETokenizer(vocab=vocab)

    vocab = tokenizer.vocab

    tokens = tokenizer.tokenize(text)
    token_ids = [vocab.token_to_id(token) for token in tokens]

    if add_bos:
        token_ids.insert(0, vocab.token_to_id(BOS_TOKEN))

    return token_ids