"""Token decoding."""
from typing import List, Optional
from .vocab import Vocabulary
from .special_tokens import SPECIAL_TOKENS


def decode(ids: List[int], vocab: Optional[Vocabulary] = None, skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs back to a string.
    """
    if vocab is None:
        vocab = Vocabulary()

    tokens = [vocab.id_to_token(i) for i in ids]

    if skip_special_tokens:
        if tokens and vocab.token_to_id(tokens[0]) == vocab.bos_token_id:
            tokens = tokens[1:]
        if tokens and vocab.token_to_id(tokens[-1]) == vocab.eos_token_id:
            tokens = tokens[:-1]
        tokens = [t for t in tokens if t not in SPECIAL_TOKENS]

    return " ".join(tokens)
