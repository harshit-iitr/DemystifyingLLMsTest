"""Token decoding."""
from typing import List, Optional

from .vocab import Vocabulary
from .special_tokens import SPECIAL_TOKENS


def decode(
    ids: List[int],
    vocab: Optional[Vocabulary] = None,
    skip_special_tokens: bool = True,
) -> str:
    """
    Decode token IDs back into text.

    Special tokens are filtered by identity, not removed based on position.
    """
    if vocab is None:
        vocab = Vocabulary()

    tokens = [vocab.id_to_token(int(token_id)) for token_id in ids]

    if skip_special_tokens:
        tokens = [token for token in tokens if token not in SPECIAL_TOKENS]

    return " ".join(tokens)