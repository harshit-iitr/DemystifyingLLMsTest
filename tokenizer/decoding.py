"""Token decoding."""
from typing import List, Optional
from .vocab import Vocabulary
from .special_tokens import SPECIAL_TOKENS


def decode(ids: List[int], vocab: Optional[Vocabulary] = None, skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs back to a string.

    Accepts any iterable of IDs, including 1-D or 2-D tensors (flattened in
    row-major order).
    """
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().reshape(-1).tolist()
    elif not isinstance(ids, (list, tuple)):
        ids = list(ids)

    if vocab is None:
        vocab = Vocabulary()

    tokens = [vocab.id_to_token(int(i)) for i in ids]
    tokens = [token for token in tokens if token is not None]

    if skip_special_tokens:
        tokens = [t for t in tokens if t not in SPECIAL_TOKENS]

    return " ".join(tokens)
