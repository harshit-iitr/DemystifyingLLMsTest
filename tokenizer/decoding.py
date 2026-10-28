"""Token decoding."""
from typing import List, Optional
import torch
from .vocab import Vocabulary
from .special_tokens import SPECIAL_TOKENS


def decode(ids: List[int], vocab: Optional[Vocabulary] = None, skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs back to a string.
    """
    if vocab is None:
        vocab = Vocabulary()

    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().reshape(-1).tolist()
    tokens = [vocab.id_to_token(int(i)) for i in ids]

    if skip_special_tokens:
        tokens = [t for t in tokens if t not in SPECIAL_TOKENS]

    return " ".join(tokens)
