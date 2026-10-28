"""Token decoding."""
from typing import List, Optional
from .vocab import Vocabulary
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


def decode(ids: List[int], vocab: Optional[Vocabulary] = None, skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs back to a string.
    """
    if vocab is None:
        vocab = Vocabulary()

    tokens = []
    special_tokens = {PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN}
    token_to_id = getattr(vocab, "_token_to_id", {})
    shared_token_to_id = getattr(type(vocab), "_shared_vocab", {})
    for token_id in ids:
        token_id = int(token_id)
        token = vocab.id_to_token(token_id)
        # If a legacy/custom vocabulary maps multiple strings to one ID, prefer
        # the ordinary token over a special-token alias when decoding content.
        if token in special_tokens:
            token = next(
                (
                    candidate
                    for mapping in (token_to_id, shared_token_to_id)
                    for candidate, candidate_id in mapping.items()
                    if candidate_id == token_id and candidate not in special_tokens
                ),
                token,
            )
        tokens.append(token)

    if skip_special_tokens:
        tokens = [
            token
            for token in tokens
            if token not in special_tokens
        ]

    return " ".join(tokens)
