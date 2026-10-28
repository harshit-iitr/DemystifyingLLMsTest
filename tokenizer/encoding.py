"""Text encoding."""
from typing import List, Optional, Sequence
from .bpe import BPETokenizer
from .vocab import Vocabulary
from .special_tokens import BOS_TOKEN
from config.constants import UNK_TOKEN_ID


def fit_token_ids_to_vocab(
    token_ids: Sequence[int], vocab_size: int, unk_token_id: int = UNK_TOKEN_ID
) -> List[int]:
    """Map out-of-range tokenizer IDs to a valid unknown-token ID for a model."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    fallback_id = unk_token_id if 0 <= unk_token_id < vocab_size else 0
    fitted = []
    for token_id in token_ids:
        token_id = int(token_id)
        fitted.append(token_id if 0 <= token_id < vocab_size else fallback_id)
    return fitted


def encode(text: str, tokenizer: Optional[BPETokenizer] = None, vocab: Optional[Vocabulary] = None, add_bos: bool = True) -> List[int]:
    """
    Encode input string into token IDs.
    """
    if tokenizer is None:
        tokenizer = BPETokenizer(vocab=vocab)
    vocab = tokenizer.vocab

    tokens = tokenizer.tokenize(text)
    token_ids = [vocab.token_to_id(t) for t in tokens]

    if add_bos:
        token_ids = [vocab.bos_token_id] + token_ids

    return token_ids
