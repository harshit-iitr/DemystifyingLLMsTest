"""Token decoding."""
from typing import List, Optional
from .vocab import Vocabulary


def decode(ids: List[int], vocab: Optional[Vocabulary] = None, skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs back to a string.
    """
    if vocab is None:
        vocab = Vocabulary()

    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, int):
        ids = [ids]

    special_ids = {
        token_id
        for token_id in (
            vocab.pad_token_id,
            vocab.unk_token_id,
            vocab.bos_token_id,
            vocab.eos_token_id,
        )
        if token_id is not None
    }
    token_ids = [int(token_id) for token_id in ids]
    if skip_special_tokens:
        token_ids = [token_id for token_id in token_ids if token_id not in special_ids]

    id_to_token = getattr(vocab, "_id_to_token", None)
    if not id_to_token:
        id_to_token = {int(token_id): token for token, token_id in vocab._token_to_id.items()}
    tokens = [id_to_token.get(token_id, "") for token_id in token_ids]
    # Byte-level tokenizers can map IDs to byte fragments that are not valid
    # UTF-8 individually. Join the complete sequence before decoding it.
    if any(isinstance(token, bytes) for token in tokens):
        byte_sequence = b"".join(
            token if isinstance(token, bytes) else token.encode("utf-8")
            for token in tokens
        )
        return byte_sequence.decode("utf-8", errors="replace").replace("Ġ", " ")

    return "".join(str(token) for token in tokens).replace("Ġ", " ")
