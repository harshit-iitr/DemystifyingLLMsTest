"""
Weight tying logic for Language Model head and Token Embeddings.

Status: Clean — Red Herring (Shares weights between input embedding and output projection;
        standard architectural technique in GPT-2 and Transformer LMs).
"""

import torch
import torch.nn as nn
from typing import Union, Any


def tie_weights(lm_head: Any, token_embedding: Any) -> None:
    """
    Tie the weights of the LM Head output projection to the token embedding matrix.

    Args:
        lm_head: LMHead or linear projection module.
        token_embedding: TokenEmbedding or nn.Embedding instance.
    """
    if hasattr(token_embedding, "embedding"):
        weight = token_embedding.embedding.weight
    elif hasattr(token_embedding, "weight"):
        weight = token_embedding.weight
    else:
        raise AttributeError("token_embedding does not possess a weight parameter")

    if hasattr(lm_head, "proj") and hasattr(lm_head.proj, "weight"):
        lm_head.proj.weight = weight
    elif hasattr(lm_head, "weight"):
        lm_head.weight = weight
    else:
        raise AttributeError("lm_head does not possess a projectable weight parameter")


def is_tied(lm_head: Any, token_embedding: Any) -> bool:
    """Check if weights between LM Head and Token Embedding are tied."""
    if hasattr(token_embedding, "embedding"):
        emb_weight = token_embedding.embedding.weight
    else:
        emb_weight = getattr(token_embedding, "weight", None)

    if hasattr(lm_head, "proj"):
        head_weight = lm_head.proj.weight
    else:
        head_weight = getattr(lm_head, "weight", None)

    if emb_weight is None or head_weight is None:
        return False

    return emb_weight is head_weight
