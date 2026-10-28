"""
Batch inference processing and dynamic request aggregation.

Status: Clean — Red Herring (Complex padding, mask construction, and dynamic micro-batch scheduling).
"""

import time
import math
import threading
from typing import List, Tuple, Dict, Any, Optional
import torch

from utils.logging_utils import get_logger

logger = get_logger("batch_processor")


class BatchProcessor:
    """
    Handles tensor alignment, dynamic padding, and attention mask construction
    for multi-sequence batch inference.
    """

    def __init__(self, pad_id: int = 0, default_side: str = "left"):
        self.pad_id = pad_id
        self.default_side = default_side

    def pad_sequences(
        self,
        token_lists: List[List[int]],
        side: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a batch of variable-length token lists to a uniform rectangular tensor,
        returning both the padded token tensor and the corresponding attention mask.

        Args:
            token_lists: List of token ID sequences.
            side: 'left' (for causal decoder generation) or 'right'.
            device: Target device.

        Returns:
            Tuple of (padded_input_ids, attention_mask)
        """
        if not token_lists:
            empty = torch.empty((0, 0), dtype=torch.long, device=device)
            return empty, empty

        padding_side = side or self.default_side
        batch_size = len(token_lists)
        max_len = max(len(s) for s in token_lists)

        padded_tensor = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for i, seq in enumerate(token_lists):
            seq_len = len(seq)
            if seq_len == 0:
                continue
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            if padding_side == "left":
                padded_tensor[i, max_len - seq_len :] = seq_t
                attention_mask[i, max_len - seq_len :] = 1
            elif padding_side == "right":
                padded_tensor[i, :seq_len] = seq_t
                attention_mask[i, :seq_len] = 1
            else:
                raise ValueError(f"Invalid padding side '{padding_side}', must be 'left' or 'right'")

        return padded_tensor, attention_mask

    def unpad_sequences(
        self,
        padded_tensor: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Extract active sequence elements by masking out padding tokens.
        """
        batch_size = padded_tensor.size(0)
        unpadded = []
        for i in range(batch_size):
            valid_idx = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
            if len(valid_idx) > 0:
                unpadded.append(padded_tensor[i, valid_idx])
            else:
                unpadded.append(torch.empty(0, dtype=padded_tensor.dtype, device=padded_tensor.device))
        return unpadded

    def chunk_batch(
        self,
        tensor: torch.Tensor,
        chunk_size: int
    ) -> List[torch.Tensor]:
        """Split a large batch tensor into smaller micro-batch chunks."""
        return list(torch.split(tensor, chunk_size, dim=0))


class DynamicBatcher:
    """
    Asynchronous queue aggregator collecting incoming single-token requests
    into batched forward-pass execution slots.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        max_wait_ms: float = 20.0,
        pad_id: int = 0
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not math.isfinite(max_wait_ms) or max_wait_ms < 0:
            raise ValueError("max_wait_ms must be finite and non-negative")
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.processor = BatchProcessor(pad_id=pad_id, default_side="left")
        self._queue: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def enqueue(self, request_id: str, token_ids: List[int]) -> None:
        """Enqueue an incoming request."""
        with self._lock:
            self._queue.append({
                "id": request_id,
                "tokens": token_ids,
                "enqueued_at": time.perf_counter()
            })

    def ready_batch(self) -> Optional[Tuple[List[str], torch.Tensor, torch.Tensor]]:
        """
        Inspect the queue and extract a batch if either max_batch_size is reached
        or the oldest request has waited longer than max_wait_ms.
        """
        with self._lock:
            if not self._queue:
                return None

            oldest_wait = (time.perf_counter() - self._queue[0]["enqueued_at"]) * 1000.0
            if len(self._queue) < self.max_batch_size and oldest_wait < self.max_wait_ms:
                return None
            batch_items = self._queue[: self.max_batch_size]
            del self._queue[: self.max_batch_size]

        ids = [item["id"] for item in batch_items]
        sequences = [item["tokens"] for item in batch_items]
        padded, mask = self.processor.pad_sequences(sequences)
        return ids, padded, mask
