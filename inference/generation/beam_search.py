"""Beam search for single-sequence autoregressive decoding."""
from typing import List, Optional, Tuple

import torch


class BeamSearch:
    def __init__(self, beam_width: int = 4, max_length: int = 50, eos_token_id: Optional[int] = None):
        if not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width < 1:
            raise ValueError("beam_width must be a positive integer")
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 0:
            raise ValueError("max_length must be a non-negative integer")
        if eos_token_id is not None and (not isinstance(eos_token_id, int) or eos_token_id < 0):
            raise ValueError("eos_token_id must be a non-negative integer or None")
        self.beam_width = beam_width
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def search(self, model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the highest-scoring sequence, expanding globally across beams.

        The implementation intentionally accepts one prompt at a time. A beam is
        represented as ``(sequence, cumulative_log_probability)``.
        """
        if input_ids.ndim != 2 or input_ids.size(0) != 1:
            raise ValueError("BeamSearch.search expects input_ids with shape (1, seq_len)")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids must contain at least one token")

        beams: List[Tuple[torch.Tensor, float]] = [(input_ids, 0.0)]
        completed: List[Tuple[torch.Tensor, float]] = []

        for _ in range(self.max_length):
            candidates: List[Tuple[torch.Tensor, float]] = []
            for seq, score in beams:
                if self.eos_token_id is not None and seq[0, -1].item() == self.eos_token_id:
                    completed.append((seq, score))
                    continue

                logits = model(seq)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if logits.ndim != 3 or logits.size(0) != 1:
                    raise ValueError("model must return logits with shape (1, seq_len, vocab_size)")
                log_probs = torch.log_softmax(logits[0, -1], dim=-1)
                count = min(self.beam_width, log_probs.numel())
                token_scores, token_ids = torch.topk(log_probs, count)
                for token_score, token_id in zip(token_scores, token_ids):
                    token = token_id.reshape(1, 1).to(device=seq.device, dtype=seq.dtype)
                    candidates.append((torch.cat((seq, token), dim=-1), score + token_score.item()))

            if not candidates:
                break
            candidates.sort(key=lambda item: item[1], reverse=True)
            beams = []
            for candidate in candidates[: self.beam_width]:
                is_eos = self.eos_token_id is not None and candidate[0][0, -1].item() == self.eos_token_id
                if is_eos:
                    completed.append(candidate)
                else:
                    beams.append(candidate)
            if not beams:
                break

        choices = completed + beams
        if not choices:
            return input_ids
        # Cumulative log probability is the standard beam-search objective;
        # avoid a length bias here unless a length penalty is explicitly added.
        return max(choices, key=lambda item: item[1])[0]
