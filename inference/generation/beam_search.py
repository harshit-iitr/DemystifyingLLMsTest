"""Beam search (Clean - Red Herring)."""
import torch
from typing import List, Tuple, Optional


class BeamSearch:
    def __init__(self, beam_width: int = 4, max_length: int = 50, eos_token_id: Optional[int] = None):
        self.beam_width = beam_width
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def search(self, model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Standard beam search implementation (red herring).
        """
        if input_ids.size(0) != 1:
            raise ValueError("BeamSearch currently supports batch_size=1")
        if self.beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        if self.max_length < 0:
            raise ValueError("max_length must be non-negative")
        beams: List[Tuple[torch.Tensor, float]] = [(input_ids, 0.0)]
        completed: List[Tuple[torch.Tensor, float]] = []

        for _ in range(self.max_length):
            new_candidates: List[Tuple[torch.Tensor, float]] = []
            for seq, score in beams:
                if self.eos_token_id is not None and seq[0, -1].item() == self.eos_token_id:
                    completed.append((seq, score))
                    continue

                logits = model(seq)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                count = min(self.beam_width, next_log_probs.size(-1))
                topk_probs, topk_ids = torch.topk(next_log_probs, count, dim=-1)

                for k in range(count):
                    cand_id = topk_ids[:, k:k+1]
                    cand_score = score + topk_probs[0, k].item()
                    cand_seq = torch.cat([seq, cand_id], dim=-1)
                    new_candidates.append((cand_seq, cand_score))

            if not new_candidates:
                break

            new_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = new_candidates[:self.beam_width]

        all_candidates = completed + beams
        if not all_candidates:
            return input_ids
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        return all_candidates[0][0]
