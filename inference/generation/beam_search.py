"""Beam search (Clean - Red Herring)."""
import torch
from typing import List, Tuple, Optional


class BeamSearch:
    def __init__(self, beam_width: int = 4, max_length: int = 50, eos_token_id: Optional[int] = None):
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        self.beam_width = beam_width
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def search(self, model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Standard beam search implementation (red herring).
        """
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError("BeamSearch.search currently supports input_ids with batch size 1")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids must contain at least one token")
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
                k = min(self.beam_width, next_log_probs.size(-1))
                topk_probs, topk_ids = torch.topk(next_log_probs, k, dim=-1)

                for candidate_idx in range(k):
                    cand_id = topk_ids[:, candidate_idx:candidate_idx + 1]
                    cand_score = score + topk_probs[0, candidate_idx].item()
                    cand_seq = torch.cat([seq, cand_id], dim=-1)
                    new_candidates.append((cand_seq, cand_score))

            if not new_candidates:
                break

            new_candidates.sort(key=lambda x: x[1] / max(x[0].size(1), 1), reverse=True)
            beams = new_candidates[:self.beam_width]

            if len(completed) >= self.beam_width:
                break

        all_candidates = completed + beams
        all_candidates.sort(key=lambda x: x[1] / max(x[0].size(1), 1), reverse=True)
        return all_candidates[0][0]
