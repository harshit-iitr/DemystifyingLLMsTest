"""Beam search decoding."""
import torch
from typing import List, Tuple, Optional, Union, Iterable
from .stop_criteria import normalize_eos_token_ids


class BeamSearch:
    def __init__(
        self,
        beam_width: int = 4,
        max_length: int = 50,
        eos_token_id: Optional[Union[int, Iterable[int]]] = None,
        early_stopping: bool = True,
    ):
        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {beam_width}")
        if max_length < 0:
            raise ValueError(f"max_length must be non-negative, got {max_length}")
        self.beam_width = beam_width
        self.max_length = max_length
        self.eos_token_ids = normalize_eos_token_ids(eos_token_id)
        self.early_stopping = early_stopping

    def search(self, model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run beam search for a single input sequence.
        """
        batch_size = input_ids.size(0)
        if batch_size != 1:
            raise ValueError("BeamSearch currently supports batch_size=1")
        beams: List[Tuple[torch.Tensor, float]] = [(input_ids, 0.0)]
        completed: List[Tuple[torch.Tensor, float]] = []

        for _ in range(self.max_length):
            new_candidates: List[Tuple[torch.Tensor, float]] = []
            for seq, score in beams:
                if self.eos_token_ids and seq[0, -1].item() in self.eos_token_ids:
                    completed.append((seq, score))
                    continue

                logits = model(seq)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                expansion_width = min(self.beam_width, next_log_probs.size(-1))
                topk_probs, topk_ids = torch.topk(next_log_probs, expansion_width, dim=-1)

                for k in range(expansion_width):
                    cand_id = topk_ids[:, k:k+1]
                    cand_score = score + topk_probs[0, k].item()
                    cand_seq = torch.cat([seq, cand_id], dim=-1)
                    new_candidates.append((cand_seq, cand_score))

            if not new_candidates:
                break

            new_candidates.sort(key=lambda x: x[1] / max(x[0].size(1), 1), reverse=True)
            beams = new_candidates[:self.beam_width]

            if self.early_stopping and len(completed) >= self.beam_width:
                break

        all_candidates = completed + beams
        all_candidates.sort(key=lambda x: x[1] / max(x[0].size(1), 1), reverse=True)
        return all_candidates[0][0]
