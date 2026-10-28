"""Beam search (Clean - Red Herring)."""
import torch
from typing import List, Tuple, Optional


class BeamSearch:
    def __init__(self, beam_width: int = 4, max_length: int = 50, eos_token_id: Optional[int] = None, stop_token_ids=None):
        if beam_width < 1 or max_length < 0:
            raise ValueError("beam_width must be positive and max_length non-negative")
        self.beam_width = beam_width
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.stop_token_ids = set(stop_token_ids or ())
        if eos_token_id is not None:
            self.stop_token_ids.add(eos_token_id)

    @torch.no_grad()
    def search(self, model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Standard beam search implementation (red herring).
        """
        if input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.size(1) == 0:
            raise ValueError("Beam search requires one nonempty prompt")
        model.eval()
        prompt_length = input_ids.size(1)
        # Supports batch_size = 1
        beams: List[Tuple[torch.Tensor, float]] = [(input_ids, 0.0)]
        completed: List[Tuple[torch.Tensor, float]] = []

        limit = self.max_length
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "max_seq_len"):
            if prompt_length > config.max_seq_len:
                raise ValueError("Prompt exceeds the model context window")
            limit = min(limit, config.max_seq_len - prompt_length)
        for _ in range(limit):
            new_candidates: List[Tuple[torch.Tensor, float]] = []
            for seq, score in beams:
                if seq.size(1) > prompt_length and seq[0, -1].item() in self.stop_token_ids:
                    completed.append((seq, score))
                    continue

                logits = model(seq)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                topk_probs, topk_ids = torch.topk(next_log_probs, min(self.beam_width, next_log_probs.size(-1)), dim=-1)

                for k in range(topk_ids.size(-1)):
                    cand_id = topk_ids[:, k:k+1]
                    cand_score = score + topk_probs[0, k].item()
                    cand_seq = torch.cat([seq, cand_id], dim=-1)
                    new_candidates.append((cand_seq, cand_score))

            if not new_candidates:
                beams = []
                break

            new_candidates.sort(key=lambda x: x[1] / max(x[0].size(1), 1), reverse=True)
            beams = new_candidates[:self.beam_width]

            if len(completed) >= self.beam_width:
                break

        all_candidates = completed + beams
        all_candidates.sort(key=lambda x: x[1] / max(x[0].size(1), 1), reverse=True)
        return all_candidates[0][0]
