"""Beam search (Clean - Red Herring)."""
import torch
from typing import List, Tuple, Optional


class BeamSearch:
    def __init__(
        self,
        beam_width: int = 4,
        max_length: int = 50,
        eos_token_id: Optional[int] = None,
        eos_token_ids: Optional[List[int]] = None,
        max_context_length: Optional[int] = None,
    ):
        if beam_width < 1:
            raise ValueError(f"beam_width must be positive, got {beam_width}")
        if max_length < 0:
            raise ValueError(f"max_length must be non-negative, got {max_length}")
        if max_context_length is not None and max_context_length <= 0:
            raise ValueError(f"max_context_length must be positive, got {max_context_length}")
        self.beam_width = beam_width
        self.max_length = max_length
        self.eos_token_ids = set(eos_token_ids or [])
        if eos_token_id is not None:
            self.eos_token_ids.add(eos_token_id)
        self.max_context_length = max_context_length

    def search(self, model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Standard beam search implementation (red herring).
        """
        batch_size = input_ids.size(0)
        if batch_size != 1:
            raise ValueError("BeamSearch currently supports batch_size=1")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids must contain at least one prompt token")
        if self.eos_token_ids and input_ids[0, -1].item() in self.eos_token_ids:
            return input_ids
        beams: List[Tuple[torch.Tensor, float]] = [(input_ids, 0.0)]
        completed: List[Tuple[torch.Tensor, float]] = []

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for _ in range(self.max_length):
                    new_candidates: List[Tuple[torch.Tensor, float]] = []
                    for seq, score in beams:
                        model_input = (
                            seq[:, -self.max_context_length:]
                            if self.max_context_length is not None else seq
                        )
                        logits = model(model_input)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        next_log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                        top_k = min(self.beam_width, next_log_probs.size(-1))
                        topk_probs, topk_ids = torch.topk(next_log_probs, top_k, dim=-1)

                        for k in range(top_k):
                            cand_id = topk_ids[:, k:k + 1]
                            cand_score = score + topk_probs[0, k].item()
                            cand_seq = torch.cat([seq, cand_id], dim=-1)
                            candidate = (cand_seq, cand_score)
                            if cand_id.item() in self.eos_token_ids:
                                completed.append(candidate)
                            else:
                                new_candidates.append(candidate)

                    if not new_candidates:
                        beams = []
                        break

                    new_candidates.sort(key=lambda x: x[1] / max(x[0].size(1) - input_ids.size(1), 1), reverse=True)
                    beams = new_candidates[:self.beam_width]

                    if len(completed) >= self.beam_width:
                        break
        finally:
            model.train(was_training)

        all_candidates = completed + beams
        all_candidates.sort(key=lambda x: x[1] / max(x[0].size(1) - input_ids.size(1), 1), reverse=True)
        return all_candidates[0][0]
