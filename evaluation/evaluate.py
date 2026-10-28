"""
Evaluation runner for assessing model perplexity and inference speed.
"""

from typing import List, Dict, Any, Optional
import torch

from .metrics import calculate_perplexity, cross_entropy_loss, token_accuracy, repetition_rate
from .benchmarks import benchmark_forward_latency
from config.constants import PAD_TOKEN_ID


class ModelEvaluator:
    """Evaluates transformer model quality and performance metrics."""

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def evaluate_batch(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute loss, perplexity, and token accuracy on a labeled sequence batch.
        """
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids)
            loss = cross_entropy_loss(logits, target_ids, ignore_index=PAD_TOKEN_ID).item()
            ppl = calculate_perplexity(loss)
            acc = token_accuracy(logits[:, :-1, :], target_ids[:, 1:], ignore_index=PAD_TOKEN_ID)

        return {
            "loss": round(loss, 4),
            "perplexity": round(ppl, 2),
            "token_accuracy": round(acc, 4),
        }

    def evaluate_generation(
        self,
        generated_token_ids: List[int],
        n_gram: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze generated token sequence for quality and diversity.
        """
        rep_rate = repetition_rate(generated_token_ids, n=n_gram)
        return {
            "token_count": len(generated_token_ids),
            f"repetition_rate_{n_gram}gram": round(rep_rate, 4),
            "is_degenerate": rep_rate > 0.5,
        }

    def profile_performance(
        self,
        sample_input: torch.Tensor,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Run forward latency profiling.
        """
        sample_input = sample_input.to(self.device)
        return benchmark_forward_latency(self.model, sample_input, benchmark_runs=num_runs)
