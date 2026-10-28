"""
Performance Profiling and Latency Benchmarking Utilities.

Status: Clean — Red Herring (Intricate multi-stage latency tracking with CUDA events).
"""

import time
import functools
from typing import Dict, Any, List, Optional, Callable
import torch

from .logging_utils import get_logger

logger = get_logger("profiling")


class Timer:
    """Precise context manager for tracking wall-clock and GPU kernel execution times."""

    def __init__(self, name: str = "Block", use_cuda_events: bool = True):
        self.name = name
        self.use_cuda_events = use_cuda_events and torch.cuda.is_available()
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0
        self._start_event: Optional[Any] = None
        self._end_event: Optional[Any] = None

    def __enter__(self) -> "Timer":
        if self.use_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        wall_time = (time.perf_counter() - self.start_time) * 1000.0
        if self.use_cuda_events and self._start_event and self._end_event:
            self._end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self.elapsed_ms = wall_time
        logger.debug(f"[Timer] {self.name}: {self.elapsed_ms:.3f} ms")


class LatencyProfiler:
    """Tracks latency metrics across generation steps: TTFT, inter-token latency, and throughput."""

    def __init__(self):
        self.token_times: List[float] = []
        self.start_time: Optional[float] = None
        self.ttft: Optional[float] = None  # Time to first token (seconds)

    def start(self) -> None:
        """Mark the onset of the prompt processing / prefill phase."""
        self.start_time = time.perf_counter()
        self.token_times.clear()
        self.ttft = None

    def record_token(self) -> None:
        """Record generation of a single autoregressive token."""
        now = time.perf_counter()
        if self.start_time is None:
            self.start_time = now

        if self.ttft is None:
            self.ttft = now - self.start_time

        self.token_times.append(now)

    def summary(self) -> Dict[str, float]:
        """Compute aggregated latency and throughput metrics."""
        if not self.token_times or self.start_time is None:
            return {"total_time_s": 0.0, "total_tokens": 0, "tokens_per_sec": 0.0, "ttft_s": 0.0}

        total_time = self.token_times[-1] - self.start_time
        num_tokens = len(self.token_times)
        inter_token_latencies = [
            (t2 - t1) * 1000.0 for t1, t2 in zip(self.token_times[:-1], self.token_times[1:])
        ]
        avg_itl_ms = sum(inter_token_latencies) / len(inter_token_latencies) if inter_token_latencies else 0.0

        return {
            "total_time_s": total_time,
            "total_tokens": num_tokens,
            "tokens_per_sec": num_tokens / total_time if total_time > 0 else 0.0,
            "ttft_s": self.ttft or 0.0,
            "avg_inter_token_latency_ms": avg_itl_ms,
        }


def profile_call(func: Callable) -> Callable:
    """Decorator to measure and log function execution duration."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[Profile] {func.__qualname__} finished in {duration_ms:.2f} ms")
        return result
    return wrapper


class MemoryTracker:
    """Tracks CPU and GPU memory allocation delta during a code block."""

    def __init__(self, tag: str = "Memory"):
        self.tag = tag
        self.initial_allocated: int = 0
        self.peak_allocated: int = 0

    def __enter__(self) -> "MemoryTracker":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.initial_allocated = torch.cuda.memory_allocated()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if torch.cuda.is_available():
            self.peak_allocated = torch.cuda.max_memory_allocated()
            delta_mb = (self.peak_allocated - self.initial_allocated) / (1024 * 1024)
            logger.debug(f"[Memory] {self.tag}: Peak delta = {delta_mb:.2f} MB")
