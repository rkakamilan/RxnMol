"""
Metrics collection for CSA optimization runs.

Tracks performance, diversity, and convergence metrics.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
import time


@dataclass
class IterationMetrics:
    """Metrics for a single iteration."""
    iteration: int
    best_score: float
    worst_score: float
    avg_score: float
    d_cut: float
    d_avg: float
    n_accepted: int
    n_evaluated: int
    time_elapsed: float


class MetricsCollector:
    """
    Collects and stores metrics during CSA optimization.

    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.record_iteration(iteration=1, best_score=0.5, ...)
        >>> summary = metrics.get_summary()
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = time.time()
        self.iteration_metrics: List[IterationMetrics] = []
        self.total_evaluations = 0

    def record_iteration(self, **kwargs):
        """Record metrics for an iteration."""
        metrics = IterationMetrics(
            time_elapsed=time.time() - self.start_time,
            **kwargs
        )
        self.iteration_metrics.append(metrics)
        self.total_evaluations += kwargs.get('n_evaluated', 0)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.iteration_metrics:
            return {}

        return {
            'total_iterations': len(self.iteration_metrics),
            'total_evaluations': self.total_evaluations,
            'total_time': time.time() - self.start_time,
            'best_score': min(m.best_score for m in self.iteration_metrics),
            'final_d_cut': self.iteration_metrics[-1].d_cut if self.iteration_metrics else 0,
        }
