"""Lightweight LLM output evaluation framework."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

@dataclass
class EvalResult:
    metric: str
    score: Any
    passed: bool
    prediction: str
    reference: str

class Evalframe:
    """Evaluate LLM outputs with pluggable metrics."""

    def __init__(self):
        self._metrics: Dict[str, Callable[[str, str], Any]] = {}

    def add_metric(self, name: str, fn: Callable[[str, str], Any]) -> None:
        """Register an evaluation metric."""
        self._metrics[name] = fn

    def evaluate(self, prediction: str, reference: str) -> Dict[str, EvalResult]:
        """Evaluate prediction against reference using all metrics."""
        results: Dict[str, EvalResult] = {}
        for name, fn in self._metrics.items():
            try:
                score = fn(prediction, reference)
                passed = bool(score)
            except Exception as e:
                score = None
                passed = False
            results[name] = EvalResult(
                metric=name, score=score, passed=passed,
                prediction=prediction, reference=reference
            )
        return results

    def score(self, prediction: str, reference: str) -> Dict[str, Any]:
        """Return just the scores as a flat dict."""
        return {k: v.score for k, v in self.evaluate(prediction, reference).items()}

    def metrics(self) -> List[str]:
        """Return registered metric names."""
        return list(self._metrics.keys())
