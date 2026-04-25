"""Lightweight LLM evaluation framework with built-in metrics and batch support."""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class EvalResult:
    metric: str
    score: Any
    passed: bool
    prediction: str
    reference: str


# ------------------------------------------------------------------
# Built-in metrics
# ------------------------------------------------------------------

def _exact_match(pred: str, ref: str) -> bool:
    return pred.strip() == ref.strip()


def _contains(pred: str, ref: str) -> bool:
    return ref in pred


def _prefix_match(pred: str, ref: str) -> bool:
    return pred.strip().startswith(ref.strip())


def _f1_score(pred: str, ref: str) -> float:
    """Token-level F1 (word overlap)."""
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def _rouge1(pred: str, ref: str) -> float:
    """ROUGE-1 recall (unigram overlap)."""
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not ref_tokens:
        return 0.0
    common = sum(1 for t in ref_tokens if t in pred_tokens)
    return round(common / len(ref_tokens), 4)


BUILTIN_METRICS: Dict[str, Callable[[str, str], Any]] = {
    "exact_match": _exact_match,
    "contains": _contains,
    "prefix_match": _prefix_match,
    "f1": _f1_score,
    "rouge1": _rouge1,
}


class Evalframe:
    """Evaluate LLM outputs with pluggable and built-in metrics."""

    def __init__(self, include_builtins: bool = False):
        self._metrics: Dict[str, Callable[[str, str], Any]] = {}
        if include_builtins:
            self._metrics.update(BUILTIN_METRICS)

    # ------------------------------------------------------------------
    # Metric registration
    # ------------------------------------------------------------------

    def add_metric(self, name: str, fn: Callable[[str, str], Any]) -> None:
        """Register a custom evaluation metric."""
        self._metrics[name] = fn

    def add_builtin(self, name: str) -> None:
        """Add a built-in metric by name (exact_match, contains, prefix_match, f1, rouge1)."""
        if name not in BUILTIN_METRICS:
            raise ValueError(f"Unknown built-in metric: {name!r}. Available: {list(BUILTIN_METRICS)}")
        self._metrics[name] = BUILTIN_METRICS[name]

    def remove_metric(self, name: str) -> bool:
        if name in self._metrics:
            del self._metrics[name]
            return True
        return False

    def metrics(self) -> List[str]:
        return list(self._metrics.keys())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, prediction: str, reference: str) -> Dict[str, EvalResult]:
        """Run all metrics on one (prediction, reference) pair."""
        results: Dict[str, EvalResult] = {}
        for mname, fn in self._metrics.items():
            try:
                score = fn(prediction, reference)
                passed = bool(score)
            except Exception:
                score = None
                passed = False
            results[mname] = EvalResult(
                metric=mname, score=score, passed=passed,
                prediction=prediction, reference=reference,
            )
        return results

    def score(self, prediction: str, reference: str) -> Dict[str, Any]:
        """Return just the scores as a flat dict."""
        return {k: v.score for k, v in self.evaluate(prediction, reference).items()}

    def batch_evaluate(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, EvalResult]]:
        """Evaluate a list of (prediction, reference) pairs."""
        return [self.evaluate(pred, ref) for pred, ref in pairs]

    def summary(self, results: List[Dict[str, EvalResult]]) -> Dict[str, Any]:
        """Aggregate pass rates and average scores across a batch."""
        if not results:
            return {}
        out: Dict[str, Any] = {}
        for mname in self._metrics:
            scores = [r[mname].score for r in results if mname in r and r[mname].score is not None]
            passes = [r[mname].passed for r in results if mname in r]
            out[mname] = {
                "pass_rate": round(sum(passes) / len(passes), 4) if passes else 0.0,
                "avg_score": round(sum(float(s) for s in scores) / len(scores), 4) if scores else None,
                "n": len(passes),
            }
        return out

    def assert_passes(self, prediction: str, reference: str,
                      min_pass_rate: float = 1.0) -> bool:
        """Return True if the fraction of passing metrics >= min_pass_rate."""
        results = self.evaluate(prediction, reference)
        if not results:
            return True
        passing = sum(1 for r in results.values() if r.passed)
        return (passing / len(results)) >= min_pass_rate
