"""Lightweight LLM evaluation framework with built-in metrics and batch support."""

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class EvalResult:
    """Result of running a single metric on one (prediction, reference) pair.

    Attributes:
        metric: Name of the metric that produced this result.
        score: Raw score returned by the metric function, or ``None`` if the
            metric raised an exception during evaluation.
        passed: ``True`` when *score* is truthy (non-zero, non-empty, etc.),
            ``False`` otherwise.
        prediction: The model prediction that was evaluated.
        reference: The ground-truth reference string.
    """

    metric: str
    score: Any
    passed: bool
    prediction: str
    reference: str


# ------------------------------------------------------------------
# Built-in metrics
# ------------------------------------------------------------------

def _exact_match(pred: str, ref: str) -> bool:
    """Return ``True`` when stripped prediction equals stripped reference."""
    return pred.strip() == ref.strip()


def _contains(pred: str, ref: str) -> bool:
    """Return ``True`` when *ref* is a non-empty substring of *pred*.

    An empty reference always returns ``False`` to avoid vacuous matches.
    """
    if not ref:
        return False
    return ref in pred


def _prefix_match(pred: str, ref: str) -> bool:
    """Return ``True`` when *pred* starts with *ref* (both stripped).

    Returns ``False`` when *ref* is empty to avoid vacuous matches.
    """
    ref_stripped = ref.strip()
    if not ref_stripped:
        return False
    return pred.strip().startswith(ref_stripped)


def _f1_score(pred: str, ref: str) -> float:
    """Compute token-level F1 score using the SQuAD convention.

    Tokenises by whitespace and lower-cases both strings.  Token counts are
    handled with a multiset (``Counter``) so repeated tokens are weighted
    correctly.  Precision and recall are computed against the clipped overlap
    (the intersection of the two Counters), and the harmonic mean is returned.

    Returns:
        A float in ``[0, 1]``.  Returns ``0.0`` when either string is empty.
    """
    pred_counts = Counter(pred.lower().split())
    ref_counts = Counter(ref.lower().split())
    if not pred_counts or not ref_counts:
        return 0.0
    common = sum((pred_counts & ref_counts).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_counts.values())
    recall = common / sum(ref_counts.values())
    return round(2 * precision * recall / (precision + recall), 4)


def _rouge1(pred: str, ref: str) -> float:
    """Compute ROUGE-1 unigram recall with clipped token counts.

    Each token in the reference is matched at most as many times as it appears
    in the prediction (clipping), which prevents inflated scores when the
    prediction repeats tokens.  The score is the fraction of reference tokens
    that are covered.

    Returns:
        A float in ``[0, 1]``.  Returns ``0.0`` when the reference is empty.
    """
    pred_counts = Counter(pred.lower().split())
    ref_counts = Counter(ref.lower().split())
    if not ref_counts:
        return 0.0
    common = sum((pred_counts & ref_counts).values())
    return round(common / sum(ref_counts.values()), 4)


BUILTIN_METRICS: Dict[str, Callable[[str, str], Any]] = {
    "exact_match": _exact_match,
    "contains": _contains,
    "prefix_match": _prefix_match,
    "f1": _f1_score,
    "rouge1": _rouge1,
}


class Evalframe:
    """Evaluate LLM outputs with pluggable and built-in metrics.

    Parameters:
        include_builtins: When ``True``, all five built-in metrics
            (``exact_match``, ``contains``, ``prefix_match``, ``f1``,
            ``rouge1``) are registered at construction time.

    Example::

        ef = Evalframe()
        ef.add_builtin("f1")
        ef.add_metric("my_metric", lambda pred, ref: pred == ref)
        results = ef.evaluate("The answer is 42", "42")
    """

    def __init__(self, include_builtins: bool = False) -> None:
        self._metrics: Dict[str, Callable[[str, str], Any]] = {}
        if include_builtins:
            self._metrics.update(BUILTIN_METRICS)

    # ------------------------------------------------------------------
    # Metric registration
    # ------------------------------------------------------------------

    def add_metric(self, name: str, fn: Callable[[str, str], Any]) -> None:
        """Register a custom evaluation metric.

        Args:
            name: Unique identifier for the metric.
            fn: A callable ``(prediction: str, reference: str) -> Any``.
                A truthy return value is treated as a *pass*.
        """
        self._metrics[name] = fn

    def add_builtin(self, name: str) -> None:
        """Add a built-in metric by name.

        Args:
            name: One of ``"exact_match"``, ``"contains"``,
                ``"prefix_match"``, ``"f1"``, or ``"rouge1"``.

        Raises:
            ValueError: If *name* is not a known built-in metric.
        """
        if name not in BUILTIN_METRICS:
            raise ValueError(
                f"Unknown built-in metric: {name!r}. "
                f"Available: {sorted(BUILTIN_METRICS)}"
            )
        self._metrics[name] = BUILTIN_METRICS[name]

    def remove_metric(self, name: str) -> bool:
        """Remove a registered metric.

        Args:
            name: Name of the metric to remove.

        Returns:
            ``True`` if the metric was found and removed, ``False`` otherwise.
        """
        if name in self._metrics:
            del self._metrics[name]
            return True
        return False

    def metrics(self) -> List[str]:
        """Return the names of all currently registered metrics."""
        return list(self._metrics.keys())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, prediction: str, reference: str) -> Dict[str, EvalResult]:
        """Run all registered metrics on one (prediction, reference) pair.

        If a metric function raises an exception, the result for that metric
        will have ``score=None`` and ``passed=False``; a :class:`RuntimeWarning`
        is emitted so callers can detect and diagnose failures without crashing.

        Args:
            prediction: Model output to evaluate.
            reference: Ground-truth reference string.

        Returns:
            A dict mapping metric name to :class:`EvalResult`.
        """
        results: Dict[str, EvalResult] = {}
        for mname, fn in self._metrics.items():
            try:
                score = fn(prediction, reference)
                passed = bool(score)
            except Exception as exc:
                warnings.warn(
                    f"Metric '{mname}' raised {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                score = None
                passed = False
            results[mname] = EvalResult(
                metric=mname,
                score=score,
                passed=passed,
                prediction=prediction,
                reference=reference,
            )
        return results

    def score(self, prediction: str, reference: str) -> Dict[str, Any]:
        """Return just the raw scores as a flat dict.

        Args:
            prediction: Model output to evaluate.
            reference: Ground-truth reference string.

        Returns:
            A dict mapping metric name to its raw score value.
        """
        return {k: v.score for k, v in self.evaluate(prediction, reference).items()}

    def batch_evaluate(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Dict[str, EvalResult]]:
        """Evaluate a list of (prediction, reference) pairs.

        Args:
            pairs: A sequence of ``(prediction, reference)`` tuples.

        Returns:
            A list of result dicts, one per pair, in the same order as
            *pairs*.  Each dict maps metric name to :class:`EvalResult`.
        """
        return [self.evaluate(pred, ref) for pred, ref in pairs]

    def summary(self, results: List[Dict[str, EvalResult]]) -> Dict[str, Any]:
        """Aggregate pass rates and average scores across a batch.

        Pairs where a metric raised an exception (``score is None``) are
        excluded from the average score computation but are still counted in
        ``n``.

        Args:
            results: Output of :meth:`batch_evaluate`.

        Returns:
            A dict mapping metric name to a stats dict with keys:

            * ``pass_rate`` (float): fraction of pairs where the metric passed.
            * ``avg_score`` (float | None): mean score over non-``None``
              results; ``None`` if every result errored.
            * ``n`` (int): total number of pairs evaluated with this metric.
        """
        if not results:
            return {}
        out: Dict[str, Any] = {}
        for mname in self._metrics:
            pairs_with_metric = [r for r in results if mname in r]
            scores = [
                r[mname].score
                for r in pairs_with_metric
                if r[mname].score is not None
            ]
            passes = [r[mname].passed for r in pairs_with_metric]
            out[mname] = {
                "pass_rate": round(sum(passes) / len(passes), 4) if passes else 0.0,
                "avg_score": (
                    round(sum(float(s) for s in scores) / len(scores), 4)
                    if scores
                    else None
                ),
                "n": len(passes),
            }
        return out

    def assert_passes(
        self,
        prediction: str,
        reference: str,
        min_pass_rate: float = 1.0,
    ) -> bool:
        """Return ``True`` when the fraction of passing metrics >= *min_pass_rate*.

        Args:
            prediction: Model output to evaluate.
            reference: Ground-truth reference string.
            min_pass_rate: Minimum required fraction of metrics that must pass
                (default ``1.0``, i.e. all metrics must pass).

        Returns:
            ``True`` if the pass fraction meets or exceeds *min_pass_rate*.

        Raises:
            ValueError: If no metrics are registered, as the result would be
                ambiguous.
        """
        results = self.evaluate(prediction, reference)
        if not results:
            raise ValueError(
                "assert_passes() called with no metrics registered. "
                "Add at least one metric before calling assert_passes()."
            )
        passing = sum(1 for r in results.values() if r.passed)
        return (passing / len(results)) >= min_pass_rate
