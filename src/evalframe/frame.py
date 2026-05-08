"""Lightweight LLM evaluation framework with built-in metrics and batch support."""
from __future__ import annotations

import logging
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result for a single (prediction, reference) pair evaluated by one metric.

    Attributes
    ----------
    metric:
        Registered name of the metric.
    score:
        Value returned by the metric function, or ``None`` on error.
    passed:
        ``True`` when *score* is truthy.
    prediction:
        The model output string.
    reference:
        The ground-truth string.
    error:
        Exception message when the metric function raised; ``None`` otherwise.
    """

    metric: str
    score: Union[float, bool, int, None]
    passed: bool
    prediction: str
    reference: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Built-in metric implementations
# ---------------------------------------------------------------------------

def _exact_match(pred: str, ref: str) -> bool:
    """Return True when prediction and reference are identical after stripping whitespace."""
    return pred.strip() == ref.strip()


def _contains(pred: str, ref: str) -> bool:
    """Return True when the reference appears as a substring of the prediction."""
    return ref in pred


def _prefix_match(pred: str, ref: str) -> bool:
    """Return True when the stripped prediction starts with the stripped reference."""
    return pred.strip().startswith(ref.strip())


def _f1_score(pred: str, ref: str) -> float:
    """Token-level F1 score using Counter-based overlap (handles repeated tokens).

    Follows the SQuAD evaluation convention: precision is the fraction of
    prediction tokens that overlap with reference tokens; recall is the
    fraction of reference tokens that overlap with prediction tokens.
    ``collections.Counter`` intersection correctly clips repeated tokens.
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
    """ROUGE-1 recall: fraction of reference unigrams present in the prediction.

    ``collections.Counter`` intersection clips repeated tokens so that a token
    appearing *k* times in the reference is only credited up to its count in
    the prediction (standard ROUGE clipping).
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

    Parameters
    ----------
    include_builtins:
        When ``True``, all built-in metrics (``exact_match``, ``contains``,
        ``prefix_match``, ``f1``, ``rouge1``) are registered at construction.

    Examples
    --------
    >>> ef = Evalframe()
    >>> ef.add_builtin("exact_match")
    >>> ef.score("Paris", "Paris")
    {'exact_match': True}

    >>> ef = Evalframe(include_builtins=True)
    >>> ef.assert_passes("the cat sat on the mat", "cat sat", min_pass_rate=0.5)
    True
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

        Parameters
        ----------
        name:
            Unique identifier for the metric.
        fn:
            A callable ``(prediction: str, reference: str) -> Any`` whose
            return value is truthy when the prediction passes.

        Raises
        ------
        ValueError
            If *name* is not a non-empty string.
        TypeError
            If *fn* is not callable.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Metric name must be a non-empty string.")
        if not callable(fn):
            raise TypeError(
                f"Expected a callable for metric {name!r}, got {type(fn).__name__}."
            )
        self._metrics[name] = fn

    def add_builtin(self, name: str) -> None:
        """Register one of the built-in metrics by name.

        Available names: ``exact_match``, ``contains``, ``prefix_match``,
        ``f1``, ``rouge1``.

        Raises
        ------
        ValueError
            If *name* does not correspond to a known built-in metric.
        """
        if name not in BUILTIN_METRICS:
            raise ValueError(
                f"Unknown built-in metric: {name!r}. "
                f"Available: {sorted(BUILTIN_METRICS)}"
            )
        self._metrics[name] = BUILTIN_METRICS[name]

    def remove_metric(self, name: str) -> bool:
        """Remove a registered metric.

        Returns
        -------
        bool
            ``True`` if the metric existed and was removed, ``False`` otherwise.
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
        """Run all registered metrics on a single (prediction, reference) pair.

        Parameters
        ----------
        prediction:
            The model output to evaluate.
        reference:
            The ground-truth string to compare against.

        Returns
        -------
        dict[str, EvalResult]
            Mapping from metric name to its :class:`EvalResult`.

        Raises
        ------
        TypeError
            If *prediction* or *reference* is not a string.
        """
        if not isinstance(prediction, str):
            raise TypeError(
                f"'prediction' must be a str, got {type(prediction).__name__}."
            )
        if not isinstance(reference, str):
            raise TypeError(
                f"'reference' must be a str, got {type(reference).__name__}."
            )
        results: Dict[str, EvalResult] = {}
        for mname, fn in self._metrics.items():
            try:
                score = fn(prediction, reference)
                passed = bool(score)
                error = None
            except Exception as exc:
                logger.warning(
                    "Metric %r raised an exception: %s", mname, exc, exc_info=True
                )
                score = None
                passed = False
                error = str(exc)
            results[mname] = EvalResult(
                metric=mname,
                score=score,
                passed=passed,
                prediction=prediction,
                reference=reference,
                error=error,
            )
        return results

    def score(self, prediction: str, reference: str) -> Dict[str, Any]:
        """Return just the raw scores as a flat mapping.

        Parameters
        ----------
        prediction:
            The model output to evaluate.
        reference:
            The ground-truth string.

        Returns
        -------
        dict[str, Any]
            Metric name → raw score value.
        """
        return {k: v.score for k, v in self.evaluate(prediction, reference).items()}

    def batch_evaluate(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Dict[str, EvalResult]]:
        """Evaluate a list of (prediction, reference) pairs.

        Parameters
        ----------
        pairs:
            Sequence of ``(prediction, reference)`` tuples.

        Returns
        -------
        list[dict[str, EvalResult]]
            One result dict per input pair, in the same order.
        """
        return [self.evaluate(pred, ref) for pred, ref in pairs]

    def summary(self, results: List[Dict[str, EvalResult]]) -> Dict[str, Any]:
        """Aggregate pass rates and average scores across a batch.

        Parameters
        ----------
        results:
            Output of :meth:`batch_evaluate`.

        Returns
        -------
        dict[str, dict]
            Per-metric statistics: ``pass_rate`` (float), ``avg_score``
            (float or ``None``), and ``n`` (int).
        """
        if not results:
            return {}
        out: Dict[str, Any] = {}
        for mname in self._metrics:
            scores = [
                r[mname].score
                for r in results
                if mname in r and r[mname].score is not None
            ]
            passes = [r[mname].passed for r in results if mname in r]
            numeric_scores: List[float] = []
            for s in scores:
                try:
                    numeric_scores.append(float(s))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    pass
            out[mname] = {
                "pass_rate": round(sum(passes) / len(passes), 4) if passes else 0.0,
                "avg_score": (
                    round(sum(numeric_scores) / len(numeric_scores), 4)
                    if numeric_scores
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
        """Return ``True`` if the fraction of passing metrics meets *min_pass_rate*.

        Parameters
        ----------
        prediction:
            The model output to evaluate.
        reference:
            The ground-truth string.
        min_pass_rate:
            Minimum fraction of metrics that must pass (default ``1.0``).

        Returns
        -------
        bool
            ``True`` when at least *min_pass_rate* of registered metrics pass.

        Warns
        -----
        UserWarning
            Issued when no metrics are registered (returns ``True`` vacuously).
        """
        results = self.evaluate(prediction, reference)
        if not results:
            warnings.warn(
                "assert_passes called with no metrics registered; returning True vacuously.",
                UserWarning,
                stacklevel=2,
            )
            return True
        passing = sum(1 for r in results.values() if r.passed)
        return (passing / len(results)) >= min_pass_rate
