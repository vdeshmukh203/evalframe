"""Lightweight LLM evaluation framework with built-in metrics and batch support."""
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings


@dataclass
class EvalResult:
    """Result from running a single metric on a (prediction, reference) pair.

    Attributes
    ----------
    metric:
        Name of the metric that produced this result.
    score:
        Raw metric output — a ``bool`` for discrete metrics or a ``float`` in
        [0, 1] for continuous ones.
    passed:
        ``True`` when ``bool(score)`` is truthy (i.e. score > 0 for floats,
        ``True`` for booleans).
    prediction:
        The model output that was evaluated.
    reference:
        The ground-truth string used as the target.
    """

    metric: str
    score: Any
    passed: bool
    prediction: str
    reference: str


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------


def _exact_match(pred: str, ref: str) -> bool:
    """Return ``True`` if prediction equals reference after stripping whitespace."""
    return pred.strip() == ref.strip()


def _contains(pred: str, ref: str) -> bool:
    """Return ``True`` if reference is a substring of prediction."""
    return ref in pred


def _prefix_match(pred: str, ref: str) -> bool:
    """Return ``True`` if prediction starts with reference (after stripping)."""
    return pred.strip().startswith(ref.strip())


def _f1_score(pred: str, ref: str) -> float:
    """Token-level F1 using multiset (Counter) intersection.

    Implements the bag-of-words F1 from the SQuAD evaluation script:
    precision = |common| / |pred tokens|,
    recall    = |common| / |ref tokens|,
    F1        = harmonic mean of precision and recall,
    where |common| counts each token up to the minimum frequency in both strings.
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
    """ROUGE-1 recall using multiset (Counter) intersection.

    Recall = |common| / |ref tokens|, where |common| counts each unigram
    up to the minimum frequency appearing in both strings.  This prevents
    inflated scores when a token is repeated in the reference but rare in
    the prediction.
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
        Pre-load all five built-in metrics (``exact_match``, ``contains``,
        ``prefix_match``, ``f1``, ``rouge1``) on construction.

    Examples
    --------
    >>> ef = Evalframe()
    >>> ef.add_builtin("exact_match")
    >>> ef.add_builtin("f1")
    >>> result = ef.evaluate("the cat sat", "the cat")
    >>> result["exact_match"].passed
    False
    >>> result["f1"].score
    0.8
    """

    def __init__(self, include_builtins: bool = False) -> None:
        self._metrics: Dict[str, Callable[[str, str], Any]] = {}
        if include_builtins:
            self._metrics.update(BUILTIN_METRICS)

    # -----------------------------------------------------------------------
    # Metric registration
    # -----------------------------------------------------------------------

    def add_metric(self, name: str, fn: Callable[[str, str], Any]) -> None:
        """Register a custom evaluation metric.

        Parameters
        ----------
        name:
            Unique identifier for the metric.  Registering a name that already
            exists silently overwrites the previous function.
        fn:
            Callable ``(prediction: str, reference: str) -> score``.  The score
            may be a ``bool`` or a ``float`` in [0, 1].

        Raises
        ------
        TypeError
            If *fn* is not callable.
        """
        if not callable(fn):
            raise TypeError(
                f"Metric function must be callable, got {type(fn).__name__!r}"
            )
        self._metrics[name] = fn

    def add_builtin(self, name: str) -> None:
        """Add a built-in metric by name.

        Available names: ``exact_match``, ``contains``, ``prefix_match``,
        ``f1``, ``rouge1``.

        Raises
        ------
        ValueError
            If *name* is not a recognised built-in metric.
        """
        if name not in BUILTIN_METRICS:
            raise ValueError(
                f"Unknown built-in metric: {name!r}. "
                f"Available: {sorted(BUILTIN_METRICS)}"
            )
        self._metrics[name] = BUILTIN_METRICS[name]

    def remove_metric(self, name: str) -> bool:
        """Remove a registered metric by name.

        Returns
        -------
        bool
            ``True`` if the metric existed and was removed, ``False`` if no
            metric with that name was registered.
        """
        if name in self._metrics:
            del self._metrics[name]
            return True
        return False

    def metrics(self) -> List[str]:
        """Return the names of all currently registered metrics."""
        return list(self._metrics.keys())

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def evaluate(self, prediction: str, reference: str) -> Dict[str, "EvalResult"]:
        """Run all registered metrics on one (prediction, reference) pair.

        Parameters
        ----------
        prediction:
            The model output to evaluate.
        reference:
            The ground-truth string.

        Returns
        -------
        Dict[str, EvalResult]
            Mapping of metric name → :class:`EvalResult`.  If a metric raises
            an exception a :class:`RuntimeWarning` is emitted and that metric's
            result will have ``score=None`` and ``passed=False``.
        """
        results: Dict[str, EvalResult] = {}
        for mname, fn in self._metrics.items():
            try:
                score = fn(prediction, reference)
                passed = bool(score)
            except Exception as exc:
                warnings.warn(
                    f"Metric {mname!r} raised an exception and will be skipped: {exc}",
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

        Parameters
        ----------
        prediction:
            The model output to evaluate.
        reference:
            The ground-truth string.

        Returns
        -------
        Dict[str, Any]
            Mapping of metric name → raw score (``bool`` or ``float``).
        """
        return {k: v.score for k, v in self.evaluate(prediction, reference).items()}

    def batch_evaluate(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Dict[str, "EvalResult"]]:
        """Evaluate a list of (prediction, reference) pairs.

        Parameters
        ----------
        pairs:
            Sequence of ``(prediction, reference)`` tuples.

        Returns
        -------
        List[Dict[str, EvalResult]]
            One result dict per pair, in the same order as *pairs*.
        """
        return [self.evaluate(pred, ref) for pred, ref in pairs]

    def summary(self, results: List[Dict[str, "EvalResult"]]) -> Dict[str, Any]:
        """Aggregate pass rates and average scores across a batch.

        Parameters
        ----------
        results:
            Output of :meth:`batch_evaluate`.

        Returns
        -------
        Dict[str, Any]
            Per-metric dict with keys:

            ``pass_rate``
                Fraction of pairs where the metric passed (0.0–1.0).
            ``avg_score``
                Mean of numeric scores, or ``None`` when all scores are
                non-numeric (e.g. pure boolean metrics).
            ``n``
                Number of pairs evaluated by this metric.

            Returns an empty dict when *results* is empty.
        """
        if not results:
            return {}
        out: Dict[str, Any] = {}
        for mname in self._metrics:
            passes = [r[mname].passed for r in results if mname in r]
            numeric_scores: List[float] = []
            for r in results:
                if mname not in r or r[mname].score is None:
                    continue
                try:
                    numeric_scores.append(float(r[mname].score))
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
        """Return ``True`` if the fraction of passing metrics >= *min_pass_rate*.

        Parameters
        ----------
        prediction:
            The model output to evaluate.
        reference:
            The ground-truth string.
        min_pass_rate:
            Minimum fraction of metrics that must pass (0.0–1.0).  Defaults to
            ``1.0`` (all metrics must pass).

        Returns
        -------
        bool
            ``True`` when the pass rate meets or exceeds *min_pass_rate*.
            Returns ``True`` vacuously when no metrics are registered.
        """
        results = self.evaluate(prediction, reference)
        if not results:
            return True
        passing = sum(1 for r in results.values() if r.passed)
        return (passing / len(results)) >= min_pass_rate
