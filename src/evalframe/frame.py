"""Lightweight LLM evaluation framework with built-in metrics and batch support."""
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class EvalResult:
    """Result of running a single metric on one (prediction, reference) pair.

    Attributes
    ----------
    metric : str
        Name of the metric that produced this result.
    score : Any
        Raw value returned by the metric function (bool, float, or None on
        error).
    passed : bool
        Whether the metric considers the prediction acceptable.  For numeric
        scores this is ``bool(score)``; any non-zero value counts as passing.
    prediction : str
        The model output that was evaluated.
    reference : str
        The ground-truth string used for comparison.
    """

    metric: str
    score: Any
    passed: bool
    prediction: str
    reference: str

    def __repr__(self) -> str:
        return (
            f"EvalResult(metric={self.metric!r}, score={self.score!r}, "
            f"passed={self.passed!r})"
        )


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------

def _exact_match(pred: str, ref: str) -> bool:
    """Return True when stripped strings are identical."""
    return pred.strip() == ref.strip()


def _contains(pred: str, ref: str) -> bool:
    """Return True when *ref* appears as a substring of *pred*."""
    return ref in pred


def _prefix_match(pred: str, ref: str) -> bool:
    """Return True when *pred* (stripped) starts with *ref* (stripped)."""
    return pred.strip().startswith(ref.strip())


def _f1_score(pred: str, ref: str) -> float:
    """Token-level F1 using multi-set (Counter) token overlap.

    Follows the SQuAD evaluation convention: duplicate tokens contribute
    independently to precision and recall, so ``Counter`` intersection is used
    instead of set intersection.

    Returns 0.0 when either string is empty or there is no token overlap.
    """
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def _rouge1(pred: str, ref: str) -> float:
    """ROUGE-1 recall using multi-set unigram overlap.

    Uses ``Counter`` intersection so that a token appearing *k* times in the
    reference can only be credited at most *k* times across both the prediction
    and the reference (multi-set semantics).

    Returns 0.0 when the reference is empty.
    """
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    return round(num_common / len(ref_tokens), 4)


#: Mapping of built-in metric names to their implementations.
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
    include_builtins : bool, optional
        When ``True`` all built-in metrics (``exact_match``, ``contains``,
        ``prefix_match``, ``f1``, ``rouge1``) are registered at construction
        time.  Default is ``False``.

    Examples
    --------
    >>> ev = Evalframe()
    >>> ev.add_builtin("exact_match")
    >>> ev.score("hello", "hello")
    {'exact_match': True}

    >>> ev2 = Evalframe(include_builtins=True)
    >>> ev2.metrics()
    ['exact_match', 'contains', 'prefix_match', 'f1', 'rouge1']
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
        name : str
            Unique identifier for the metric.  Must be a non-empty string.
        fn : Callable[[str, str], Any]
            A callable that accepts ``(prediction, reference)`` strings and
            returns a numeric score or boolean.

        Raises
        ------
        TypeError
            If *name* is not a ``str`` or *fn* is not callable.
        ValueError
            If *name* is an empty string.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"Metric name must be a str, got {type(name).__name__!r}"
            )
        if not name:
            raise ValueError("Metric name must not be empty.")
        if not callable(fn):
            raise TypeError(
                f"Metric function must be callable, got {type(fn).__name__!r}"
            )
        self._metrics[name] = fn

    def add_builtin(self, name: str) -> None:
        """Add a built-in metric by name.

        Parameters
        ----------
        name : str
            One of: ``exact_match``, ``contains``, ``prefix_match``,
            ``f1``, ``rouge1``.

        Raises
        ------
        ValueError
            If *name* does not correspond to a known built-in metric.
        """
        if name not in BUILTIN_METRICS:
            raise ValueError(
                f"Unknown built-in metric: {name!r}. "
                f"Available: {list(BUILTIN_METRICS)}"
            )
        self._metrics[name] = BUILTIN_METRICS[name]

    def remove_metric(self, name: str) -> bool:
        """Remove a registered metric.

        Parameters
        ----------
        name : str
            Name of the metric to remove.

        Returns
        -------
        bool
            ``True`` if the metric existed and was removed, ``False``
            otherwise.
        """
        if name in self._metrics:
            del self._metrics[name]
            return True
        return False

    def metrics(self) -> List[str]:
        """Return the names of all currently registered metrics.

        Returns
        -------
        List[str]
            Metric names in registration order.
        """
        return list(self._metrics.keys())

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def evaluate(self, prediction: str, reference: str) -> Dict[str, EvalResult]:
        """Run all registered metrics on one (prediction, reference) pair.

        If a metric raises an exception a :class:`RuntimeWarning` is issued and
        the corresponding result has ``score=None`` and ``passed=False``.

        Parameters
        ----------
        prediction : str
            The model output to evaluate.
        reference : str
            The ground-truth string.

        Returns
        -------
        Dict[str, EvalResult]
            Mapping of metric name to its :class:`EvalResult`.
        """
        results: Dict[str, EvalResult] = {}
        for mname, fn in self._metrics.items():
            try:
                score = fn(prediction, reference)
                passed = bool(score)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Metric {mname!r} raised {type(exc).__name__}: {exc}",
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
        """Return only the scalar scores for one (prediction, reference) pair.

        Parameters
        ----------
        prediction : str
            The model output to evaluate.
        reference : str
            The ground-truth string.

        Returns
        -------
        Dict[str, Any]
            Mapping of metric name to its raw score value.
        """
        return {k: v.score for k, v in self.evaluate(prediction, reference).items()}

    def batch_evaluate(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Dict[str, EvalResult]]:
        """Evaluate a list of (prediction, reference) pairs.

        Parameters
        ----------
        pairs : List[Tuple[str, str]]
            Each element is a ``(prediction, reference)`` tuple.

        Returns
        -------
        List[Dict[str, EvalResult]]
            One result dict per input pair, in the same order.
        """
        return [self.evaluate(pred, ref) for pred, ref in pairs]

    def summary(self, results: List[Dict[str, EvalResult]]) -> Dict[str, Any]:
        """Aggregate pass rates and average scores across a batch.

        Parameters
        ----------
        results : List[Dict[str, EvalResult]]
            As returned by :meth:`batch_evaluate`.

        Returns
        -------
        Dict[str, Any]
            For each registered metric a sub-dict with keys:

            * ``pass_rate`` – fraction of pairs where the metric passed.
            * ``avg_score`` – mean of all non-``None`` numeric scores, or
              ``None`` when no valid score exists.
            * ``n`` – number of pairs that included this metric.

            Returns an empty dict when *results* is empty.
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
                    numeric_scores.append(float(s))
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
        prediction : str
            The model output to evaluate.
        reference : str
            The ground-truth string.
        min_pass_rate : float, optional
            Minimum fraction of metrics that must pass (``0.0``–``1.0``).
            Default is ``1.0`` (all metrics must pass).

        Returns
        -------
        bool
            ``True`` when the pass rate meets or exceeds the threshold.
            Also returns ``True`` when no metrics are registered.
        """
        results = self.evaluate(prediction, reference)
        if not results:
            return True
        passing = sum(1 for r in results.values() if r.passed)
        return (passing / len(results)) >= min_pass_rate
