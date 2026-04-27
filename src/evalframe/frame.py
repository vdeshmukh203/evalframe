"""Lightweight LLM evaluation framework with built-in metrics and batch support."""
from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class EvalResult:
    """Holds the outcome of one metric applied to one (prediction, reference) pair.

    Attributes:
        metric: Name of the metric that produced this result.
        score: Raw value returned by the metric function (``bool`` or ``float``).
            A value of ``None`` means the metric raised an exception.
        passed: ``bool(score)`` — ``False`` when *score* is ``0``, ``0.0``,
            ``False``, or ``None``.
        prediction: The model output that was evaluated.
        reference: The ground-truth string used as the evaluation target.
    """

    metric: str
    score: Any
    passed: bool
    prediction: str
    reference: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict copy suitable for serialisation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------

def _exact_match(pred: str, ref: str) -> bool:
    """Return ``True`` when stripped prediction equals stripped reference."""
    return pred.strip() == ref.strip()


def _contains(pred: str, ref: str) -> bool:
    """Return ``True`` when the reference string appears verbatim in prediction."""
    return ref in pred


def _prefix_match(pred: str, ref: str) -> bool:
    """Return ``True`` when the stripped prediction starts with the stripped reference."""
    return pred.strip().startswith(ref.strip())


def _f1_score(pred: str, ref: str) -> float:
    """Token-level F1 based on word-set overlap.

    Returns the harmonic mean of precision (fraction of prediction tokens that
    appear in the reference) and recall (fraction of reference tokens that appear
    in the prediction).  Empty inputs yield ``0.0``.
    """
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
    """ROUGE-1 recall: fraction of reference unigrams present in the prediction.

    An empty reference yields ``0.0``.
    """
    pred_tokens = set(pred.lower().split())   # set for O(1) membership test
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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Evalframe:
    """Evaluate LLM outputs with pluggable and built-in metrics.

    Parameters
    ----------
    include_builtins:
        When ``True`` all five built-in metrics (``exact_match``, ``contains``,
        ``prefix_match``, ``f1``, ``rouge1``) are registered at construction time.

    Examples
    --------
    >>> from evalframe import Evalframe
    >>> ef = Evalframe()
    >>> ef.add_builtin("exact_match")
    >>> ef.score("hello", "hello")
    {'exact_match': True}
    """

    def __init__(self, include_builtins: bool = False) -> None:
        self._metrics: Dict[str, Callable[[str, str], Any]] = {}
        if include_builtins:
            self._metrics.update(BUILTIN_METRICS)

    def __repr__(self) -> str:
        names = list(self._metrics)
        return f"Evalframe(metrics={names!r})"

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
            Callable with signature ``(prediction: str, reference: str) -> Any``.
            The return value is cast to ``bool`` to determine *passed*.
        """
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__!r}")
        self._metrics[name] = fn

    def add_builtin(self, name: str) -> None:
        """Enable a named built-in metric.

        Parameters
        ----------
        name:
            One of ``"exact_match"``, ``"contains"``, ``"prefix_match"``,
            ``"f1"``, ``"rouge1"``.

        Raises
        ------
        ValueError
            If *name* is not a recognised built-in.
        """
        if name not in BUILTIN_METRICS:
            raise ValueError(
                f"Unknown built-in metric: {name!r}. "
                f"Available: {sorted(BUILTIN_METRICS)}"
            )
        self._metrics[name] = BUILTIN_METRICS[name]

    def remove_metric(self, name: str) -> bool:
        """Remove a metric by name.

        Parameters
        ----------
        name:
            The metric to remove.

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

    def evaluate(
        self, prediction: str, reference: str
    ) -> Dict[str, EvalResult]:
        """Run all registered metrics on one (prediction, reference) pair.

        Each metric is called independently; an exception in one metric does
        not abort the others.  A failing metric yields ``score=None`` and
        ``passed=False``.

        Parameters
        ----------
        prediction:
            The model-generated string to evaluate.
        reference:
            The ground-truth string to compare against.

        Returns
        -------
        Dict[str, EvalResult]
            Mapping from metric name to its :class:`EvalResult`.
        """
        results: Dict[str, EvalResult] = {}
        for mname, fn in self._metrics.items():
            try:
                score = fn(prediction, reference)
                passed = bool(score)
            except Exception:
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
        """Return raw metric scores as a flat ``{name: value}`` dict.

        Convenience wrapper around :meth:`evaluate`.
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
        List[Dict[str, EvalResult]]
            One result dict per pair, in the same order as *pairs*.
        """
        return [self.evaluate(pred, ref) for pred, ref in pairs]

    def summary(
        self, results: List[Dict[str, EvalResult]]
    ) -> Dict[str, Any]:
        """Aggregate pass rates and average scores across a batch.

        Parameters
        ----------
        results:
            Output of :meth:`batch_evaluate`.

        Returns
        -------
        Dict[str, Any]
            Per-metric dict with keys ``pass_rate``, ``avg_score``, and ``n``
            (number of evaluated pairs for that metric).  Returns an empty
            dict when *results* is empty.
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
        """Return ``True`` if at least *min_pass_rate* fraction of metrics pass.

        Parameters
        ----------
        prediction:
            The model-generated string to evaluate.
        reference:
            The ground-truth string to compare against.
        min_pass_rate:
            Required fraction of passing metrics, in ``[0, 1]``.  Defaults to
            ``1.0`` (all metrics must pass).

        Raises
        ------
        ValueError
            If *min_pass_rate* is outside ``[0, 1]``.
        """
        if not 0.0 <= min_pass_rate <= 1.0:
            raise ValueError(
                f"min_pass_rate must be in [0, 1], got {min_pass_rate!r}"
            )
        results = self.evaluate(prediction, reference)
        if not results:
            return True
        passing = sum(1 for r in results.values() if r.passed)
        return (passing / len(results)) >= min_pass_rate

    def to_csv(
        self,
        results: List[Dict[str, EvalResult]],
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Serialise batch results to a CSV string.

        Parameters
        ----------
        results:
            Output of :meth:`batch_evaluate`.
        pairs:
            Optional list of ``(prediction, reference)`` tuples used to
            populate the *prediction* and *reference* columns.  When omitted,
            these columns are populated from the :class:`EvalResult` objects.

        Returns
        -------
        str
            CSV text with columns: ``pair_index``, ``metric``, ``score``,
            ``passed``, ``prediction``, ``reference``.
        """
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["pair_index", "metric", "score", "passed",
                        "prediction", "reference"],
            lineterminator="\n",
        )
        writer.writeheader()
        for i, row_dict in enumerate(results):
            for mname, er in row_dict.items():
                writer.writerow({
                    "pair_index": i,
                    "metric": mname,
                    "score": er.score,
                    "passed": er.passed,
                    "prediction": er.prediction,
                    "reference": er.reference,
                })
        return buf.getvalue()
