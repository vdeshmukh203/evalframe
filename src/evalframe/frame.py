"""Lightweight LLM evaluation framework with built-in metrics and batch support."""
from __future__ import annotations

import json
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class EvalResult:
    """Structured result of evaluating one (prediction, reference) pair.

    Attributes
    ----------
    metric:
        Name of the metric that produced this result.
    score:
        Raw value returned by the metric function.  Boolean for membership
        metrics (``exact_match``, ``contains``, ``prefix_match``); a float
        in [0, 1] for numeric metrics (``f1``, ``rouge1``).
    passed:
        ``True`` when *score* is truthy (``True`` or any non-zero float).
        For float metrics this means any positive overlap is considered a
        pass; callers that need a stricter threshold should inspect *score*
        directly.
    prediction:
        The model-generated string that was evaluated.
    reference:
        The ground-truth string used as the evaluation target.
    """

    metric: str
    score: Any
    passed: bool
    prediction: str
    reference: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------


def _exact_match(pred: str, ref: str) -> bool:
    """Return ``True`` when prediction and reference are identical after stripping whitespace."""
    return pred.strip() == ref.strip()


def _contains(pred: str, ref: str) -> bool:
    """Return ``True`` when the reference string appears verbatim inside the prediction."""
    return ref in pred


def _prefix_match(pred: str, ref: str) -> bool:
    """Return ``True`` when the stripped prediction starts with the stripped reference."""
    return pred.strip().startswith(ref.strip())


def _f1_score(pred: str, ref: str) -> float:
    """Token-level F1 following the SQuAD evaluation protocol.

    Uses ``collections.Counter`` for token frequency so that repeated words
    are handled correctly: the shared token count is
    ``sum((Counter(pred) & Counter(ref)).values())``.

    Returns
    -------
    float
        Harmonic mean of precision and recall in [0, 1], rounded to 4 d.p.
        Returns 0.0 when either string is empty.
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
    """ROUGE-1 recall with proper count-based unigram overlap.

    Uses ``collections.Counter`` so that a word appearing *n* times in the
    reference can only be matched up to *n* times in the prediction (i.e.
    the standard clipped-count definition from Lin 2004).

    Returns
    -------
    float
        Recall in [0, 1], rounded to 4 d.p.  Returns 0.0 when the reference
        is empty.
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


# ---------------------------------------------------------------------------
# Evalframe class
# ---------------------------------------------------------------------------


class Evalframe:
    """Evaluate LLM outputs with pluggable and built-in metrics.

    Parameters
    ----------
    include_builtins:
        When ``True`` all five built-in metrics (``exact_match``,
        ``contains``, ``prefix_match``, ``f1``, ``rouge1``) are registered
        on construction.  Defaults to ``False`` so that users opt in
        explicitly to the metrics they want.

    Examples
    --------
    >>> ef = Evalframe()
    >>> ef.add_builtin("f1")
    >>> ef.score("the cat sat on the mat", "the cat sat")
    {'f1': 0.8571}

    >>> ef = Evalframe(include_builtins=True)
    >>> results = ef.batch_evaluate([("Paris", "Paris"), ("Rome", "Paris")])
    >>> ef.summary(results)["exact_match"]["pass_rate"]
    0.5
    """

    def __init__(self, include_builtins: bool = False) -> None:
        self._metrics: Dict[str, Callable[[str, str], Any]] = {}
        if include_builtins:
            self._metrics.update(BUILTIN_METRICS)

    def __repr__(self) -> str:
        names = list(self._metrics)
        return f"Evalframe(metrics={names})"

    # ------------------------------------------------------------------
    # Metric registration
    # ------------------------------------------------------------------

    def add_metric(self, name: str, fn: Callable[[str, str], Any]) -> None:
        """Register a custom evaluation metric.

        Parameters
        ----------
        name:
            Unique identifier for the metric.  Overwrites any existing
            metric with the same name.
        fn:
            Callable with signature ``(prediction: str, reference: str) ->
            bool | float``.  The return value is stored in
            :attr:`EvalResult.score`; it is cast to ``bool`` to populate
            :attr:`EvalResult.passed`.
        """
        self._metrics[name] = fn

    def add_builtin(self, name: str) -> None:
        """Register a built-in metric by name.

        Parameters
        ----------
        name:
            One of ``"exact_match"``, ``"contains"``, ``"prefix_match"``,
            ``"f1"``, or ``"rouge1"``.

        Raises
        ------
        ValueError
            If *name* is not a recognised built-in.
        """
        if name not in BUILTIN_METRICS:
            raise ValueError(
                f"Unknown built-in metric: {name!r}. "
                f"Available: {list(BUILTIN_METRICS)}"
            )
        self._metrics[name] = BUILTIN_METRICS[name]

    def remove_metric(self, name: str) -> bool:
        """Remove a registered metric.

        Returns
        -------
        bool
            ``True`` if the metric existed and was removed; ``False`` if it
            was not registered.
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

        Parameters
        ----------
        prediction:
            Model-generated output string.
        reference:
            Ground-truth string to compare against.

        Returns
        -------
        dict
            Mapping from metric name to :class:`EvalResult`.  Metrics that
            raise an exception are recorded with ``score=None`` and
            ``passed=False``; a warning is emitted so the error is visible
            without interrupting a batch run.
        """
        results: Dict[str, EvalResult] = {}
        for mname, fn in self._metrics.items():
            try:
                score = fn(prediction, reference)
                passed = bool(score)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Metric '{mname}' raised an exception and will be "
                    f"skipped: {exc}",
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
        """Return the raw scores for all metrics as a flat dictionary.

        This is a convenience wrapper around :meth:`evaluate` for callers
        that do not need the full :class:`EvalResult` objects.
        """
        return {k: v.score for k, v in self.evaluate(prediction, reference).items()}

    def batch_evaluate(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Dict[str, EvalResult]]:
        """Evaluate a list of (prediction, reference) pairs.

        Parameters
        ----------
        pairs:
            Sequence of ``(prediction, reference)`` 2-tuples.

        Returns
        -------
        list
            One result dictionary per pair, in the same order as *pairs*.
        """
        return [self.evaluate(pred, ref) for pred, ref in pairs]

    def summary(
        self, results: List[Dict[str, EvalResult]]
    ) -> Dict[str, Any]:
        """Aggregate pass rates and mean scores across a batch.

        Parameters
        ----------
        results:
            Output of :meth:`batch_evaluate`.

        Returns
        -------
        dict
            Mapping from metric name to a dict with keys:

            * ``"pass_rate"`` – fraction of examples where ``passed`` is
              ``True`` (float, rounded to 4 d.p.).
            * ``"avg_score"`` – arithmetic mean of non-``None`` numeric
              scores (float, rounded to 4 d.p.), or ``None`` when all
              scores are non-numeric or missing.
            * ``"n"`` – total number of examples evaluated.
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
            try:
                avg: Optional[float] = (
                    round(sum(float(s) for s in scores) / len(scores), 4)
                    if scores
                    else None
                )
            except (TypeError, ValueError):
                avg = None
            out[mname] = {
                "pass_rate": round(sum(passes) / len(passes), 4) if passes else 0.0,
                "avg_score": avg,
                "n": len(passes),
            }
        return out

    def assert_passes(
        self,
        prediction: str,
        reference: str,
        min_pass_rate: float = 1.0,
    ) -> bool:
        """Return ``True`` when at least *min_pass_rate* of metrics pass.

        Parameters
        ----------
        prediction:
            Model-generated output string.
        reference:
            Ground-truth string.
        min_pass_rate:
            Minimum fraction of metrics that must pass (default ``1.0``
            requires all metrics to pass).

        Returns
        -------
        bool
            ``True`` when no metrics are registered (vacuously true) or when
            the fraction of passing metrics is ≥ *min_pass_rate*.
        """
        results = self.evaluate(prediction, reference)
        if not results:
            return True
        passing = sum(1 for r in results.values() if r.passed)
        return (passing / len(results)) >= min_pass_rate

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_results(
        results: Union[Dict[str, EvalResult], List[Dict[str, EvalResult]]],
        path: Union[str, Path],
    ) -> None:
        """Serialise evaluation results to a JSON file.

        Parameters
        ----------
        results:
            Either the single-pair dict returned by :meth:`evaluate`, or the
            list returned by :meth:`batch_evaluate`.
        path:
            Destination file path.  Parent directories must already exist.
        """
        if isinstance(results, dict):
            data = {k: v.to_dict() for k, v in results.items()}
        else:
            data = [{k: v.to_dict() for k, v in row.items()} for row in results]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @staticmethod
    def load_results(
        path: Union[str, Path],
    ) -> Union[Dict[str, EvalResult], List[Dict[str, EvalResult]]]:
        """Load serialised evaluation results from a JSON file.

        Returns the same structure that was originally saved: a single dict
        if the file contains a JSON object, or a list of dicts if the file
        contains a JSON array.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        def _from_dict(d: dict) -> EvalResult:
            return EvalResult(
                metric=d["metric"],
                score=d["score"],
                passed=d["passed"],
                prediction=d["prediction"],
                reference=d["reference"],
            )

        if isinstance(data, list):
            return [{k: _from_dict(v) for k, v in row.items()} for row in data]
        return {k: _from_dict(v) for k, v in data.items()}
