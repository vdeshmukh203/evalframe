"""Lightweight LLM evaluation framework with built-in metrics and batch support."""
from __future__ import annotations

import csv
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class EvalResult:
    """Structured result produced by a single metric evaluation.

    Attributes
    ----------
    metric:
        Name of the metric that produced this result.
    score:
        Raw value returned by the metric function.  Boolean metrics return
        ``True``/``False``; float-valued metrics return a value in ``[0, 1]``.
    passed:
        ``True`` when *score* is truthy (non-zero / non-empty / ``True``).
    prediction:
        The model output that was evaluated.
    reference:
        The ground-truth string used as the evaluation target.
    """

    metric: str
    score: Any
    passed: bool
    prediction: str
    reference: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation suitable for JSON serialisation."""
        return {
            "metric": self.metric,
            "score": self.score,
            "passed": self.passed,
            "prediction": self.prediction,
            "reference": self.reference,
        }

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"EvalResult(metric={self.metric!r}, score={self.score!r}, [{status}])"


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------

def _exact_match(pred: str, ref: str) -> bool:
    """Return ``True`` when the stripped strings are identical."""
    return pred.strip() == ref.strip()


def _contains(pred: str, ref: str) -> bool:
    """Return ``True`` when *pred* contains *ref* as a substring."""
    return ref in pred


def _prefix_match(pred: str, ref: str) -> bool:
    """Return ``True`` when *pred* starts with *ref* (both stripped)."""
    return pred.strip().startswith(ref.strip())


def _f1_score(pred: str, ref: str) -> float:
    """Token-level F1 (word overlap) using multiset (bag-of-words) counts.

    Precision and recall are computed over clipped token counts so that
    repeated tokens are not overcounted — matching the standard SQuAD metric.

    Parameters
    ----------
    pred:
        Predicted text.
    ref:
        Reference text.

    Returns
    -------
    float
        F1 score in ``[0.0, 1.0]``, rounded to four decimal places.
        Returns ``0.0`` when either string is empty.
    """
    pred_counter = Counter(pred.lower().split())
    ref_counter = Counter(ref.lower().split())
    if not pred_counter or not ref_counter:
        return 0.0
    common = sum((pred_counter & ref_counter).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_counter.values())
    recall = common / sum(ref_counter.values())
    return round(2 * precision * recall / (precision + recall), 4)


def _rouge1(pred: str, ref: str) -> float:
    """ROUGE-1 recall using clipped unigram counts.

    Each reference token is counted at most as many times as it appears in
    the prediction, preventing artificial inflation from repeated words.

    Parameters
    ----------
    pred:
        Predicted (system) text.
    ref:
        Reference text.

    Returns
    -------
    float
        ROUGE-1 recall in ``[0.0, 1.0]``, rounded to four decimal places.
        Returns ``0.0`` when the reference is empty.
    """
    pred_counter = Counter(pred.lower().split())
    ref_counter = Counter(ref.lower().split())
    if not ref_counter:
        return 0.0
    clipped = sum((pred_counter & ref_counter).values())
    return round(clipped / sum(ref_counter.values()), 4)


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
        When ``True`` all five built-in metrics (``exact_match``,
        ``contains``, ``prefix_match``, ``f1``, ``rouge1``) are registered
        on initialisation.  Defaults to ``False`` so the starting metric set
        is explicit.

    Examples
    --------
    >>> ev = Evalframe()
    >>> ev.add_builtin("exact_match")
    >>> ev.add_builtin("f1")
    >>> result = ev.evaluate("the cat sat", "the cat sat")
    >>> result["exact_match"].passed
    True
    >>> result["f1"].score
    1.0
    """

    def __init__(self, include_builtins: bool = False) -> None:
        self._metrics: Dict[str, Callable[[str, str], Any]] = {}
        if include_builtins:
            self._metrics.update(BUILTIN_METRICS)

    def __repr__(self) -> str:
        return f"Evalframe(metrics={self.metrics()!r})"

    # ------------------------------------------------------------------
    # Metric registration
    # ------------------------------------------------------------------

    def add_metric(self, name: str, fn: Callable[[str, str], Any]) -> None:
        """Register a custom evaluation metric.

        Parameters
        ----------
        name:
            Unique identifier for the metric.  Overwrites any existing metric
            with the same name.
        fn:
            Callable that accepts ``(prediction, reference)`` and returns a
            score.  Boolean or float return values work with all aggregation
            helpers.
        """
        self._metrics[name] = fn

    def add_builtin(self, name: str) -> None:
        """Register one of the built-in metrics by name.

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

        Parameters
        ----------
        name:
            Name of the metric to remove.

        Returns
        -------
        bool
            ``True`` if the metric existed and was removed, ``False`` if it
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
    # Single-pair evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, prediction: str, reference: str
    ) -> Dict[str, EvalResult]:
        """Run all registered metrics on one ``(prediction, reference)`` pair.

        Metric functions that raise an exception produce an
        :class:`EvalResult` with ``score=None`` and ``passed=False``, and a
        :mod:`warnings` message is emitted.

        Parameters
        ----------
        prediction:
            Text generated by the model under evaluation.
        reference:
            Ground-truth target string.

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
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Metric {mname!r} raised an exception and will be "
                    f"recorded as failed: {exc}",
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

    def score(
        self, prediction: str, reference: str
    ) -> Dict[str, Any]:
        """Return only the raw scores as a flat dict.

        Convenience wrapper around :meth:`evaluate` when :class:`EvalResult`
        metadata is not needed.

        Returns
        -------
        Dict[str, Any]
            Mapping from metric name to its score value.
        """
        return {k: v.score for k, v in self.evaluate(prediction, reference).items()}

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def batch_evaluate(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Dict[str, EvalResult]]:
        """Evaluate a list of ``(prediction, reference)`` pairs.

        Parameters
        ----------
        pairs:
            Ordered sequence of ``(prediction, reference)`` tuples.

        Returns
        -------
        List[Dict[str, EvalResult]]
            One result dict per input pair, in the same order.
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
            Per-metric dict with keys ``"pass_rate"``, ``"avg_score"``,
            and ``"n"`` (number of evaluated pairs).  ``"avg_score"`` is
            ``None`` when all scores are non-numeric.
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
        """Return ``True`` when the fraction of passing metrics >= *min_pass_rate*.

        Parameters
        ----------
        prediction:
            Text generated by the model.
        reference:
            Ground-truth target string.
        min_pass_rate:
            Minimum fraction of metrics that must pass.  Defaults to
            ``1.0`` (all metrics must pass).

        Returns
        -------
        bool
            ``True`` if the pass-rate threshold is met (or no metrics are
            registered).
        """
        results = self.evaluate(prediction, reference)
        if not results:
            return True
        passing = sum(1 for r in results.values() if r.passed)
        return (passing / len(results)) >= min_pass_rate

    # ------------------------------------------------------------------
    # CSV I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_pairs_csv(path: Union[str, Path]) -> List[Tuple[str, str]]:
        """Load ``(prediction, reference)`` pairs from a CSV file.

        The file must contain columns named ``prediction`` and
        ``reference`` (case-insensitive).  Extra columns are ignored.

        Parameters
        ----------
        path:
            Path to a UTF-8 encoded CSV file.

        Returns
        -------
        List[Tuple[str, str]]
            Ordered list of ``(prediction, reference)`` pairs.

        Raises
        ------
        ValueError
            If the required columns are absent.
        OSError
            If the file cannot be opened.
        """
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError("CSV file appears to be empty.")
            lower = {n.lower(): n for n in reader.fieldnames}
            if "prediction" not in lower or "reference" not in lower:
                raise ValueError(
                    "CSV must contain 'prediction' and 'reference' columns; "
                    f"found: {list(reader.fieldnames)}"
                )
            pred_col = lower["prediction"]
            ref_col = lower["reference"]
            return [(row[pred_col], row[ref_col]) for row in reader]

    @staticmethod
    def save_results_csv(
        results: List[Dict[str, EvalResult]],
        pairs: List[Tuple[str, str]],
        path: Union[str, Path],
    ) -> None:
        """Write batch evaluation results to a CSV file.

        The output contains one row per pair with columns
        ``prediction``, ``reference``, and for each metric
        ``{metric}_score`` and ``{metric}_pass``.

        Parameters
        ----------
        results:
            Output of :meth:`batch_evaluate`.
        pairs:
            The corresponding input pairs (same order and length).
        path:
            Destination file path.

        Raises
        ------
        ValueError
            If *results* and *pairs* have different lengths.
        """
        if len(results) != len(pairs):
            raise ValueError(
                f"results (len={len(results)}) and pairs (len={len(pairs)}) "
                "must have the same length."
            )
        if not results:
            return

        metric_names = list(results[0].keys())
        fieldnames = ["prediction", "reference"] + [
            f"{m}_{suffix}"
            for m in metric_names
            for suffix in ("score", "pass")
        ]

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for (pred, ref), res in zip(pairs, results):
                row: Dict[str, Any] = {"prediction": pred, "reference": ref}
                for m in metric_names:
                    r = res.get(m)
                    row[f"{m}_score"] = r.score if r else None
                    row[f"{m}_pass"] = int(r.passed) if r else None
                writer.writerow(row)
