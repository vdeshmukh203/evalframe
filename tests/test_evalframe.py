"""Tests for evalframe."""
import csv
import io
import pytest

from evalframe import Evalframe
from evalframe.frame import BUILTIN_METRICS, EvalResult


# ---------------------------------------------------------------------------
# Metric registration
# ---------------------------------------------------------------------------

def test_add_metric():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert "eq" in e.metrics()


def test_add_metric_not_callable_raises():
    e = Evalframe()
    with pytest.raises(TypeError):
        e.add_metric("bad", "not_a_function")


def test_remove_metric_existing():
    e = Evalframe()
    e.add_metric("m", lambda p, r: True)
    assert e.remove_metric("m") is True
    assert "m" not in e.metrics()


def test_remove_metric_missing():
    e = Evalframe()
    assert e.remove_metric("nonexistent") is False


def test_add_builtin_unknown_raises():
    e = Evalframe()
    with pytest.raises(ValueError, match="Unknown built-in metric"):
        e.add_builtin("no_such_metric")


def test_include_builtins_flag():
    e = Evalframe(include_builtins=True)
    for name in BUILTIN_METRICS:
        assert name in e.metrics()


def test_repr_contains_metrics():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert "exact_match" in repr(e)


# ---------------------------------------------------------------------------
# evaluate / score
# ---------------------------------------------------------------------------

def test_evaluate_pass():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert e.evaluate("hi", "hi")["eq"].passed is True


def test_evaluate_fail():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert e.evaluate("hi", "bye")["eq"].passed is False


def test_evaluate_result_fields():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    result = e.evaluate("foo", "foo")["eq"]
    assert isinstance(result, EvalResult)
    assert result.metric == "eq"
    assert result.prediction == "foo"
    assert result.reference == "foo"


def test_evaluate_exception_in_metric_gives_none_score():
    e = Evalframe()
    e.add_metric("boom", lambda p, r: 1 / 0)
    result = e.evaluate("x", "y")["boom"]
    assert result.score is None
    assert result.passed is False


def test_evaluate_exception_does_not_abort_other_metrics():
    e = Evalframe()
    e.add_metric("boom", lambda p, r: 1 / 0)
    e.add_metric("ok", lambda p, r: True)
    results = e.evaluate("x", "y")
    assert results["ok"].passed is True
    assert results["boom"].score is None


def test_score():
    e = Evalframe()
    e.add_metric("contains", lambda p, r: r in p)
    assert e.score("answer is 42", "42")["contains"] is True


def test_eval_result_to_dict():
    er = EvalResult(metric="f1", score=0.75, passed=True, prediction="a", reference="b")
    d = er.to_dict()
    assert d == {
        "metric": "f1", "score": 0.75, "passed": True,
        "prediction": "a", "reference": "b",
    }


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------

def test_builtin_exact_match_pass():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.score("hello", "hello")["exact_match"] is True


def test_builtin_exact_match_fail():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.score("hello", "world")["exact_match"] is False


def test_builtin_exact_match_strips_whitespace():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.score("  hello  ", "hello")["exact_match"] is True


def test_builtin_contains():
    e = Evalframe()
    e.add_builtin("contains")
    assert e.score("the answer is 42", "42")["contains"] is True


def test_builtin_contains_fail():
    e = Evalframe()
    e.add_builtin("contains")
    assert e.score("the answer is 42", "43")["contains"] is False


def test_builtin_prefix_match_pass():
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("hello world", "hello")["prefix_match"] is True


def test_builtin_prefix_match_fail():
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("world hello", "hello")["prefix_match"] is False


def test_builtin_prefix_match_strips_whitespace():
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("  hello world", "hello")["prefix_match"] is True


def test_builtin_f1_partial_overlap():
    e = Evalframe()
    e.add_builtin("f1")
    score = e.score("the cat sat on the mat", "the cat sat")["f1"]
    assert 0 < score <= 1.0


def test_builtin_f1_perfect():
    e = Evalframe()
    e.add_builtin("f1")
    assert e.score("cat sat", "cat sat")["f1"] == pytest.approx(1.0)


def test_builtin_f1_empty_prediction():
    e = Evalframe()
    e.add_builtin("f1")
    assert e.score("", "cat sat")["f1"] == 0.0


def test_builtin_f1_empty_reference():
    e = Evalframe()
    e.add_builtin("f1")
    assert e.score("cat sat", "")["f1"] == 0.0


def test_builtin_rouge1_perfect_recall():
    e = Evalframe()
    e.add_builtin("rouge1")
    assert e.score("the cat sat on the mat", "the cat sat")["rouge1"] == 1.0


def test_builtin_rouge1_partial():
    e = Evalframe()
    e.add_builtin("rouge1")
    score = e.score("the cat", "the cat sat")["rouge1"]
    assert 0 < score < 1.0


def test_builtin_rouge1_empty_reference():
    e = Evalframe()
    e.add_builtin("rouge1")
    assert e.score("something", "")["rouge1"] == 0.0


# ---------------------------------------------------------------------------
# batch_evaluate
# ---------------------------------------------------------------------------

def test_batch_evaluate_length():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    assert len(e.batch_evaluate(pairs)) == 3


def test_batch_evaluate_empty():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.batch_evaluate([]) == []


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_pass_rate():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    s = e.summary(e.batch_evaluate(pairs))
    assert s["exact_match"]["pass_rate"] == pytest.approx(2 / 3, rel=1e-3)


def test_summary_n():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c")]
    s = e.summary(e.batch_evaluate(pairs))
    assert s["exact_match"]["n"] == 2


def test_summary_avg_score_boolean_metrics():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "b"), ("c", "x")]
    s = e.summary(e.batch_evaluate(pairs))
    assert s["exact_match"]["avg_score"] == pytest.approx(2 / 3, rel=1e-3)


def test_summary_empty_returns_empty_dict():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.summary([]) == {}


# ---------------------------------------------------------------------------
# assert_passes
# ---------------------------------------------------------------------------

def test_assert_passes_all():
    e = Evalframe()
    e.add_builtin("exact_match")
    e.add_builtin("contains")
    assert e.assert_passes("42", "42", min_pass_rate=1.0) is True


def test_assert_passes_partial():
    e = Evalframe()
    e.add_builtin("exact_match")
    e.add_builtin("contains")
    assert e.assert_passes("the answer is 42", "42", min_pass_rate=0.5) is True


def test_assert_passes_no_metrics_returns_true():
    e = Evalframe()
    assert e.assert_passes("x", "y") is True


def test_assert_passes_invalid_rate_raises():
    e = Evalframe()
    e.add_builtin("exact_match")
    with pytest.raises(ValueError):
        e.assert_passes("x", "y", min_pass_rate=1.5)


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------

def test_to_csv_structure():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c")]
    results = e.batch_evaluate(pairs)
    csv_text = e.to_csv(results)
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert len(rows) == 2
    assert set(rows[0].keys()) == {
        "pair_index", "metric", "score", "passed", "prediction", "reference"
    }


def test_to_csv_values():
    e = Evalframe()
    e.add_builtin("exact_match")
    results = e.batch_evaluate([("hello", "hello")])
    reader = csv.DictReader(io.StringIO(e.to_csv(results)))
    row = next(reader)
    assert row["metric"] == "exact_match"
    assert row["passed"] == "True"
    assert row["prediction"] == "hello"
