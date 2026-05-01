"""Tests for evalframe."""
import warnings

import pytest

from evalframe import Evalframe, EvalResult
from evalframe.frame import BUILTIN_METRICS


# ---------------------------------------------------------------------------
# Metric registration
# ---------------------------------------------------------------------------


def test_add_metric():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert "eq" in e.metrics()


def test_add_metric_overwrites_silently():
    e = Evalframe()
    e.add_metric("m", lambda p, r: True)
    e.add_metric("m", lambda p, r: False)
    assert e.score("x", "x")["m"] is False


def test_add_metric_noncallable_raises():
    e = Evalframe()
    with pytest.raises(TypeError, match="callable"):
        e.add_metric("bad", "not_a_function")  # type: ignore[arg-type]


def test_add_builtin_invalid_raises():
    e = Evalframe()
    with pytest.raises(ValueError, match="Unknown built-in metric"):
        e.add_builtin("nonexistent")


def test_remove_metric_returns_true():
    e = Evalframe()
    e.add_metric("m", lambda p, r: True)
    assert e.remove_metric("m") is True
    assert "m" not in e.metrics()


def test_remove_metric_missing_returns_false():
    e = Evalframe()
    assert e.remove_metric("ghost") is False


def test_metrics_list_order():
    e = Evalframe()
    e.add_metric("a", lambda p, r: True)
    e.add_metric("b", lambda p, r: True)
    assert e.metrics() == ["a", "b"]


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------


def test_evaluate_pass():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert e.evaluate("hi", "hi")["eq"].passed is True


def test_evaluate_fail():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert e.evaluate("hi", "bye")["eq"].passed is False


def test_evaluate_returns_evalresult():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    result = e.evaluate("x", "x")["eq"]
    assert isinstance(result, EvalResult)
    assert result.metric == "eq"
    assert result.prediction == "x"
    assert result.reference == "x"


def test_evaluate_error_emits_warning():
    e = Evalframe()
    e.add_metric("boom", lambda p, r: 1 / 0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = e.evaluate("x", "y")
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
    assert result["boom"].score is None
    assert result["boom"].passed is False


def test_score():
    e = Evalframe()
    e.add_metric("contains", lambda p, r: r in p)
    assert e.score("answer is 42", "42")["contains"] is True


def test_evaluate_no_metrics_returns_empty():
    e = Evalframe()
    assert e.evaluate("x", "y") == {}


# ---------------------------------------------------------------------------
# Built-in metrics — correctness
# ---------------------------------------------------------------------------


def test_builtin_exact_match():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.score("hello", "hello")["exact_match"] is True
    assert e.score("hello", "world")["exact_match"] is False


def test_builtin_exact_match_strips_whitespace():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.score("  hello  ", "hello")["exact_match"] is True


def test_builtin_contains():
    e = Evalframe()
    e.add_builtin("contains")
    assert e.score("the answer is 42", "42")["contains"] is True
    assert e.score("the answer is 42", "99")["contains"] is False


def test_builtin_prefix_match():
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("Hello, world!", "Hello") is not None
    assert e.score("Hello, world!", "Hello")["prefix_match"] is True
    assert e.score("Hello, world!", "world")["prefix_match"] is False


def test_builtin_f1_basic():
    e = Evalframe()
    e.add_builtin("f1")
    score = e.score("the cat sat on the mat", "the cat sat")["f1"]
    assert 0 < score <= 1.0


def test_builtin_f1_perfect():
    e = Evalframe()
    e.add_builtin("f1")
    assert e.score("hello world", "hello world")["f1"] == pytest.approx(1.0)


def test_builtin_f1_no_overlap():
    e = Evalframe()
    e.add_builtin("f1")
    assert e.score("foo bar", "baz qux")["f1"] == pytest.approx(0.0)


def test_builtin_f1_multiset_precision():
    """Repeated prediction tokens should not inflate precision."""
    e = Evalframe()
    e.add_builtin("f1")
    # pred has "the" twice, ref has "the" once.
    # With Counter: common=1, precision=1/2, recall=1/1, F1=2/3 ≈ 0.6667
    score = e.score("the the", "the")["f1"]
    assert score == pytest.approx(2 / 3, rel=1e-3)


def test_builtin_rouge1_basic():
    e = Evalframe()
    e.add_builtin("rouge1")
    score = e.score("the cat sat on the mat", "the cat sat")["rouge1"]
    assert score == pytest.approx(1.0)


def test_builtin_rouge1_partial():
    e = Evalframe()
    e.add_builtin("rouge1")
    score = e.score("the cat", "the cat sat")["rouge1"]
    assert score == pytest.approx(2 / 3, rel=1e-3)


def test_builtin_rouge1_multiset_recall():
    """Repeated reference tokens should not be over-counted."""
    e = Evalframe()
    e.add_builtin("rouge1")
    # pred has "the" once, ref has "the" three times.
    # correct recall = 1/3; naive list check would give 3/3=1.0.
    score = e.score("the cat", "the the the")["rouge1"]
    assert score == pytest.approx(1 / 3, rel=1e-3)


def test_builtin_f1_empty_pred():
    e = Evalframe()
    e.add_builtin("f1")
    assert e.score("", "hello")["f1"] == pytest.approx(0.0)


def test_builtin_rouge1_empty_ref():
    e = Evalframe()
    e.add_builtin("rouge1")
    assert e.score("hello", "")["rouge1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# include_builtins flag
# ---------------------------------------------------------------------------


def test_include_builtins_flag():
    e = Evalframe(include_builtins=True)
    for name in BUILTIN_METRICS:
        assert name in e.metrics()


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def test_batch_evaluate():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    results = e.batch_evaluate(pairs)
    assert len(results) == 3


def test_batch_evaluate_empty():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.batch_evaluate([]) == []


def test_batch_evaluate_pass_counts():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    results = e.batch_evaluate(pairs)
    passes = [r["exact_match"].passed for r in results]
    assert passes == [True, False, True]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    results = e.batch_evaluate(pairs)
    s = e.summary(results)
    assert s["exact_match"]["pass_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert s["exact_match"]["n"] == 3


def test_summary_empty_returns_empty_dict():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.summary([]) == {}


def test_summary_avg_score_float_metric():
    e = Evalframe()
    e.add_builtin("f1")
    pairs = [("cat", "cat"), ("dog", "cat")]
    results = e.batch_evaluate(pairs)
    s = e.summary(results)
    assert s["f1"]["avg_score"] is not None
    assert 0.0 <= s["f1"]["avg_score"] <= 1.0


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


def test_assert_passes_fails():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.assert_passes("wrong", "right", min_pass_rate=1.0) is False


def test_assert_passes_no_metrics_returns_true():
    e = Evalframe()
    assert e.assert_passes("x", "y") is True
