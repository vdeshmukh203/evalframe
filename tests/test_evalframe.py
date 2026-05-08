"""Tests for evalframe."""
import warnings

import pytest

from evalframe import BUILTIN_METRICS, EvalResult, Evalframe
from evalframe.frame import _f1_score, _rouge1


# ---------------------------------------------------------------------------
# Metric registration
# ---------------------------------------------------------------------------

def test_add_metric():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert "eq" in e.metrics()


def test_add_metric_invalid_name_empty():
    e = Evalframe()
    with pytest.raises(ValueError, match="non-empty"):
        e.add_metric("", lambda p, r: True)


def test_add_metric_invalid_name_whitespace():
    e = Evalframe()
    with pytest.raises(ValueError, match="non-empty"):
        e.add_metric("   ", lambda p, r: True)


def test_add_metric_not_callable():
    e = Evalframe()
    with pytest.raises(TypeError, match="callable"):
        e.add_metric("bad", "not_a_function")  # type: ignore[arg-type]


def test_add_builtin_unknown():
    e = Evalframe()
    with pytest.raises(ValueError, match="Unknown built-in"):
        e.add_builtin("nonexistent_metric")


# ---------------------------------------------------------------------------
# evaluate() – pass / fail / TypeError on bad input
# ---------------------------------------------------------------------------

def test_evaluate_pass():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert e.evaluate("hi", "hi")["eq"].passed is True


def test_evaluate_fail():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert e.evaluate("hi", "bye")["eq"].passed is False


def test_evaluate_non_string_prediction():
    e = Evalframe()
    e.add_builtin("exact_match")
    with pytest.raises(TypeError, match="prediction"):
        e.evaluate(42, "ref")  # type: ignore[arg-type]


def test_evaluate_non_string_reference():
    e = Evalframe()
    e.add_builtin("exact_match")
    with pytest.raises(TypeError, match="reference"):
        e.evaluate("pred", None)  # type: ignore[arg-type]


def test_evaluate_metric_error_stored():
    """A metric that raises should record error=<message>, score=None, passed=False."""
    e = Evalframe()

    def boom(p, r):
        raise RuntimeError("kaboom")

    e.add_metric("broken", boom)
    result = e.evaluate("x", "y")["broken"]
    assert result.score is None
    assert result.passed is False
    assert "kaboom" in (result.error or "")


def test_evalresult_error_field_none_on_success():
    e = Evalframe()
    e.add_builtin("exact_match")
    result = e.evaluate("hello", "hello")["exact_match"]
    assert result.error is None


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

def test_score():
    e = Evalframe()
    e.add_metric("contains", lambda p, r: r in p)
    assert e.score("answer is 42", "42")["contains"] is True


# ---------------------------------------------------------------------------
# Built-in metrics – correctness
# ---------------------------------------------------------------------------

def test_builtin_exact_match():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.score("hello", "hello")["exact_match"] is True
    assert e.score("hello", "world")["exact_match"] is False


def test_builtin_contains():
    e = Evalframe()
    e.add_builtin("contains")
    assert e.score("the answer is 42", "42")["contains"] is True


def test_builtin_prefix_match():
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("Paris is the capital", "Paris")["prefix_match"] is True
    assert e.score("The capital is Paris", "Paris")["prefix_match"] is False


def test_builtin_f1():
    e = Evalframe()
    e.add_builtin("f1")
    score = e.score("the cat sat on the mat", "the cat sat")["f1"]
    assert 0 < score <= 1.0


def test_builtin_f1_repeated_tokens():
    """Counter-based F1: repeated pred tokens must not inflate precision."""
    # pred has "cat" 3× but ref has it 1×; only 1 common token should count
    score = _f1_score("cat cat cat", "cat")
    # common=1, precision=1/3, recall=1/1 → F1 = 2*(1/3*1)/(1/3+1) ≈ 0.5
    assert abs(score - 0.5) < 0.001


def test_builtin_f1_empty_pred():
    assert _f1_score("", "cat") == 0.0


def test_builtin_f1_empty_ref():
    assert _f1_score("cat", "") == 0.0


def test_builtin_rouge1():
    e = Evalframe()
    e.add_builtin("rouge1")
    score = e.score("the cat sat on the mat", "the cat sat")["rouge1"]
    assert score == 1.0


def test_builtin_rouge1_repeated_ref_tokens():
    """Counter-based ROUGE-1: repeated ref tokens are clipped to pred count."""
    # ref = "cat cat cat" (3 tokens), pred = "cat" (1 token)
    # common = min(1,3) = 1, recall = 1/3
    score = _rouge1("cat", "cat cat cat")
    assert abs(score - round(1 / 3, 4)) < 0.0001


def test_builtin_rouge1_empty_ref():
    assert _rouge1("cat", "") == 0.0


# ---------------------------------------------------------------------------
# include_builtins flag & remove_metric
# ---------------------------------------------------------------------------

def test_include_builtins_flag():
    e = Evalframe(include_builtins=True)
    assert "exact_match" in e.metrics()
    assert "f1" in e.metrics()
    assert set(e.metrics()) == set(BUILTIN_METRICS)


def test_remove_metric():
    e = Evalframe()
    e.add_metric("m", lambda p, r: True)
    assert e.remove_metric("m") is True
    assert "m" not in e.metrics()


def test_remove_metric_missing():
    e = Evalframe()
    assert e.remove_metric("nonexistent") is False


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
    e = Evalframe(include_builtins=True)
    assert e.batch_evaluate([]) == []


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

def test_summary():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    results = e.batch_evaluate(pairs)
    s = e.summary(results)
    assert s["exact_match"]["pass_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert s["exact_match"]["n"] == 3


def test_summary_empty_results():
    e = Evalframe(include_builtins=True)
    assert e.summary([]) == {}


def test_summary_avg_score_numeric():
    e = Evalframe()
    e.add_builtin("f1")
    pairs = [("cat sat", "cat sat"), ("dog ran", "cat sat")]
    s = e.summary(e.batch_evaluate(pairs))
    assert s["f1"]["avg_score"] is not None
    assert 0.0 <= s["f1"]["avg_score"] <= 1.0


# ---------------------------------------------------------------------------
# assert_passes()
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


def test_assert_passes_fail():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.assert_passes("wrong answer", "correct", min_pass_rate=1.0) is False


def test_assert_passes_no_metrics_warns():
    e = Evalframe()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = e.assert_passes("x", "y")
    assert result is True
    assert any(issubclass(w.category, UserWarning) for w in caught)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

def test_evalresult_importable():
    from evalframe import EvalResult  # noqa: F401
    assert EvalResult is not None


def test_builtin_metrics_importable():
    from evalframe import BUILTIN_METRICS  # noqa: F401
    assert "f1" in BUILTIN_METRICS
