"""Tests for evalframe."""
import warnings

import pytest

from evalframe import Evalframe
from evalframe.frame import BUILTIN_METRICS


# ── Metric registration ────────────────────────────────────────────────────────

def test_add_metric():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert "eq" in e.metrics()


def test_add_builtin_unknown_raises():
    e = Evalframe()
    with pytest.raises(ValueError, match="Unknown built-in metric"):
        e.add_builtin("does_not_exist")


def test_remove_metric():
    e = Evalframe()
    e.add_metric("m", lambda p, r: True)
    assert e.remove_metric("m") is True
    assert "m" not in e.metrics()


def test_remove_metric_missing():
    e = Evalframe()
    assert e.remove_metric("nonexistent") is False


# ── Single evaluation ──────────────────────────────────────────────────────────

def test_evaluate_pass():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert e.evaluate("hi", "hi")["eq"].passed is True


def test_evaluate_fail():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert e.evaluate("hi", "bye")["eq"].passed is False


def test_evaluate_emits_warning_on_metric_error():
    e = Evalframe()
    e.add_metric("broken", lambda p, r: 1 / 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = e.evaluate("a", "b")
    assert result["broken"].passed is False
    assert result["broken"].score is None
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_score():
    e = Evalframe()
    e.add_metric("contains", lambda p, r: r in p)
    assert e.score("answer is 42", "42")["contains"] is True


# ── Built-in: exact_match ──────────────────────────────────────────────────────

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


# ── Built-in: contains ────────────────────────────────────────────────────────

def test_builtin_contains_pass():
    e = Evalframe()
    e.add_builtin("contains")
    assert e.score("the answer is 42", "42")["contains"] is True


def test_builtin_contains_fail():
    e = Evalframe()
    e.add_builtin("contains")
    assert e.score("the answer is 42", "99")["contains"] is False


def test_builtin_contains_empty_reference_returns_false():
    """Empty reference must not vacuously match."""
    e = Evalframe()
    e.add_builtin("contains")
    assert e.score("hello world", "")["contains"] is False


# ── Built-in: prefix_match ────────────────────────────────────────────────────

def test_builtin_prefix_match_pass():
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("Paris is the capital", "Paris")["prefix_match"] is True


def test_builtin_prefix_match_fail():
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("London is the capital", "Paris")["prefix_match"] is False


def test_builtin_prefix_match_empty_reference_returns_false():
    """Empty reference must not vacuously match."""
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("hello world", "")["prefix_match"] is False


# ── Built-in: f1 ──────────────────────────────────────────────────────────────

def test_builtin_f1_partial_overlap():
    e = Evalframe()
    e.add_builtin("f1")
    score = e.score("the cat sat on the mat", "the cat sat")["f1"]
    assert 0 < score <= 1.0


def test_builtin_f1_perfect():
    e = Evalframe()
    e.add_builtin("f1")
    assert e.score("the cat", "the cat")["f1"] == pytest.approx(1.0)


def test_builtin_f1_no_overlap():
    e = Evalframe()
    e.add_builtin("f1")
    assert e.score("dog", "cat")["f1"] == pytest.approx(0.0)


def test_builtin_f1_repeated_tokens_uses_counter():
    """Repeated tokens must be handled with counts, not sets.

    pred="the the the" vs ref="the": set-based F1 would give 1.0;
    counter-based (SQuAD) precision = 1/3, recall = 1/1, F1 = 0.5.
    """
    e = Evalframe()
    e.add_builtin("f1")
    score = e.score("the the the", "the")["f1"]
    assert score == pytest.approx(0.5, rel=1e-3)


def test_builtin_f1_empty_prediction():
    e = Evalframe()
    e.add_builtin("f1")
    assert e.score("", "cat")["f1"] == pytest.approx(0.0)


# ── Built-in: rouge1 ──────────────────────────────────────────────────────────

def test_builtin_rouge1_full_recall():
    e = Evalframe()
    e.add_builtin("rouge1")
    score = e.score("the cat sat on the mat", "the cat sat")["rouge1"]
    assert score == pytest.approx(1.0)


def test_builtin_rouge1_partial_recall():
    e = Evalframe()
    e.add_builtin("rouge1")
    score = e.score("cat", "the cat")["rouge1"]
    assert score == pytest.approx(0.5, rel=1e-3)


def test_builtin_rouge1_clipped_counts():
    """Clipping prevents the prediction inflating recall via repetition.

    pred="the the the cat" (the×3, cat×1)
    ref="the cat"         (the×1, cat×1)
    clipped intersection: the×1 + cat×1 = 2
    recall = 2/2 = 1.0  (same as a non-repeating pred)
    """
    e = Evalframe()
    e.add_builtin("rouge1")
    score = e.score("the the the cat", "the cat")["rouge1"]
    assert score == pytest.approx(1.0)


def test_builtin_rouge1_empty_reference():
    e = Evalframe()
    e.add_builtin("rouge1")
    assert e.score("something", "")["rouge1"] == pytest.approx(0.0)


# ── include_builtins flag ──────────────────────────────────────────────────────

def test_include_builtins_flag():
    e = Evalframe(include_builtins=True)
    assert set(e.metrics()) == set(BUILTIN_METRICS)


# ── Batch evaluation ───────────────────────────────────────────────────────────

def test_batch_evaluate_length():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    results = e.batch_evaluate(pairs)
    assert len(results) == 3


def test_summary_pass_rate():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    results = e.batch_evaluate(pairs)
    s = e.summary(results)
    assert s["exact_match"]["pass_rate"] == pytest.approx(2 / 3, rel=1e-3)


def test_summary_empty_input():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.summary([]) == {}


# ── assert_passes ─────────────────────────────────────────────────────────────

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


def test_assert_passes_no_metrics_raises():
    """assert_passes with no registered metrics must raise ValueError."""
    e = Evalframe()
    with pytest.raises(ValueError, match="no metrics registered"):
        e.assert_passes("hello", "hello")
