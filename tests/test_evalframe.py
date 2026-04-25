"""Tests for evalframe."""
import pytest
from evalframe import Evalframe
from evalframe.frame import BUILTIN_METRICS


def test_add_metric():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert "eq" in e.metrics()


def test_evaluate_pass():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert e.evaluate("hi", "hi")["eq"].passed is True


def test_evaluate_fail():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert e.evaluate("hi", "bye")["eq"].passed is False


def test_score():
    e = Evalframe()
    e.add_metric("contains", lambda p, r: r in p)
    assert e.score("answer is 42", "42")["contains"] is True


def test_builtin_exact_match():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.score("hello", "hello")["exact_match"] is True
    assert e.score("hello", "world")["exact_match"] is False


def test_builtin_contains():
    e = Evalframe()
    e.add_builtin("contains")
    assert e.score("the answer is 42", "42")["contains"] is True


def test_builtin_f1():
    e = Evalframe()
    e.add_builtin("f1")
    score = e.score("the cat sat on the mat", "the cat sat")["f1"]
    assert 0 < score <= 1.0


def test_builtin_rouge1():
    e = Evalframe()
    e.add_builtin("rouge1")
    score = e.score("the cat sat on the mat", "the cat sat")["rouge1"]
    assert score == 1.0


def test_include_builtins_flag():
    e = Evalframe(include_builtins=True)
    assert "exact_match" in e.metrics()
    assert "f1" in e.metrics()


def test_remove_metric():
    e = Evalframe()
    e.add_metric("m", lambda p, r: True)
    assert e.remove_metric("m") is True
    assert "m" not in e.metrics()


def test_batch_evaluate():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    results = e.batch_evaluate(pairs)
    assert len(results) == 3


def test_summary():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    results = e.batch_evaluate(pairs)
    s = e.summary(results)
    assert s["exact_match"]["pass_rate"] == pytest.approx(2/3, rel=1e-3)


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
