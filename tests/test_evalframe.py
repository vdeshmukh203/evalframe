"""Tests for evalframe."""
import json
import tempfile
from pathlib import Path

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


def test_add_builtin_unknown_raises():
    e = Evalframe()
    with pytest.raises(ValueError, match="Unknown built-in"):
        e.add_builtin("nonexistent")


def test_remove_metric():
    e = Evalframe()
    e.add_metric("m", lambda p, r: True)
    assert e.remove_metric("m") is True
    assert "m" not in e.metrics()


def test_remove_nonexistent_returns_false():
    e = Evalframe()
    assert e.remove_metric("ghost") is False


def test_include_builtins_flag():
    e = Evalframe(include_builtins=True)
    assert set(e.metrics()) == set(BUILTIN_METRICS)


def test_repr():
    e = Evalframe()
    e.add_builtin("f1")
    assert "f1" in repr(e)


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


def test_evaluate_metric_exception_returns_none_score():
    e = Evalframe()

    def boom(p, r):
        raise RuntimeError("oops")

    e.add_metric("boom", boom)
    with pytest.warns(UserWarning, match="boom"):
        result = e.evaluate("x", "y")
    assert result["boom"].score is None
    assert result["boom"].passed is False


def test_score():
    e = Evalframe()
    e.add_metric("contains", lambda p, r: r in p)
    assert e.score("answer is 42", "42")["contains"] is True


# ---------------------------------------------------------------------------
# Built-in: exact_match
# ---------------------------------------------------------------------------


def test_builtin_exact_match_true():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.score("hello", "hello")["exact_match"] is True


def test_builtin_exact_match_false():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.score("hello", "world")["exact_match"] is False


def test_builtin_exact_match_strips_whitespace():
    e = Evalframe()
    e.add_builtin("exact_match")
    assert e.score("  hello  ", "hello")["exact_match"] is True


# ---------------------------------------------------------------------------
# Built-in: contains
# ---------------------------------------------------------------------------


def test_builtin_contains_true():
    e = Evalframe()
    e.add_builtin("contains")
    assert e.score("the answer is 42", "42")["contains"] is True


def test_builtin_contains_false():
    e = Evalframe()
    e.add_builtin("contains")
    assert e.score("the answer is 42", "99")["contains"] is False


# ---------------------------------------------------------------------------
# Built-in: prefix_match
# ---------------------------------------------------------------------------


def test_builtin_prefix_match_true():
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("Paris is beautiful", "Paris")["prefix_match"] is True


def test_builtin_prefix_match_false():
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("beautiful Paris", "Paris")["prefix_match"] is False


# ---------------------------------------------------------------------------
# Built-in: f1 (Counter-based)
# ---------------------------------------------------------------------------


def test_builtin_f1_range():
    e = Evalframe()
    e.add_builtin("f1")
    score = e.score("the cat sat on the mat", "the cat sat")["f1"]
    assert 0 < score <= 1.0


def test_f1_perfect():
    assert _f1_score("a b c", "a b c") == pytest.approx(1.0)


def test_f1_zero_overlap():
    assert _f1_score("x y z", "a b c") == pytest.approx(0.0)


def test_f1_repeated_tokens_counter_based():
    """Set-based F1 would overcount; Counter-based gives the correct value."""
    # pred = "the the the"  (3× 'the')
    # ref  = "the cat dog"  (1× 'the', 1× 'cat', 1× 'dog')
    # common = min(3,1) = 1; precision = 1/3; recall = 1/3; F1 = 1/3
    score = _f1_score("the the the", "the cat dog")
    assert score == pytest.approx(1 / 3, rel=1e-3)


def test_f1_empty_pred():
    assert _f1_score("", "hello") == pytest.approx(0.0)


def test_f1_empty_ref():
    assert _f1_score("hello", "") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Built-in: rouge1 (Counter-based)
# ---------------------------------------------------------------------------


def test_builtin_rouge1_perfect_recall():
    e = Evalframe()
    e.add_builtin("rouge1")
    assert e.score("the cat sat on the mat", "the cat sat")["rouge1"] == pytest.approx(1.0)


def test_rouge1_repeated_tokens_counter_based():
    """'the the the' vs ref 'the the dog': common = min(3,2) = 2; recall = 2/3."""
    score = _rouge1("the the the", "the the dog")
    assert score == pytest.approx(2 / 3, rel=1e-3)


def test_rouge1_empty_ref():
    assert _rouge1("anything", "") == pytest.approx(0.0)


def test_rouge1_no_overlap():
    assert _rouge1("x y z", "a b c") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


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


def test_summary_empty_returns_empty():
    e = Evalframe(include_builtins=True)
    assert e.summary([]) == {}


def test_summary_avg_score_f1():
    e = Evalframe()
    e.add_builtin("f1")
    pairs = [("a b c", "a b c"), ("x y z", "a b c")]
    results = e.batch_evaluate(pairs)
    s = e.summary(results)
    assert s["f1"]["avg_score"] == pytest.approx(0.5, rel=1e-2)


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


def test_assert_passes_no_metrics():
    e = Evalframe()
    assert e.assert_passes("anything", "anything") is True


# ---------------------------------------------------------------------------
# EvalResult.to_dict
# ---------------------------------------------------------------------------


def test_evalresult_to_dict_keys():
    r = EvalResult(
        metric="f1", score=0.75, passed=True,
        prediction="hello world", reference="hello",
    )
    d = r.to_dict()
    assert set(d) == {"metric", "score", "passed", "prediction", "reference"}


def test_evalresult_to_dict_values():
    r = EvalResult(
        metric="exact_match", score=True, passed=True,
        prediction="hi", reference="hi",
    )
    d = r.to_dict()
    assert d["score"] is True
    assert d["passed"] is True


# ---------------------------------------------------------------------------
# Persistence: save_results / load_results
# ---------------------------------------------------------------------------


def test_save_load_single(tmp_path):
    e = Evalframe()
    e.add_builtin("f1")
    results = e.evaluate("the cat", "the cat sat")
    path = tmp_path / "single.json"
    Evalframe.save_results(results, path)
    loaded = Evalframe.load_results(path)
    assert isinstance(loaded, dict)
    assert "f1" in loaded
    assert abs(loaded["f1"].score - results["f1"].score) < 1e-6


def test_save_load_batch(tmp_path):
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c")]
    results = e.batch_evaluate(pairs)
    path = tmp_path / "batch.json"
    Evalframe.save_results(results, path)
    loaded = Evalframe.load_results(path)
    assert isinstance(loaded, list)
    assert len(loaded) == 2
    assert loaded[0]["exact_match"].passed is True
    assert loaded[1]["exact_match"].passed is False


def test_save_results_valid_json(tmp_path):
    e = Evalframe(include_builtins=True)
    results = e.evaluate("hello world", "hello")
    path = tmp_path / "out.json"
    Evalframe.save_results(results, path)
    with open(path) as fh:
        data = json.load(fh)
    assert "f1" in data
    assert "exact_match" in data


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------


def test_public_exports():
    from evalframe import BUILTIN_METRICS, EvalResult, Evalframe  # noqa: F401
    assert callable(Evalframe)
    assert isinstance(BUILTIN_METRICS, dict)
