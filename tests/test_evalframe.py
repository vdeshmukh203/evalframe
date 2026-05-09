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


def test_add_metric_rejects_non_str_name():
    e = Evalframe()
    with pytest.raises(TypeError, match="must be a str"):
        e.add_metric(42, lambda p, r: True)  # type: ignore[arg-type]


def test_add_metric_rejects_empty_name():
    e = Evalframe()
    with pytest.raises(ValueError, match="must not be empty"):
        e.add_metric("", lambda p, r: True)


def test_add_metric_rejects_non_callable():
    e = Evalframe()
    with pytest.raises(TypeError, match="must be callable"):
        e.add_metric("bad", "not_a_function")  # type: ignore[arg-type]


def test_remove_metric_existing():
    e = Evalframe()
    e.add_metric("m", lambda p, r: True)
    assert e.remove_metric("m") is True
    assert "m" not in e.metrics()


def test_remove_metric_nonexistent():
    e = Evalframe()
    assert e.remove_metric("does_not_exist") is False


def test_add_builtin_unknown_raises():
    e = Evalframe()
    with pytest.raises(ValueError, match="Unknown built-in metric"):
        e.add_builtin("nonexistent")


def test_include_builtins_flag():
    e = Evalframe(include_builtins=True)
    for name in BUILTIN_METRICS:
        assert name in e.metrics()


# ---------------------------------------------------------------------------
# Single-pair evaluation
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
    result = e.evaluate("a", "a")["eq"]
    assert isinstance(result, EvalResult)
    assert result.prediction == "a"
    assert result.reference == "a"
    assert result.metric == "eq"


def test_evalresult_repr():
    er = EvalResult(metric="f1", score=0.75, passed=True, prediction="a", reference="b")
    assert "f1" in repr(er)
    assert "0.75" in repr(er)


def test_evaluate_faulty_metric_warns():
    e = Evalframe()
    e.add_metric("boom", lambda p, r: 1 / 0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = e.evaluate("x", "y")
    assert any(issubclass(warning.category, RuntimeWarning) for warning in w)
    assert result["boom"].score is None
    assert result["boom"].passed is False


def test_score():
    e = Evalframe()
    e.add_metric("contains", lambda p, r: r in p)
    assert e.score("answer is 42", "42")["contains"] is True


# ---------------------------------------------------------------------------
# Built-in metrics – correctness
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
    assert e.score("the answer is 42", "43")["contains"] is False


def test_builtin_prefix_match():
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("hello world", "hello")["prefix_match"] is True
    assert e.score("world hello", "hello")["prefix_match"] is False


def test_builtin_prefix_match_strips_whitespace():
    e = Evalframe()
    e.add_builtin("prefix_match")
    assert e.score("  hello world", "hello")["prefix_match"] is True


# F1 correctness --------------------------------------------------------

def test_builtin_f1_partial_overlap():
    e = Evalframe()
    e.add_builtin("f1")
    score = e.score("the cat sat on the mat", "the cat sat")["f1"]
    assert 0 < score <= 1.0


def test_f1_perfect():
    assert _f1_score("the cat", "the cat") == pytest.approx(1.0)


def test_f1_no_overlap():
    assert _f1_score("foo bar", "baz qux") == pytest.approx(0.0)


def test_f1_empty_prediction():
    assert _f1_score("", "foo") == pytest.approx(0.0)


def test_f1_empty_reference():
    assert _f1_score("foo", "") == pytest.approx(0.0)


def test_f1_multiset_duplicates():
    # "a a b" vs "a b": with set both have {a,b} → F1=1.0, but with Counter:
    # pred=[a,a,b], ref=[a,b], common=Counter({a:1,b:1})=2
    # precision=2/3, recall=2/2=1.0 → F1 = 2*(2/3)*1/(2/3+1) = 0.8
    score = _f1_score("a a b", "a b")
    assert score == pytest.approx(0.8, rel=1e-3)


# ROUGE-1 correctness ---------------------------------------------------

def test_builtin_rouge1_full_recall():
    e = Evalframe()
    e.add_builtin("rouge1")
    score = e.score("the cat sat on the mat", "the cat sat")["rouge1"]
    assert score == pytest.approx(1.0)


def test_rouge1_partial():
    assert 0 < _rouge1("the cat", "the cat sat") < 1.0


def test_rouge1_empty_reference():
    assert _rouge1("something", "") == pytest.approx(0.0)


def test_rouge1_multiset_duplicates():
    # ref="cat cat dog" (3 tokens), pred="cat dog" (2 tokens)
    # Counter intersection: cat→1, dog→1 → common=2
    # recall = 2/3
    score = _rouge1("cat dog", "cat cat dog")
    assert score == pytest.approx(2 / 3, rel=1e-3)


def test_rouge1_multiset_no_inflation():
    # Without multiset fix: "cat" in ["cat","cat","dog"] would count twice → 3/3=1.0
    # With fix it should be 2/3
    score = _rouge1("cat dog", "cat cat dog")
    assert score < 1.0


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def test_batch_evaluate_length():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    results = e.batch_evaluate(pairs)
    assert len(results) == 3


def test_batch_evaluate_correctness():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c")]
    results = e.batch_evaluate(pairs)
    assert results[0]["exact_match"].passed is True
    assert results[1]["exact_match"].passed is False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

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


def test_summary_avg_score_boolean():
    # Boolean scores (True/False) should be coerced to 1.0/0.0 for avg_score
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c")]
    s = e.summary(e.batch_evaluate(pairs))
    assert s["exact_match"]["avg_score"] == pytest.approx(0.5)


def test_summary_n_field():
    e = Evalframe()
    e.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    s = e.summary(e.batch_evaluate(pairs))
    assert s["exact_match"]["n"] == 3


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
    # exact_match fails, contains passes → 50 % pass rate
    assert e.assert_passes("the answer is 42", "42", min_pass_rate=0.5) is True
    assert e.assert_passes("the answer is 42", "42", min_pass_rate=1.0) is False


def test_assert_passes_no_metrics():
    e = Evalframe()
    # No metrics → vacuously true
    assert e.assert_passes("anything", "anything") is True


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

def test_public_exports():
    from evalframe import BUILTIN_METRICS, EvalResult, Evalframe, __version__
    assert isinstance(__version__, str)
    assert callable(Evalframe)
    assert isinstance(BUILTIN_METRICS, dict)
    assert EvalResult is not None
