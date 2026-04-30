"""Tests for evalframe."""
from __future__ import annotations

import csv
import os
import tempfile

import pytest

from evalframe import BUILTIN_METRICS, EvalResult, Evalframe
from evalframe.frame import _exact_match, _f1_score, _rouge1


# ---------------------------------------------------------------------------
# Metric registration
# ---------------------------------------------------------------------------

def test_add_metric():
    ev = Evalframe()
    ev.add_metric("eq", lambda p, r: p == r)
    assert "eq" in ev.metrics()


def test_add_builtin_valid():
    ev = Evalframe()
    ev.add_builtin("exact_match")
    assert "exact_match" in ev.metrics()


def test_add_builtin_invalid():
    ev = Evalframe()
    with pytest.raises(ValueError, match="Unknown built-in metric"):
        ev.add_builtin("nonexistent_metric")


def test_remove_metric_exists():
    ev = Evalframe()
    ev.add_metric("m", lambda p, r: True)
    assert ev.remove_metric("m") is True
    assert "m" not in ev.metrics()


def test_remove_metric_missing():
    ev = Evalframe()
    assert ev.remove_metric("ghost") is False


def test_include_builtins_flag():
    ev = Evalframe(include_builtins=True)
    assert set(ev.metrics()) == set(BUILTIN_METRICS)


def test_repr_evalframe():
    ev = Evalframe()
    ev.add_builtin("f1")
    assert "f1" in repr(ev)


# ---------------------------------------------------------------------------
# evaluate / score
# ---------------------------------------------------------------------------

def test_evaluate_pass():
    ev = Evalframe()
    ev.add_metric("eq", lambda p, r: p == r)
    assert ev.evaluate("hi", "hi")["eq"].passed is True


def test_evaluate_fail():
    ev = Evalframe()
    ev.add_metric("eq", lambda p, r: p == r)
    assert ev.evaluate("hi", "bye")["eq"].passed is False


def test_evaluate_returns_evalresult():
    ev = Evalframe()
    ev.add_metric("eq", lambda p, r: p == r)
    r = ev.evaluate("hi", "hi")["eq"]
    assert isinstance(r, EvalResult)
    assert r.metric == "eq"
    assert r.prediction == "hi"
    assert r.reference == "hi"


def test_evaluate_error_metric_warns(recwarn):
    ev = Evalframe()
    ev.add_metric("boom", lambda p, r: 1 / 0)
    r = ev.evaluate("a", "b")["boom"]
    assert r.score is None
    assert r.passed is False
    assert len(recwarn) == 1


def test_score():
    ev = Evalframe()
    ev.add_metric("contains", lambda p, r: r in p)
    assert ev.score("answer is 42", "42")["contains"] is True


def test_evaluate_no_metrics():
    ev = Evalframe()
    assert ev.evaluate("x", "y") == {}


# ---------------------------------------------------------------------------
# EvalResult helpers
# ---------------------------------------------------------------------------

def test_eval_result_to_dict():
    r = EvalResult(metric="f1", score=0.8, passed=True, prediction="a", reference="b")
    d = r.to_dict()
    assert d == {
        "metric": "f1",
        "score": 0.8,
        "passed": True,
        "prediction": "a",
        "reference": "b",
    }


def test_eval_result_repr():
    r = EvalResult(metric="f1", score=0.5, passed=True, prediction="a", reference="b")
    assert "PASS" in repr(r)
    r2 = EvalResult(metric="f1", score=0.0, passed=False, prediction="a", reference="b")
    assert "FAIL" in repr(r2)


# ---------------------------------------------------------------------------
# Built-in: exact_match
# ---------------------------------------------------------------------------

def test_exact_match_pass():
    ev = Evalframe()
    ev.add_builtin("exact_match")
    assert ev.score("hello", "hello")["exact_match"] is True


def test_exact_match_fail():
    ev = Evalframe()
    ev.add_builtin("exact_match")
    assert ev.score("hello", "world")["exact_match"] is False


def test_exact_match_strips_whitespace():
    assert _exact_match("  hi  ", "hi") is True


# ---------------------------------------------------------------------------
# Built-in: contains
# ---------------------------------------------------------------------------

def test_contains_pass():
    ev = Evalframe()
    ev.add_builtin("contains")
    assert ev.score("the answer is 42", "42")["contains"] is True


def test_contains_fail():
    ev = Evalframe()
    ev.add_builtin("contains")
    assert ev.score("no numbers here", "42")["contains"] is False


# ---------------------------------------------------------------------------
# Built-in: prefix_match
# ---------------------------------------------------------------------------

def test_prefix_match_pass():
    ev = Evalframe()
    ev.add_builtin("prefix_match")
    assert ev.score("hello world", "hello")["prefix_match"] is True


def test_prefix_match_fail():
    ev = Evalframe()
    ev.add_builtin("prefix_match")
    assert ev.score("world hello", "hello")["prefix_match"] is False


# ---------------------------------------------------------------------------
# Built-in: f1 — correctness
# ---------------------------------------------------------------------------

def test_f1_perfect():
    ev = Evalframe()
    ev.add_builtin("f1")
    assert ev.score("the cat sat", "the cat sat")["f1"] == pytest.approx(1.0)


def test_f1_zero():
    ev = Evalframe()
    ev.add_builtin("f1")
    assert ev.score("apple banana", "cat dog")["f1"] == 0.0


def test_f1_partial():
    score = _f1_score("the cat sat on the mat", "the cat sat")
    assert 0 < score <= 1.0


def test_f1_empty_pred():
    assert _f1_score("", "the cat") == 0.0


def test_f1_empty_ref():
    assert _f1_score("the cat", "") == 0.0


def test_f1_repeated_tokens_multiset():
    # "the the the" vs "the cat": only 1 "the" should match (clipped)
    # pred_counter = {the:3}, ref_counter = {the:1, cat:1}
    # clipped common = 1; precision = 1/3; recall = 1/2; F1 ≈ 0.4
    score = _f1_score("the the the", "the cat")
    assert score == pytest.approx(0.4, rel=1e-3)


def test_f1_repeated_tokens_set_vs_counter():
    # Set-based would yield precision=1/1=1.0 (only "the" in set),
    # recall=1/2=0.5, F1≈0.667.  Counter-based gives 0.4.
    score = _f1_score("the the the", "the cat")
    assert score < 0.5  # counter-based result; set-based would be ≥ 0.667


# ---------------------------------------------------------------------------
# Built-in: rouge1 — correctness
# ---------------------------------------------------------------------------

def test_rouge1_perfect():
    ev = Evalframe()
    ev.add_builtin("rouge1")
    assert ev.score("the cat sat on the mat", "the cat sat")["rouge1"] == pytest.approx(
        1.0
    )


def test_rouge1_zero():
    ev = Evalframe()
    ev.add_builtin("rouge1")
    assert ev.score("apple banana", "cat dog")["rouge1"] == 0.0


def test_rouge1_empty_ref():
    assert _rouge1("the cat", "") == 0.0


def test_rouge1_clipped_counts():
    # ref = "the the the" (3 tokens), pred = "the cat" (1 "the")
    # clipped overlap = min(1,3) = 1; recall = 1/3 ≈ 0.3333
    score = _rouge1("the cat", "the the the")
    assert score == pytest.approx(1 / 3, rel=1e-3)


def test_rouge1_unclipped_would_be_wrong():
    # Unclipped (old) implementation would give 3/3 = 1.0 — assert it doesn't
    score = _rouge1("the cat", "the the the")
    assert score < 0.5


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def test_batch_evaluate_length():
    ev = Evalframe()
    ev.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    assert len(ev.batch_evaluate(pairs)) == 3


def test_summary_pass_rate():
    ev = Evalframe()
    ev.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c"), ("d", "d")]
    results = ev.batch_evaluate(pairs)
    s = ev.summary(results)
    assert s["exact_match"]["pass_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert s["exact_match"]["n"] == 3


def test_summary_avg_score_f1():
    ev = Evalframe()
    ev.add_builtin("f1")
    pairs = [("the cat sat", "the cat sat"), ("apple", "banana")]
    results = ev.batch_evaluate(pairs)
    s = ev.summary(results)
    assert s["f1"]["avg_score"] is not None
    assert 0 < s["f1"]["avg_score"] <= 1.0


def test_summary_empty():
    ev = Evalframe()
    ev.add_builtin("exact_match")
    assert ev.summary([]) == {}


# ---------------------------------------------------------------------------
# assert_passes
# ---------------------------------------------------------------------------

def test_assert_passes_all():
    ev = Evalframe()
    ev.add_builtin("exact_match")
    ev.add_builtin("contains")
    assert ev.assert_passes("42", "42", min_pass_rate=1.0) is True


def test_assert_passes_partial():
    ev = Evalframe()
    ev.add_builtin("exact_match")
    ev.add_builtin("contains")
    assert ev.assert_passes("the answer is 42", "42", min_pass_rate=0.5) is True


def test_assert_passes_no_metrics():
    ev = Evalframe()
    assert ev.assert_passes("x", "y") is True


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def test_load_pairs_csv(tmp_path):
    csvfile = tmp_path / "pairs.csv"
    csvfile.write_text("prediction,reference\nhello,hello\nbye,hello\n", encoding="utf-8")
    pairs = Evalframe.load_pairs_csv(csvfile)
    assert pairs == [("hello", "hello"), ("bye", "hello")]


def test_load_pairs_csv_case_insensitive(tmp_path):
    csvfile = tmp_path / "pairs.csv"
    csvfile.write_text("Prediction,Reference\na,b\n", encoding="utf-8")
    pairs = Evalframe.load_pairs_csv(csvfile)
    assert pairs == [("a", "b")]


def test_load_pairs_csv_missing_column(tmp_path):
    csvfile = tmp_path / "bad.csv"
    csvfile.write_text("output,target\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction"):
        Evalframe.load_pairs_csv(csvfile)


def test_save_results_csv_roundtrip(tmp_path):
    ev = Evalframe()
    ev.add_builtin("exact_match")
    pairs = [("a", "a"), ("b", "c")]
    results = ev.batch_evaluate(pairs)
    outfile = tmp_path / "results.csv"
    Evalframe.save_results_csv(results, pairs, outfile)

    with open(outfile, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["prediction"] == "a"
    assert rows[0]["exact_match_pass"] == "1"
    assert rows[1]["exact_match_pass"] == "0"


def test_save_results_csv_length_mismatch(tmp_path):
    ev = Evalframe()
    ev.add_builtin("exact_match")
    results = ev.batch_evaluate([("a", "a")])
    with pytest.raises(ValueError, match="same length"):
        Evalframe.save_results_csv(results, [("a", "a"), ("b", "b")], tmp_path / "out.csv")


def test_save_results_csv_empty(tmp_path):
    outfile = tmp_path / "empty.csv"
    Evalframe.save_results_csv([], [], outfile)
    assert not outfile.exists()
