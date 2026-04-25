"""Tests for evalframe."""
from evalframe import Evalframe

def test_add_metric():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    assert "eq" in e.metrics()

def test_evaluate_pass():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    results = e.evaluate("hello", "hello")
    assert results["eq"].passed is True

def test_evaluate_fail():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    results = e.evaluate("hello", "world")
    assert results["eq"].passed is False

def test_score():
    e = Evalframe()
    e.add_metric("contains", lambda p, r: r in p)
    scores = e.score("the answer is 42", "42")
    assert scores["contains"] is True

def test_multiple_metrics():
    e = Evalframe()
    e.add_metric("eq", lambda p, r: p == r)
    e.add_metric("contains", lambda p, r: r in p)
    results = e.evaluate("answer: 42", "42")
    assert len(results) == 2
    assert results["contains"].passed is True
    assert results["eq"].passed is False
