# evalframe

**Lightweight evaluation framework for LLM outputs.**

evalframe provides built-in metrics for evaluating large language model (LLM) outputs against reference data, a pluggable interface for registering custom metrics, and a graphical user interface for interactive exploration.

[![CI](https://github.com/vdeshmukh203/evalframe/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/evalframe/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)

---

## Installation

```bash
pip install evalframe
```

No runtime dependencies beyond the Python standard library.

---

## Quick start

```python
from evalframe import Evalframe

ef = Evalframe()
ef.add_builtin("exact_match")
ef.add_builtin("f1")
ef.add_builtin("rouge1")

result = ef.evaluate("The answer is 42", "42")
print(result["exact_match"].passed)   # False
print(result["f1"].score)             # 0.5
print(result["rouge1"].score)         # 1.0
```

---

## Built-in metrics

| Name            | Type    | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `exact_match`   | `bool`  | String equality after stripping whitespace       |
| `contains`      | `bool`  | Reference is a substring of prediction           |
| `prefix_match`  | `bool`  | Prediction starts with reference (stripped)      |
| `f1`            | `float` | Token-level F1 using multiset (SQuAD-style)      |
| `rouge1`        | `float` | ROUGE-1 recall using multiset intersection       |

All five metrics can be loaded at once with `Evalframe(include_builtins=True)`.

---

## API reference

### `Evalframe(include_builtins=False)`

Main evaluation class.

```python
ef = Evalframe()
ef = Evalframe(include_builtins=True)  # load all 5 built-in metrics
```

### `add_metric(name, fn)`

Register a custom metric.  `fn` must be callable with signature
`(prediction: str, reference: str) -> bool | float`.

```python
ef.add_metric("len_ratio", lambda pred, ref: len(pred) / max(len(ref), 1))
```

Raises `TypeError` if `fn` is not callable.

### `add_builtin(name)`

Add a built-in metric by name.  Raises `ValueError` for unrecognised names.

```python
ef.add_builtin("rouge1")
```

### `remove_metric(name) -> bool`

Remove a metric.  Returns `True` if the metric existed, `False` otherwise.

### `metrics() -> List[str]`

Return names of all registered metrics.

### `evaluate(prediction, reference) -> Dict[str, EvalResult]`

Run all metrics on one pair.  Each `EvalResult` has:

| Field        | Type           | Description                        |
|--------------|----------------|------------------------------------|
| `metric`     | `str`          | Metric name                        |
| `score`      | `bool\|float`  | Raw score returned by the metric   |
| `passed`     | `bool`         | `bool(score)` — truthy score check |
| `prediction` | `str`          | The prediction string              |
| `reference`  | `str`          | The reference string               |

### `score(prediction, reference) -> Dict[str, Any]`

Shorthand — returns just `{metric_name: raw_score}`.

### `batch_evaluate(pairs) -> List[Dict[str, EvalResult]]`

Evaluate a list of `(prediction, reference)` tuples.

```python
pairs = [("cat", "cat"), ("dog", "cat"), ("kitten", "cat")]
results = ef.batch_evaluate(pairs)
```

### `summary(results) -> Dict[str, Any]`

Aggregate a batch into per-metric statistics.

```python
s = ef.summary(results)
# s["f1"] == {"pass_rate": 0.3333, "avg_score": 0.4167, "n": 3}
```

### `assert_passes(prediction, reference, min_pass_rate=1.0) -> bool`

Return `True` if the fraction of passing metrics meets `min_pass_rate`.

```python
ef.assert_passes("The cat sat", "cat", min_pass_rate=0.5)  # True (contains passes)
```

---

## Batch example

```python
from evalframe import Evalframe

ef = Evalframe(include_builtins=True)

pairs = [
    ("The quick brown fox", "quick brown fox"),
    ("Hello world",         "Hello world"),
    ("foo bar baz",         "baz qux"),
]

results = ef.batch_evaluate(pairs)
summary = ef.summary(results)

for metric, stats in summary.items():
    print(f"{metric:15s}  pass_rate={stats['pass_rate']:.0%}  avg={stats['avg_score']}")
```

---

## Graphical user interface

evalframe ships with a Tkinter-based GUI for interactive evaluation:

```bash
evalframe-gui
```

The GUI provides:
- **Metrics panel** — toggle built-in metrics or register custom metrics via
  Python expressions.
- **Single Evaluation tab** — enter a prediction and reference, view per-metric
  scores and pass/fail status.
- **Batch Evaluation tab** — evaluate multiple prediction/reference pairs and
  inspect per-pair detail alongside aggregate summary statistics.

The GUI requires no additional dependencies; Tkinter is part of the Python
standard library.

---

## Custom metrics

```python
import re

ef = Evalframe()

# Lambda
ef.add_metric("has_number", lambda pred, ref: bool(re.search(r"\d", pred)))

# Named function
def jaccard(pred: str, ref: str) -> float:
    a, b = set(pred.lower().split()), set(ref.lower().split())
    return len(a & b) / len(a | b) if a | b else 0.0

ef.add_metric("jaccard", jaccard)

print(ef.score("answer is 42", "42"))
# {'has_number': True, 'jaccard': 0.5}
```

---

## Contributing

Bug reports and pull requests are welcome at
<https://github.com/vdeshmukh203/evalframe>.

---

## License

MIT © Vaibhav Deshmukh
