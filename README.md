# evalframe

[![Tests](https://github.com/vdeshmukh203/evalframe/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/evalframe/actions)

Lightweight Python framework for evaluating large language model (LLM) outputs
against reference data.  It ships with five built-in metrics, a small extension
surface for custom metrics, batch evaluation with summary statistics, JSON
persistence, and an optional desktop GUI.

---

## Installation

```bash
pip install evalframe
```

Python ≥ 3.9 is required.  No third-party runtime dependencies.

---

## Quick start

```python
from evalframe import Evalframe

ef = Evalframe()
ef.add_builtin("f1")
ef.add_builtin("exact_match")

# Single pair
results = ef.evaluate("The answer is 42", "42")
print(results["f1"].score)   # 0.4
print(results["f1"].passed)  # True  (any positive overlap passes)

# Scores only
print(ef.score("Paris", "Paris"))
# {'f1': 1.0, 'exact_match': True}

# Batch
pairs = [
    ("Paris", "Paris"),
    ("Berlin is cold", "Paris"),
    ("the cat sat on the mat", "the cat sat"),
]
batch = ef.batch_evaluate(pairs)
print(ef.summary(batch))
# {'f1': {'pass_rate': 0.6667, 'avg_score': 0.6857, 'n': 3},
#  'exact_match': {'pass_rate': 0.3333, 'avg_score': 0.3333, 'n': 3}}
```

---

## Built-in metrics

| Name           | Type    | Description                                      |
|----------------|---------|--------------------------------------------------|
| `exact_match`  | `bool`  | Prediction equals reference after stripping whitespace |
| `contains`     | `bool`  | Reference appears verbatim inside prediction     |
| `prefix_match` | `bool`  | Prediction starts with reference (after stripping) |
| `f1`           | `float` | Token-level F1 (SQuAD protocol, Counter-based)   |
| `rouge1`       | `float` | ROUGE-1 recall (clipped unigram overlap, Lin 2004) |

Enable all at once:

```python
ef = Evalframe(include_builtins=True)
```

Or selectively:

```python
ef = Evalframe()
ef.add_builtin("f1")
ef.add_builtin("rouge1")
```

---

## Custom metrics

```python
import re

ef = Evalframe()
ef.add_metric(
    "number_match",
    lambda pred, ref: re.search(r"\d+", pred) is not None
                      and re.search(r"\d+", pred).group() == ref.strip(),
)
```

Any callable `(prediction: str, reference: str) -> bool | float` is accepted.

---

## Persistence

```python
# Save
results = ef.batch_evaluate(pairs)
Evalframe.save_results(results, "results.json")

# Load
loaded = Evalframe.load_results("results.json")
```

`EvalResult.to_dict()` also produces a JSON-serialisable dictionary for
integration with custom pipelines.

---

## Assertion helper

```python
# Returns True when ≥ 50 % of metrics pass
ef.assert_passes("the answer is 42", "42", min_pass_rate=0.5)
```

---

## Desktop GUI

A Tkinter-based desktop application is included for interactive exploration.

```bash
evalframe-gui          # after pip install
# or
python -m evalframe.gui
```

The GUI provides:

- **Sidebar** — toggle built-in metrics on/off; add custom lambda expressions;
  remove any registered metric.
- **Single tab** — enter a prediction/reference pair, evaluate all active
  metrics, see pass/fail results colour-coded in a table, and export to JSON.
- **Batch tab** — paste tab- or comma-separated pairs (or load a CSV/TSV
  file), run batch evaluation, and inspect a summary table with pass rates and
  mean scores.

> **Note**: Tkinter is part of Python's standard library.  On some Linux
> distributions you may need to install `python3-tk` separately
> (`sudo apt install python3-tk`).

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE).
