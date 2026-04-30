# evalframe

Lightweight evaluation framework for LLM outputs.

## Installation

```bash
pip install evalframe
```

## Quick start

```python
from evalframe import Evalframe

ev = Evalframe()
ev.add_builtin("exact_match")
ev.add_builtin("f1")
ev.add_builtin("rouge1")

# Single pair
results = ev.evaluate("The answer is 42", "42")
for name, r in results.items():
    print(f"{name}: score={r.score}  passed={r.passed}")
# exact_match: score=False  passed=False
# f1:          score=0.5    passed=True
# rouge1:      score=1.0    passed=True

# Custom metric
ev.add_metric("length_ok", lambda pred, ref: abs(len(pred) - len(ref)) <= 5)

# Batch evaluation
pairs = [
    ("Paris", "Paris"),
    ("London", "Paris"),
    ("The capital is Paris", "Paris"),
]
results = ev.batch_evaluate(pairs)
print(ev.summary(results))
# {'exact_match': {'pass_rate': 0.3333, 'avg_score': 0.3333, 'n': 3}, ...}
```

## Built-in metrics

| Name          | Returns | Description                                       |
|---------------|---------|---------------------------------------------------|
| `exact_match` | `bool`  | Stripped string equality                          |
| `contains`    | `bool`  | Prediction contains reference as a substring      |
| `prefix_match`| `bool`  | Prediction starts with reference                  |
| `f1`          | `float` | Token-level F1 (bag-of-words, SQuAD-style)        |
| `rouge1`      | `float` | ROUGE-1 recall with clipped unigram counts        |

## API reference

### `Evalframe`

| Method | Description |
|--------|-------------|
| `add_builtin(name)` | Register a built-in metric by name |
| `add_metric(name, fn)` | Register a custom `(pred, ref) → score` function |
| `remove_metric(name)` | Deregister a metric; returns `True` if it existed |
| `metrics()` | List currently registered metric names |
| `evaluate(pred, ref)` | Run all metrics; returns `Dict[str, EvalResult]` |
| `score(pred, ref)` | Flat `Dict[str, score]` convenience wrapper |
| `batch_evaluate(pairs)` | Evaluate a list of `(pred, ref)` tuples |
| `summary(results)` | Aggregate pass rates and mean scores |
| `assert_passes(pred, ref, min_pass_rate)` | Returns `True` when enough metrics pass |
| `load_pairs_csv(path)` | Load `(prediction, reference)` pairs from CSV |
| `save_results_csv(results, pairs, path)` | Write batch results to CSV |

### `EvalResult`

Each entry in the dict returned by `evaluate()` is an `EvalResult` dataclass:

```python
@dataclass
class EvalResult:
    metric: str
    score: Any       # raw metric value
    passed: bool     # bool(score)
    prediction: str
    reference: str

    def to_dict(self) -> dict: ...
```

## GUI

A desktop GUI ships with the package:

```bash
evalframe-gui          # CLI entry point after pip install
```

Or from Python:

```python
from evalframe.gui import launch_gui
launch_gui()
```

The GUI provides two tabs:

- **Single Evaluation** — enter a prediction/reference pair, select metrics,
  and view a pass/fail report.
- **Batch Evaluation** — load a CSV file, run metrics across all rows, view
  an interactive results table and summary statistics, and export to CSV.

## CSV format

`load_pairs_csv` expects a file with at minimum `prediction` and `reference`
columns (case-insensitive):

```csv
prediction,reference
"Paris","Paris"
"The capital is Paris","Paris"
```

## Contributing

Bug reports and pull requests are welcome at
<https://github.com/vdeshmukh203/evalframe>.

## License

MIT
