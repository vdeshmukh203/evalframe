# evalframe

Lightweight evaluation framework for LLM outputs — no runtime dependencies, drops straight into notebooks and unit tests.

## Installation

```bash
pip install evalframe
```

## Quick start

```python
from evalframe import Evalframe

ef = Evalframe()
ef.add_builtin("exact_match")
ef.add_builtin("f1")
ef.add_builtin("rouge1")

result = ef.evaluate("The answer is 42", "42")
print(result["exact_match"].passed)  # False
print(result["f1"].score)            # float, token-level F1
```

## Built-in metrics

| Name | Description |
|---|---|
| `exact_match` | Stripped string equality |
| `contains` | Reference appears verbatim in prediction |
| `prefix_match` | Prediction starts with reference |
| `f1` | Token-level F1 (word-set overlap) |
| `rouge1` | ROUGE-1 recall (unigram overlap) |

Enable all at once:

```python
ef = Evalframe(include_builtins=True)
```

## Custom metrics

Any callable `(prediction: str, reference: str) -> Any` can be registered:

```python
ef.add_metric("length_ok", lambda pred, ref: abs(len(pred) - len(ref)) < 10)
```

## Batch evaluation

```python
pairs = [
    ("Paris", "Paris"),
    ("London", "Paris"),
    ("The capital is Paris.", "Paris"),
]
results = ef.batch_evaluate(pairs)
summary = ef.summary(results)
print(summary["exact_match"]["pass_rate"])  # 0.3333
```

Export to CSV:

```python
csv_text = ef.to_csv(results)
with open("eval_results.csv", "w") as f:
    f.write(csv_text)
```

## Assertion helper

Useful in unit tests for prompt regressions:

```python
assert ef.assert_passes("42", "42", min_pass_rate=1.0)
assert ef.assert_passes("The answer is 42", "42", min_pass_rate=0.5)
```

## GUI

A built-in graphical interface (Tkinter, no extra dependencies) is available for
interactive exploration and batch evaluation from a CSV file.

```bash
evalframe-gui          # console-script entry point
python -m evalframe.gui
```

Or launch from Python:

```python
from evalframe import launch_gui
launch_gui()
```

The GUI provides:

- **Single Pair** tab — enter prediction and reference, toggle built-in metrics,
  add custom lambda expressions, and view per-metric pass/fail results.
- **Batch Evaluation** tab — paste or load a CSV with `prediction,reference`
  columns, run all selected metrics, inspect the summary table, and export
  detailed results to CSV.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
