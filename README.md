# evalframe

Lightweight evaluation framework for large language model (LLM) outputs.

```
pip install evalframe          # core library (no extra dependencies)
pip install 'evalframe[gui]'   # + Gradio interactive GUI
```

## Quick start

```python
from evalframe import Evalframe

ef = Evalframe()
ef.add_builtin("exact_match")
ef.add_builtin("f1")

result = ef.evaluate("The answer is 42", "42")
print(result["exact_match"].passed)  # False
print(result["f1"].score)            # 0.5
```

## Built-in metrics

| Name | Description |
|------|-------------|
| `exact_match` | Strips whitespace, then compares strings literally. |
| `contains` | Reference is a non-empty substring of the prediction. |
| `prefix_match` | Prediction starts with the reference (both stripped). |
| `f1` | Token-level F1, SQuAD convention (Counter-based, handles repeated tokens). |
| `rouge1` | ROUGE-1 unigram recall with clipped token counts. |

## Custom metrics

```python
def length_ratio(pred: str, ref: str) -> float:
    return min(len(pred), len(ref)) / max(len(pred), len(ref), 1)

ef.add_metric("length_ratio", length_ratio)
```

Any callable `(str, str) -> Any` is accepted; a truthy return value counts as a pass.

## Batch evaluation

```python
pairs = [
    ("Paris", "Paris"),
    ("London", "Paris"),
    ("The capital is Paris", "Paris"),
]

ef = Evalframe(include_builtins=True)
results = ef.batch_evaluate(pairs)
summary = ef.summary(results)
# {"exact_match": {"pass_rate": 0.3333, "avg_score": 0.3333, "n": 3}, ...}
```

## Interactive GUI

```bash
evalframe-gui          # opens http://localhost:7860 in your browser
```

The GUI provides two tabs:

- **Single Evaluation** — enter one prediction/reference pair and inspect per-metric scores.
- **Batch Evaluation** — paste tab-separated `prediction⇥reference` pairs and view a summary table plus per-pair detail.

## API reference

### `Evalframe(include_builtins=False)`

| Method | Description |
|--------|-------------|
| `add_builtin(name)` | Register a built-in metric by name. Raises `ValueError` for unknown names. |
| `add_metric(name, fn)` | Register a custom metric callable. |
| `remove_metric(name)` | Unregister a metric; returns `True` on success. |
| `metrics()` | List names of all registered metrics. |
| `evaluate(prediction, reference)` | Run all metrics; returns `Dict[str, EvalResult]`. |
| `score(prediction, reference)` | Return raw scores as `Dict[str, Any]`. |
| `batch_evaluate(pairs)` | Evaluate a list of `(prediction, reference)` tuples. |
| `summary(results)` | Aggregate pass rates and average scores across a batch. |
| `assert_passes(prediction, reference, min_pass_rate=1.0)` | Return `True` if pass fraction ≥ threshold. Raises `ValueError` when no metrics are registered. |

### `EvalResult`

```python
@dataclass
class EvalResult:
    metric: str      # metric name
    score: Any       # raw score (None if the metric raised an exception)
    passed: bool     # True when score is truthy
    prediction: str
    reference: str
```

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

## Citation

If you use evalframe in academic work, please cite:

```bibtex
@software{deshmukh2026evalframe,
  author  = {Deshmukh, Vaibhav},
  title   = {evalframe: a lightweight evaluation framework for large language model outputs},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/vdeshmukh203/evalframe}
}
```

## License

MIT © Vaibhav Deshmukh
