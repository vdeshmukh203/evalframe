# evalframe

Lightweight evaluation framework for LLM outputs.

```python
from evalframe import Evalframe

eval = Evalframe()
eval.add_metric("exact_match", lambda pred, ref: pred.strip() == ref.strip())
eval.add_metric("contains", lambda pred, ref: ref in pred)

results = eval.evaluate("The answer is 42", "42")
print(results)  # {"exact_match": False, "contains": True}
```
