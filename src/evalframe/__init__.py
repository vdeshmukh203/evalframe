"""evalframe — Lightweight LLM evaluation framework."""
from .frame import BUILTIN_METRICS, EvalResult, Evalframe

__all__ = ["Evalframe", "EvalResult", "BUILTIN_METRICS"]
__version__ = "0.1.0"
