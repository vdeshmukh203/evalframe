# Changelog

All notable changes to evalframe are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-30

### Fixed
- **F1 score**: replaced set-based token overlap with `Counter` (multiset)
  intersection so repeated tokens are counted correctly, matching the standard
  SQuAD evaluation metric.
- **ROUGE-1**: replaced unclipped list membership check with clipped
  `Counter` intersection, preventing artificially high recall when reference
  tokens are repeated.

### Added
- `EvalResult.to_dict()` — plain-dict serialisation for JSON/CSV output.
- `EvalResult.__repr__()` — readable string representation including pass/fail status.
- `Evalframe.__repr__()` — shows currently registered metric names.
- `Evalframe.load_pairs_csv(path)` — load `(prediction, reference)` pairs
  from a CSV file.
- `Evalframe.save_results_csv(results, pairs, path)` — export batch results
  to CSV.
- Desktop GUI (`src/evalframe/gui.py`) with two tabs: *Single Evaluation*
  and *Batch Evaluation* (file load, results table, summary, CSV export).
- `evalframe-gui` console script entry point.
- `[project.optional-dependencies] dev` in `pyproject.toml` (pytest ≥ 7, pytest-cov).
- `BUILTIN_METRICS` exported from the top-level package.

### Improved
- Full NumPy-style docstrings on all public classes and methods.
- `warnings.warn` instead of silent pass when a metric function raises an
  exception during `evaluate()`.
- `summary()` gracefully handles non-numeric (e.g. pure boolean) scores.

## [0.1.0] - 2026-04-25

### Added
- Initial release of evalframe.
- Built-in metrics for exact match, token-level F1, and ROUGE-1.
- Pluggable custom-metric registration.
- Structured result objects suited to aggregation and comparison.
