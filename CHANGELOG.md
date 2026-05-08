# Changelog

All notable changes to evalframe are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-08

### Added
- Interactive web GUI (`evalframe-gui` entry-point) served via Python's stdlib
  `http.server`; supports single-pair evaluation, batch evaluation with summary
  statistics, and custom lambda metrics — no browser-side dependencies required.
- `EvalResult.error` field: records the exception message when a metric function
  raises, instead of silently discarding it.
- `BUILTIN_METRICS` and `EvalResult` are now exported from the top-level
  `evalframe` package (`from evalframe import EvalResult, BUILTIN_METRICS`).
- `pyproject.toml`: added PyPI classifiers, keywords, author metadata, and the
  `evalframe-gui` script entry-point.

### Fixed
- **`_f1_score`**: replaced `set`-based token intersection with
  `collections.Counter`-based intersection, correctly handling repeated tokens
  (e.g. `_f1_score("cat cat cat", "cat")` now returns ≈ 0.5 instead of 1.0).
- **`_rouge1`**: same Counter fix; reference tokens are now clipped to their
  count in the prediction before computing recall (standard ROUGE clipping).
- **`evaluate()`**: a metric that raises an exception now logs a `WARNING` via
  the `logging` module and stores the message in `EvalResult.error`, rather
  than swallowing the error silently.
- **`evaluate()`**: raises `TypeError` with a descriptive message when
  `prediction` or `reference` is not a `str`.
- **`add_metric()`**: raises `ValueError` for empty/whitespace names and
  `TypeError` when the second argument is not callable.
- **`add_builtin()`**: error message now lists available metric names in sorted
  order for easier discoverability.
- **`assert_passes()`**: issues a `UserWarning` when called with no metrics
  registered (instead of returning `True` silently).

## [0.1.0] - 2026-04-25

### Added
- Initial release of evalframe.
- Built-in metrics for exact match, token-level F1, and ROUGE-1.
- Pluggable custom-metric registration.
- Structured result objects suited to aggregation and comparison.
