# Changelog

All notable changes to evalframe are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-29

### Added
- Interactive Gradio GUI (`evalframe[gui]`, `evalframe-gui` CLI command) with
  single-pair and batch-evaluation tabs.
- `evalframe-gui` console-script entry point.
- `[tool.pytest.ini_options]` with `pythonpath = ["src"]` so tests run without
  a manual `PYTHONPATH` export.
- New tests covering: metric errors emitting `RuntimeWarning`, empty-string
  edge cases for `contains` / `prefix_match`, Counter-based token handling for
  `f1` and `rouge1`, and `assert_passes` raising `ValueError` when no metrics
  are registered (33 tests total, up from 18).
- Expanded JOSS paper with Implementation section, SQuAD and Gradio citations,
  and comparison with `lm-evaluation-harness` and Hugging Face `evaluate`.

### Fixed
- **`_f1_score`**: switched from `set`-based to `Counter`-based intersection
  (SQuAD convention). Repeated tokens are now weighted correctly; e.g.
  `f1("the the the", "the")` returns `0.5` instead of the incorrect `1.0`.
- **`_rouge1`**: switched to `Counter` with clipped counts. Repeated prediction
  tokens no longer inflate recall.
- **`_contains`**: now returns `False` when the reference is the empty string,
  preventing vacuous matches.
- **`_prefix_match`**: now returns `False` when the stripped reference is empty.
- **`assert_passes`**: raises `ValueError` (instead of silently returning
  `True`) when no metrics are registered.
- **`evaluate`**: metric exceptions now emit a `RuntimeWarning` with the metric
  name and exception details, rather than being silently swallowed.

### Changed
- All public methods now have complete docstrings (Args / Returns / Raises).
- `add_builtin` error message lists available metrics in sorted order.

## [0.1.0] - 2026-04-25

### Added
- Initial release of evalframe.
- Built-in metrics for exact match, token-level F1, and ROUGE-1.
- Pluggable custom-metric registration.
- Structured result objects suited to aggregation and comparison.
