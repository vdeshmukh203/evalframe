# Changelog

All notable changes to evalframe are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-01

### Added
- Tkinter-based graphical user interface (`evalframe-gui` entry point) with
  single-evaluation, batch-evaluation, and summary statistics panels.
- `evalframe-gui` console script registered in `pyproject.toml`.
- `EvalResult` exported from the top-level `evalframe` package.
- Project classifiers and keywords in `pyproject.toml`.
- SQuAD reference (`rajpurkar2016squad`) added to `paper.bib`.

### Fixed
- `_f1_score`: replaced `set`-based token intersection with `Counter`
  (multiset) intersection to correctly handle repeated tokens per the SQuAD
  evaluation protocol.
- `_rouge1`: replaced list-membership overlap with `Counter` intersection so
  that repeated reference tokens are not over-counted.
- `evaluate()`: metric exceptions now emit a `RuntimeWarning` instead of being
  silently swallowed, making failures visible during development.

### Changed
- `add_metric()` now raises `TypeError` if the supplied function is not callable.
- `add_builtin()` error message now lists available names in sorted order.
- `summary()` safely converts scores to `float` before averaging, handling
  edge cases where non-numeric scores are present.
- All public methods have expanded NumPy-style docstrings.
- `paper.md` updated to mention all five built-in metrics and the GUI.
- Version bumped to `0.2.0`.

## [0.1.0] - 2026-04-25

### Added
- Initial release of evalframe.
- Built-in metrics for exact match, token-level F1, and ROUGE-1.
- Pluggable custom-metric registration.
- Structured result objects suited to aggregation and comparison.
