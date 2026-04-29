---
title: 'evalframe: a lightweight evaluation framework for large language model outputs'
tags:
  - Python
  - large language models
  - evaluation
  - natural language processing
  - metrics
authors:
  - name: Vaibhav Deshmukh
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 25 April 2026
bibliography: paper.bib
---

# Summary

`evalframe` is a Python library for evaluating large language model (LLM)
[@brown2020language] outputs against reference data. It ships with five
built-in metrics—exact match, substring containment, prefix match,
token-level F1 (following the SQuAD convention [@rajpurkar2016squad]),
and ROUGE-1 unigram recall [@lin2004rouge]—and exposes a minimal extension
surface for registering arbitrary custom metrics. Each evaluation run
produces a structured `EvalResult` object that records the raw score,
a boolean pass/fail flag, and the original strings, making results easy to
aggregate, filter, or diff across model versions. A Gradio-based
interactive GUI is included as an optional add-on for rapid, notebook-free
experimentation.

# Statement of need

Reproducible, lightweight evaluation is a persistent pain point in LLM
development. At one end of the spectrum, practitioners reach for ad-hoc
one-liners (`pred == ref`) that are quick to write but impossible to track
systematically. At the other end, comprehensive platforms such as
`lm-evaluation-harness` [@gao2021framework] and the Hugging Face `evaluate`
library [@von-werra-etal-2022-evaluate] offer broad benchmark coverage but
introduce non-trivial installation footprints and configuration overhead
that are disproportionate for prompt-iteration workflows.

`evalframe` targets the gap between these two extremes. It is designed for
practitioners who need a transparent, auditable evaluation loop while
iterating on prompts, fine-tunes, or model choices, with the following
design priorities:

1. **Zero mandatory runtime dependencies.** The core library is pure Python
   (≥ 3.9) and adds nothing to `sys.path` beyond the standard library.
2. **A stable, typed API.** All public methods carry full type annotations
   and docstrings, making them discoverable in editors and notebooks.
3. **Correct standard-algorithm implementations.** The token-level F1 score
   uses a `Counter`-based multiset intersection (the SQuAD convention), and
   ROUGE-1 applies per-token count clipping, matching the behaviour
   described in the original papers [@rajpurkar2016squad; @lin2004rouge].
4. **Structured results.** Scores are returned as `EvalResult` dataclasses
   rather than bare scalars, so downstream code can inspect both the numeric
   value and the pass/fail outcome without reimplementing thresholding logic.
5. **An optional interactive GUI.** Installing `evalframe[gui]` provides a
   Gradio [@abid2019gradio] web interface for single-pair and batch
   evaluation without writing any code, which is useful for demonstrations
   and for non-programmers collaborating on prompt design.

# Implementation

`evalframe` centres on the `Evalframe` class. Metrics are registered via
`add_builtin(name)` or `add_metric(name, fn)`, where `fn` is any callable
that accepts two strings and returns a value whose truthiness indicates a
pass. Evaluation is performed by `evaluate(prediction, reference)`, which
returns a dictionary of `EvalResult` objects, one per registered metric. If
a metric function raises an exception the failure is captured, a
`RuntimeWarning` is emitted, and evaluation continues—ensuring that a
misbehaving custom metric does not abort an entire batch run.

Batch workflows are supported through `batch_evaluate(pairs)` and
`summary(results)`. The summary aggregates pass rates and average scores
across a list of `(prediction, reference)` pairs and is suitable for
reporting in notebooks or CI pipelines. An `assert_passes` helper raises a
`ValueError` when invoked without any registered metrics (an unambiguous
configuration error) and otherwise returns a boolean indicating whether the
fraction of passing metrics meets a caller-supplied threshold.

The built-in metric implementations follow their canonical reference
definitions:

- **Exact match** strips leading and trailing whitespace before comparing.
- **Token-level F1** tokenises by whitespace, lower-cases both strings,
  counts tokens with `collections.Counter`, clips the intersection, and
  computes the harmonic mean of precision and recall—identical to the SQuAD
  evaluation script [@rajpurkar2016squad].
- **ROUGE-1** applies the same Counter-based clipped intersection to compute
  unigram recall, consistent with the original ROUGE implementation
  [@lin2004rouge].
- **Contains** and **prefix match** guard against vacuous matches by
  returning `False` when the reference string is empty.

# Acknowledgements

The author thanks the open-source community whose tooling made this project
possible, and the developers of Gradio, pytest, and hatchling whose
libraries underpin the GUI, test suite, and packaging respectively.

# References
