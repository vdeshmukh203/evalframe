"""Gradio-based interactive GUI for evalframe.

Launch from the command line::

    evalframe-gui

or from Python::

    from evalframe.app import launch
    launch()
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError(
        "The evalframe GUI requires Gradio. "
        "Install it with:  pip install 'evalframe[gui]'"
    ) from exc

from evalframe.frame import BUILTIN_METRICS, Evalframe

_METRIC_CHOICES: List[str] = list(BUILTIN_METRICS)

_METRIC_DESCRIPTIONS: dict[str, str] = {
    "exact_match": "Exact string match after stripping whitespace.",
    "contains": "Reference is a non-empty substring of the prediction.",
    "prefix_match": "Prediction starts with the reference (both stripped).",
    "f1": "Token-level F1 using the SQuAD convention (Counter-based, supports repeated tokens).",
    "rouge1": "ROUGE-1 unigram recall with clipped token counts.",
}

_BATCH_PLACEHOLDER = (
    "The answer is 42\t42\n"
    "Paris is the capital of France\tParis\n"
    "The sky is blue\tblue"
)


def _make_evalframe(selected: List[str]) -> Evalframe:
    ef = Evalframe()
    for m in selected:
        ef.add_builtin(m)
    return ef


def _evaluate_single(
    prediction: str,
    reference: str,
    selected_metrics: List[str],
) -> Tuple[Optional[List[list]], str]:
    if not selected_metrics:
        return None, "Please select at least one metric."
    if not prediction.strip():
        return None, "Prediction must not be empty."
    if not reference.strip():
        return None, "Reference must not be empty."

    ef = _make_evalframe(selected_metrics)
    results = ef.evaluate(prediction, reference)
    rows = [
        [r.metric, r.score, "✓" if r.passed else "✗"]
        for r in results.values()
    ]
    return rows, ""


def _evaluate_batch(
    text_input: str,
    selected_metrics: List[str],
) -> Tuple[Optional[List[list]], Optional[List[list]], str]:
    if not selected_metrics:
        return None, None, "Please select at least one metric."

    pairs: List[Tuple[str, str]] = []
    errors: List[str] = []

    for i, line in enumerate(text_input.strip().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            errors.append(
                f"Line {i}: expected 2 tab-separated columns, got {len(parts)}."
            )
        else:
            pairs.append((parts[0].strip(), parts[1].strip()))

    if errors:
        return None, None, "\n".join(errors)
    if not pairs:
        return None, None, (
            "No valid pairs found. "
            "Enter one pair per line as  prediction<TAB>reference."
        )

    ef = _make_evalframe(selected_metrics)
    batch_results = ef.batch_evaluate(pairs)
    summary = ef.summary(batch_results)

    summary_rows = [
        [
            m,
            f"{s['pass_rate']:.1%}",
            f"{s['avg_score']:.4f}" if s["avg_score"] is not None else "—",
            s["n"],
        ]
        for m, s in summary.items()
    ]

    detail_rows: List[list] = []
    for idx, (pair_results, (pred, ref)) in enumerate(
        zip(batch_results, pairs), start=1
    ):
        for mname, r in pair_results.items():
            detail_rows.append(
                [
                    idx,
                    pred[:60] + ("…" if len(pred) > 60 else ""),
                    ref[:60] + ("…" if len(ref) > 60 else ""),
                    mname,
                    r.score,
                    "✓" if r.passed else "✗",
                ]
            )

    return summary_rows, detail_rows, ""


def launch(
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "127.0.0.1",
) -> None:
    """Launch the evalframe interactive GUI in a web browser.

    Args:
        share: If ``True``, create a public Gradio share link.
        server_port: Port for the local server (default ``7860``).
        server_name: Host/IP to bind to (default ``"127.0.0.1"``).
    """
    metric_info_md = "\n".join(
        f"- **`{k}`** — {v}" for k, v in _METRIC_DESCRIPTIONS.items()
    )

    with gr.Blocks(
        title="evalframe — LLM Evaluation Framework",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """# evalframe
**Lightweight evaluation framework for large language model outputs.**

Select metrics, enter predictions and references, then click **Evaluate**."""
        )

        with gr.Accordion("Available metrics", open=False):
            gr.Markdown(metric_info_md)

        with gr.Tabs():
            # ── Tab 1: Single Evaluation ─────────────────────────────────
            with gr.TabItem("Single Evaluation"):
                gr.Markdown(
                    "Run all selected metrics on a single "
                    "(prediction, reference) pair."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        single_metrics = gr.CheckboxGroup(
                            choices=_METRIC_CHOICES,
                            value=["exact_match", "f1", "rouge1"],
                            label="Metrics",
                        )
                        single_pred = gr.Textbox(
                            label="Prediction",
                            placeholder="Enter the model's output…",
                            lines=4,
                        )
                        single_ref = gr.Textbox(
                            label="Reference",
                            placeholder="Enter the ground-truth answer…",
                            lines=4,
                        )
                        single_btn = gr.Button("Evaluate", variant="primary")

                    with gr.Column(scale=1):
                        single_error = gr.Markdown("")
                        single_results = gr.Dataframe(
                            headers=["Metric", "Score", "Passed"],
                            label="Results",
                            interactive=False,
                            wrap=True,
                        )

                single_btn.click(
                    _evaluate_single,
                    inputs=[single_pred, single_ref, single_metrics],
                    outputs=[single_results, single_error],
                )

            # ── Tab 2: Batch Evaluation ──────────────────────────────────
            with gr.TabItem("Batch Evaluation"):
                gr.Markdown(
                    "Paste **`prediction<TAB>reference`** pairs, one per line. "
                    "The summary table shows aggregate statistics; "
                    "the detail table shows per-pair results."
                )
                batch_metrics = gr.CheckboxGroup(
                    choices=_METRIC_CHOICES,
                    value=["exact_match", "f1", "rouge1"],
                    label="Metrics",
                )
                batch_input = gr.Textbox(
                    label="Pairs  (prediction ⇥ reference)",
                    placeholder=_BATCH_PLACEHOLDER,
                    lines=8,
                )
                batch_btn = gr.Button("Evaluate Batch", variant="primary")
                batch_error = gr.Markdown("")

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_summary = gr.Dataframe(
                            headers=["Metric", "Pass Rate", "Avg Score", "N"],
                            label="Summary",
                            interactive=False,
                        )
                    with gr.Column(scale=2):
                        batch_details = gr.Dataframe(
                            headers=[
                                "#",
                                "Prediction",
                                "Reference",
                                "Metric",
                                "Score",
                                "Passed",
                            ],
                            label="Per-pair results",
                            interactive=False,
                            wrap=True,
                        )

                batch_btn.click(
                    _evaluate_batch,
                    inputs=[batch_input, batch_metrics],
                    outputs=[batch_summary, batch_details, batch_error],
                )

    demo.launch(share=share, server_port=server_port, server_name=server_name)


if __name__ == "__main__":
    launch()
