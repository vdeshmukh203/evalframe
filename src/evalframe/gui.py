"""Tkinter-based graphical interface for evalframe.

Launch from the command line::

    python -m evalframe.gui

or programmatically::

    from evalframe.gui import launch
    launch()
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from evalframe.frame import BUILTIN_METRICS, Evalframe


def launch() -> None:
    """Open the evalframe GUI window (blocking call)."""
    app = _App()
    app.mainloop()


class _App(tk.Tk):
    """Main application window for the evalframe GUI."""

    _COLUMNS = ("metric", "score", "passed", "prediction", "reference")
    _COL_WIDTHS = (110, 80, 60, 230, 230)

    def __init__(self) -> None:
        super().__init__()
        self.title("evalframe — LLM Evaluation GUI")
        self.minsize(820, 580)
        self.resizable(True, True)
        self._custom_metrics: dict[str, object] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_metrics_panel()
        self._build_input_panel()
        self._build_results_panel()

    def _build_metrics_panel(self) -> None:
        frame = ttk.LabelFrame(self, text="Metrics", padding=10)
        frame.grid(row=0, column=0, rowspan=2, padx=(10, 4), pady=10, sticky="ns")

        ttk.Label(frame, text="Built-in", font=("", 9, "bold")).pack(anchor="w")
        self._builtin_vars: dict[str, tk.BooleanVar] = {}
        for name in BUILTIN_METRICS:
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(frame, text=name, variable=var).pack(anchor="w")
            self._builtin_vars[name] = var

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(frame, text="Custom metric", font=("", 9, "bold")).pack(anchor="w")

        ttk.Label(frame, text="Name:").pack(anchor="w", pady=(4, 0))
        self._custom_name = ttk.Entry(frame, width=18)
        self._custom_name.pack(fill="x")

        ttk.Label(frame, text="Lambda body\n(pred, ref) →").pack(anchor="w", pady=(6, 0))
        self._custom_body = ttk.Entry(frame, width=18)
        self._custom_body.insert(0, "pred == ref")
        self._custom_body.pack(fill="x")

        ttk.Button(frame, text="Add custom metric", command=self._add_custom).pack(
            pady=(8, 0), fill="x"
        )

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(frame, text="Registered custom:", font=("", 9, "bold")).pack(anchor="w")
        self._custom_list_var = tk.StringVar(value="(none)")
        ttk.Label(frame, textvariable=self._custom_list_var, foreground="navy",
                  wraplength=130, justify="left").pack(anchor="w")

        ttk.Button(frame, text="Clear custom metrics", command=self._clear_custom).pack(
            pady=(6, 0), fill="x"
        )

    def _build_input_panel(self) -> None:
        frame = ttk.LabelFrame(self, text="Input", padding=10)
        frame.grid(row=0, column=1, padx=(4, 10), pady=(10, 4), sticky="nsew")
        frame.columnconfigure(0, weight=1)

        # Mode selector
        self._mode = tk.StringVar(value="single")
        mode_row = ttk.Frame(frame)
        mode_row.grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(mode_row, text="Mode:").pack(side="left")
        ttk.Radiobutton(
            mode_row, text="Single pair", variable=self._mode,
            value="single", command=self._on_mode_change,
        ).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(
            mode_row, text="Batch (one per line)", variable=self._mode,
            value="batch", command=self._on_mode_change,
        ).pack(side="left", padx=(6, 0))

        ttk.Label(frame, text="Prediction:").grid(row=1, column=0, sticky="w")
        self._pred_box = scrolledtext.ScrolledText(
            frame, height=4, wrap="word", width=50, font=("Courier", 10)
        )
        self._pred_box.grid(row=2, column=0, sticky="ew", pady=(0, 4))

        ttk.Label(frame, text="Reference:").grid(row=3, column=0, sticky="w")
        self._ref_box = scrolledtext.ScrolledText(
            frame, height=4, wrap="word", width=50, font=("Courier", 10)
        )
        self._ref_box.grid(row=4, column=0, sticky="ew")

        self._batch_hint = ttk.Label(
            frame,
            text="Batch mode: one prediction per line above, one reference per line below.",
            foreground="gray",
        )
        self._batch_hint.grid(row=5, column=0, sticky="w", pady=(4, 0))
        self._batch_hint.grid_remove()

        ctrl = ttk.Frame(frame)
        ctrl.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(ctrl, text="Min pass rate:").pack(side="left")
        self._min_pass = ttk.Spinbox(ctrl, from_=0.0, to=1.0, increment=0.1,
                                     width=6, format="%.1f")
        self._min_pass.set("1.0")
        self._min_pass.pack(side="left", padx=(4, 12))

        self._passes_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self._passes_var, width=20).pack(side="left")

        ttk.Button(frame, text="Evaluate", command=self._run).grid(
            row=7, column=0, sticky="w", pady=(8, 0)
        )

    def _build_results_panel(self) -> None:
        frame = ttk.LabelFrame(self, text="Results", padding=10)
        frame.grid(row=1, column=1, padx=(4, 10), pady=(4, 10), sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self._tree = ttk.Treeview(
            frame, columns=self._COLUMNS, show="headings", height=12
        )
        for col, width in zip(self._COLUMNS, self._COL_WIDTHS):
            self._tree.heading(col, text=col.capitalize())
            self._tree.column(col, width=width, anchor="center")
        self._tree.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self._tree.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self._tree.configure(xscrollcommand=hsb.set)

        self._tree.tag_configure("pass", foreground="#1a7a1a")
        self._tree.tag_configure("fail", foreground="#b22222")

        self._summary_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self._summary_var, foreground="navy",
                  wraplength=700, justify="left").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )

        ttk.Button(frame, text="Clear results", command=self._clear_results).grid(
            row=3, column=0, sticky="w", pady=(4, 0)
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_mode_change(self) -> None:
        if self._mode.get() == "batch":
            self._batch_hint.grid()
        else:
            self._batch_hint.grid_remove()

    def _add_custom(self) -> None:
        name = self._custom_name.get().strip()
        body = self._custom_body.get().strip()
        if not name:
            messagebox.showerror("Error", "A metric name is required.", parent=self)
            return
        try:
            # Users run this locally; eval is intentional for lambda entry.
            fn = eval(f"lambda pred, ref: {body}")  # noqa: S307, PGH001
        except SyntaxError as exc:
            messagebox.showerror("Syntax error", str(exc), parent=self)
            return
        self._custom_metrics[name] = fn
        self._refresh_custom_list()
        self._custom_name.delete(0, "end")

    def _clear_custom(self) -> None:
        self._custom_metrics.clear()
        self._refresh_custom_list()

    def _refresh_custom_list(self) -> None:
        if self._custom_metrics:
            self._custom_list_var.set("\n".join(self._custom_metrics))
        else:
            self._custom_list_var.set("(none)")

    def _build_evalframe(self) -> Evalframe:
        ev = Evalframe()
        for name, var in self._builtin_vars.items():
            if var.get():
                ev.add_builtin(name)
        for name, fn in self._custom_metrics.items():
            ev.add_metric(name, fn)  # type: ignore[arg-type]
        return ev

    def _run(self) -> None:
        ev = self._build_evalframe()
        if not ev.metrics():
            messagebox.showwarning(
                "No metrics", "Select at least one metric.", parent=self
            )
            return

        try:
            min_pass = float(self._min_pass.get())
        except ValueError:
            min_pass = 1.0

        self._clear_results()

        if self._mode.get() == "single":
            pred = self._pred_box.get("1.0", "end-1c").strip()
            ref = self._ref_box.get("1.0", "end-1c").strip()
            pairs = [(pred, ref)]
        else:
            preds = [
                line.strip()
                for line in self._pred_box.get("1.0", "end-1c").splitlines()
                if line.strip()
            ]
            refs = [
                line.strip()
                for line in self._ref_box.get("1.0", "end-1c").splitlines()
                if line.strip()
            ]
            if len(preds) != len(refs):
                messagebox.showerror(
                    "Line mismatch",
                    f"Prediction lines ({len(preds)}) ≠ reference lines ({len(refs)}).",
                    parent=self,
                )
                return
            pairs = list(zip(preds, refs))

        results_list = ev.batch_evaluate(pairs)

        for row_results in results_list:
            for er in row_results.values():
                if isinstance(er.score, float):
                    score_str = f"{er.score:.4f}"
                else:
                    score_str = str(er.score)
                tag = "pass" if er.passed else "fail"
                self._tree.insert(
                    "",
                    "end",
                    values=(
                        er.metric,
                        score_str,
                        "✓" if er.passed else "✗",
                        er.prediction[:80],
                        er.reference[:80],
                    ),
                    tags=(tag,),
                )

        summ = ev.summary(results_list)
        parts = []
        for mname, stats in summ.items():
            avg = f"{stats['avg_score']:.4f}" if stats["avg_score"] is not None else "n/a"
            parts.append(f"{mname}: pass_rate={stats['pass_rate']:.1%}, avg={avg}")
        self._summary_var.set("  |  ".join(parts))

        # assert_passes feedback
        all_pass = all(ev.assert_passes(p, r, min_pass) for p, r in pairs)
        self._passes_var.set(
            f"assert_passes: {'✓ YES' if all_pass else '✗ NO'}"
        )

    def _clear_results(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._summary_var.set("")
        self._passes_var.set("")


if __name__ == "__main__":
    launch()
