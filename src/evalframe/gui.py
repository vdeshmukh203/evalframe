"""Graphical user interface for evalframe.

Launch with::

    evalframe-gui

or directly::

    python -m evalframe.gui
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Dict, List, Tuple

from evalframe.frame import BUILTIN_METRICS, Evalframe


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the evalframe GUI application."""
    root = tk.Tk()
    _apply_theme(root)
    EvalframeApp(root)
    root.mainloop()


def _apply_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    # Use a clean built-in theme; fall back gracefully.
    for theme in ("clam", "alt", "default"):
        if theme in style.theme_names():
            style.theme_use(theme)
            break
    style.configure("TLabelframe.Label", font=("", 9, "bold"))
    style.configure("Header.TLabel", font=("", 11, "bold"))
    style.configure("Accent.TButton", font=("", 9, "bold"))


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class EvalframeApp:
    """Root application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("evalframe — LLM Evaluation Framework v0.2.0")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self._ef = Evalframe()
        self._custom_exprs: Dict[str, str] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(paned, width=270)
        paned.add(left, weight=0)
        self._build_metrics_panel(left)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self._build_eval_panel(right)

    # ------------------------------------------------------------------
    # Left panel — metrics
    # ------------------------------------------------------------------

    def _build_metrics_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Metrics", style="Header.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        # Built-in checkboxes
        builtin_lf = ttk.LabelFrame(parent, text="Built-in Metrics", padding=8)
        builtin_lf.pack(fill=tk.X, pady=(0, 8))

        self._builtin_vars: Dict[str, tk.BooleanVar] = {}
        descriptions = {
            "exact_match": "Exact string match",
            "contains":    "Reference ⊆ prediction",
            "prefix_match":"Prediction starts with ref",
            "f1":          "Token-level F1 (SQuAD)",
            "rouge1":      "ROUGE-1 recall",
        }
        for name in BUILTIN_METRICS:
            var = tk.BooleanVar()
            self._builtin_vars[name] = var
            row = ttk.Frame(builtin_lf)
            row.pack(fill=tk.X, pady=1)
            ttk.Checkbutton(
                row, text=name, variable=var, command=self._on_builtin_toggle
            ).pack(side=tk.LEFT)
            ttk.Label(
                row, text=descriptions.get(name, ""), foreground="gray",
                font=("", 8)
            ).pack(side=tk.LEFT, padx=(4, 0))

        # Custom metric
        custom_lf = ttk.LabelFrame(parent, text="Custom Metric", padding=8)
        custom_lf.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(custom_lf, text="Name:").grid(row=0, column=0, sticky="w", pady=2)
        self._custom_name_var = tk.StringVar()
        ttk.Entry(custom_lf, textvariable=self._custom_name_var).grid(
            row=0, column=1, sticky="ew", padx=(4, 0), pady=2
        )

        ttk.Label(custom_lf, text="Expression\n(pred, ref):").grid(
            row=1, column=0, sticky="nw", pady=2
        )
        self._custom_expr_text = tk.Text(custom_lf, height=3, width=20, font=("Courier", 9))
        self._custom_expr_text.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=2)
        self._custom_expr_text.insert("1.0", "pred.strip() == ref.strip()")

        custom_lf.columnconfigure(1, weight=1)
        ttk.Button(
            custom_lf, text="Add Custom Metric", command=self._add_custom_metric
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        # Active metrics list
        active_lf = ttk.LabelFrame(parent, text="Active Metrics", padding=8)
        active_lf.pack(fill=tk.BOTH, expand=True)

        self._active_listbox = tk.Listbox(active_lf, height=7, selectmode=tk.SINGLE)
        sb = ttk.Scrollbar(active_lf, orient=tk.VERTICAL,
                           command=self._active_listbox.yview)
        self._active_listbox.configure(yscrollcommand=sb.set)
        self._active_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(
            parent, text="Remove Selected Metric",
            command=self._remove_selected_metric
        ).pack(fill=tk.X, pady=(4, 0))

    # ------------------------------------------------------------------
    # Right panel — evaluation notebook
    # ------------------------------------------------------------------

    def _build_eval_panel(self, parent: ttk.Frame) -> None:
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)

        # ── Single evaluation ──────────────────────────────────────────
        single_tab = ttk.Frame(nb, padding=10)
        nb.add(single_tab, text="  Single Evaluation  ")
        self._build_single_tab(single_tab)

        # ── Batch evaluation ───────────────────────────────────────────
        batch_tab = ttk.Frame(nb, padding=10)
        nb.add(batch_tab, text="  Batch Evaluation  ")
        self._build_batch_tab(batch_tab)

    # ── Single tab ─────────────────────────────────────────────────────

    def _build_single_tab(self, parent: ttk.Frame) -> None:
        io_frame = ttk.Frame(parent)
        io_frame.pack(fill=tk.X, pady=(0, 6))
        io_frame.columnconfigure(0, weight=1)
        io_frame.columnconfigure(1, weight=1)

        ttk.Label(io_frame, text="Prediction", font=("", 9, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(io_frame, text="Reference", font=("", 9, "bold")).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )

        self._pred_text = scrolledtext.ScrolledText(
            io_frame, height=6, wrap=tk.WORD, font=("", 10)
        )
        self._pred_text.grid(row=1, column=0, sticky="nsew", pady=(2, 0))

        self._ref_text = scrolledtext.ScrolledText(
            io_frame, height=6, wrap=tk.WORD, font=("", 10)
        )
        self._ref_text.grid(row=1, column=1, sticky="nsew", pady=(2, 0), padx=(8, 0))
        io_frame.rowconfigure(1, weight=1)

        ttk.Button(
            parent, text="▶  Evaluate", style="Accent.TButton",
            command=self._run_single_eval
        ).pack(pady=8)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(parent, text="Results", font=("", 9, "bold")).pack(anchor="w")
        self._single_tree = self._make_tree(
            parent, columns=["Metric", "Score", "Passed"]
        )
        self._single_tree.column("Metric",  width=160)
        self._single_tree.column("Score",   width=120, anchor="center")
        self._single_tree.column("Passed",  width=80,  anchor="center")
        self._single_tree.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    # ── Batch tab ──────────────────────────────────────────────────────

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        pairs_lf = ttk.LabelFrame(parent, text="Input Pairs", padding=8)
        pairs_lf.pack(fill=tk.X, pady=(0, 6))

        hdr = ttk.Frame(pairs_lf)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="#",          width=3,  font=("", 8, "bold")).pack(side=tk.LEFT)
        ttk.Label(hdr, text="Prediction", width=38, font=("", 8, "bold")).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(hdr, text="Reference",  width=38, font=("", 8, "bold")).pack(side=tk.LEFT, padx=(4, 0))

        self._pairs_container = ttk.Frame(pairs_lf)
        self._pairs_container.pack(fill=tk.X)

        self._pair_rows: List[Tuple[ttk.Entry, ttk.Entry]] = []
        for _ in range(3):
            self._add_pair_row()

        btn_row = ttk.Frame(pairs_lf)
        btn_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btn_row, text="+ Add Row",     command=self._add_pair_row).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="− Remove Last", command=self._remove_last_row).pack(side=tk.LEFT)

        ttk.Button(
            parent, text="▶  Evaluate All", style="Accent.TButton",
            command=self._run_batch_eval
        ).pack(pady=8)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 8))

        results_frame = ttk.Frame(parent)
        results_frame.pack(fill=tk.BOTH, expand=True)
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(1, weight=1)

        ttk.Label(results_frame, text="Summary", font=("", 9, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(results_frame, text="Per-pair Detail", font=("", 9, "bold")).grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )

        self._summary_tree = self._make_tree(
            results_frame, columns=["Metric", "Pass Rate", "Avg Score", "N"]
        )
        self._summary_tree.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

        self._detail_tree = self._make_tree(
            results_frame, columns=["#", "Metric", "Score", "Passed"]
        )
        self._detail_tree.column("#",      width=30,  anchor="center")
        self._detail_tree.column("Metric", width=120)
        self._detail_tree.column("Score",  width=90,  anchor="center")
        self._detail_tree.column("Passed", width=70,  anchor="center")
        self._detail_tree.grid(row=1, column=1, sticky="nsew", pady=(4, 0), padx=(12, 0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_tree(parent: ttk.Frame, columns: List[str]) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=110)

        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Alternate row colours
        tree.tag_configure("odd",  background="#f5f5f5")
        tree.tag_configure("even", background="#ffffff")
        tree.tag_configure("pass", foreground="#1a7a1a")
        tree.tag_configure("fail", foreground="#a01a1a")

        return tree

    def _add_pair_row(self) -> None:
        idx = len(self._pair_rows) + 1
        row_frame = ttk.Frame(self._pairs_container)
        row_frame.pack(fill=tk.X, pady=2)
        ttk.Label(row_frame, text=str(idx), width=3, anchor="center").pack(side=tk.LEFT)
        pred_e = ttk.Entry(row_frame, width=38)
        pred_e.pack(side=tk.LEFT, padx=(4, 0))
        ref_e = ttk.Entry(row_frame, width=38)
        ref_e.pack(side=tk.LEFT, padx=(4, 0))
        self._pair_rows.append((pred_e, ref_e))

    def _remove_last_row(self) -> None:
        if len(self._pair_rows) > 1:
            self._pair_rows.pop()
            for widget in self._pairs_container.winfo_children():
                pass
            # Destroy the last child frame
            children = self._pairs_container.winfo_children()
            if children:
                children[-1].destroy()

    def _refresh_active_list(self) -> None:
        self._active_listbox.delete(0, tk.END)
        for name in self._ef.metrics():
            self._active_listbox.insert(tk.END, name)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_builtin_toggle(self) -> None:
        for name, var in self._builtin_vars.items():
            if var.get() and name not in self._ef.metrics():
                self._ef.add_builtin(name)
            elif not var.get() and name in self._ef.metrics():
                self._ef.remove_metric(name)
        self._refresh_active_list()

    def _add_custom_metric(self) -> None:
        name = self._custom_name_var.get().strip()
        expr = self._custom_expr_text.get("1.0", tk.END).strip()

        if not name:
            messagebox.showerror("Error", "Metric name cannot be empty.")
            return
        if not expr:
            messagebox.showerror("Error", "Expression cannot be empty.")
            return

        try:
            # Evaluate the expression as a lambda body; safe for a local desktop tool.
            fn = eval(f"lambda pred, ref: {expr}")  # noqa: S307
        except SyntaxError as exc:
            messagebox.showerror("Syntax Error", f"Invalid expression:\n{exc}")
            return
        except Exception as exc:
            messagebox.showerror("Error", f"Could not parse expression:\n{exc}")
            return

        try:
            self._ef.add_metric(name, fn)
        except TypeError as exc:
            messagebox.showerror("Error", str(exc))
            return

        self._custom_exprs[name] = expr
        self._refresh_active_list()
        self._custom_name_var.set("")
        self._custom_expr_text.delete("1.0", tk.END)
        self._custom_expr_text.insert("1.0", "pred.strip() == ref.strip()")

    def _remove_selected_metric(self) -> None:
        sel = self._active_listbox.curselection()
        if not sel:
            messagebox.showinfo("Info", "Select a metric in the list first.")
            return
        name = self._active_listbox.get(sel[0])
        self._ef.remove_metric(name)
        self._custom_exprs.pop(name, None)
        if name in self._builtin_vars:
            self._builtin_vars[name].set(False)
        self._refresh_active_list()

    def _run_single_eval(self) -> None:
        pred = self._pred_text.get("1.0", tk.END).strip()
        ref = self._ref_text.get("1.0", tk.END).strip()

        if not self._ef.metrics():
            messagebox.showwarning(
                "No Metrics", "Add at least one metric before evaluating."
            )
            return

        results = self._ef.evaluate(pred, ref)
        self._single_tree.delete(*self._single_tree.get_children())
        for i, (mname, r) in enumerate(results.items()):
            tag = "pass" if r.passed else "fail"
            row_tag = "even" if i % 2 == 0 else "odd"
            score_str = (
                f"{r.score:.4f}" if isinstance(r.score, float) else str(r.score)
            )
            passed_str = "✓  pass" if r.passed else "✗  fail"
            self._single_tree.insert(
                "", tk.END,
                values=[mname, score_str, passed_str],
                tags=(row_tag, tag),
            )

    def _run_batch_eval(self) -> None:
        pairs = [
            (pe.get().strip(), re.get().strip())
            for pe, re in self._pair_rows
            if pe.get().strip() or re.get().strip()
        ]
        if not pairs:
            messagebox.showwarning("No Input", "Enter at least one pair.")
            return
        if not self._ef.metrics():
            messagebox.showwarning(
                "No Metrics", "Add at least one metric before evaluating."
            )
            return

        batch_results = self._ef.batch_evaluate(pairs)
        summary = self._ef.summary(batch_results)

        # Populate summary tree
        self._summary_tree.delete(*self._summary_tree.get_children())
        for i, (mname, stats) in enumerate(summary.items()):
            avg = (
                f"{stats['avg_score']:.4f}"
                if stats["avg_score"] is not None
                else "—"
            )
            row_tag = "even" if i % 2 == 0 else "odd"
            self._summary_tree.insert(
                "", tk.END,
                values=[mname, f"{stats['pass_rate']:.1%}", avg, stats["n"]],
                tags=(row_tag,),
            )

        # Populate per-pair detail tree
        self._detail_tree.delete(*self._detail_tree.get_children())
        row_idx = 0
        for pair_idx, result in enumerate(batch_results, start=1):
            for mname, r in result.items():
                tag = "pass" if r.passed else "fail"
                row_tag = "even" if row_idx % 2 == 0 else "odd"
                score_str = (
                    f"{r.score:.4f}" if isinstance(r.score, float) else str(r.score)
                )
                passed_str = "✓" if r.passed else "✗"
                self._detail_tree.insert(
                    "", tk.END,
                    values=[pair_idx, mname, score_str, passed_str],
                    tags=(row_tag, tag),
                )
                row_idx += 1


if __name__ == "__main__":
    main()
