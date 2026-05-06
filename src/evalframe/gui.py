"""Tkinter GUI for interactive evalframe evaluation sessions.

Launch from the command line::

    python -m evalframe.gui
    # or, after installation:
    evalframe-gui
"""
from __future__ import annotations

import csv
import io
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

from .frame import BUILTIN_METRICS, Evalframe, EvalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_metric(expr: str):
    """Compile a Python lambda expression into a callable metric function.

    The expression must accept two string arguments (prediction, reference)
    and return a bool or a float.  Example::

        lambda pred, ref: pred.strip() == ref.strip()

    .. warning::
        The expression is executed with ``eval()``.  Only use this feature
        with code you trust.
    """
    try:
        fn = eval(expr, {"__builtins__": {}})  # noqa: S307
        if not callable(fn):
            raise ValueError("Expression must be callable (e.g. a lambda).")
        return fn
    except Exception as exc:
        raise ValueError(f"Invalid metric expression: {exc}") from exc


def _parse_pairs(text: str):
    """Parse tab- or comma-separated (prediction, reference) pairs from *text*.

    Returns
    -------
    pairs : list[tuple[str, str]]
    errors : list[str]
        Human-readable descriptions of lines that could not be parsed.
    """
    pairs: List[tuple] = []
    errors: List[str] = []
    for lineno, raw in enumerate(text.strip().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        # Prefer tab delimiter; fall back to CSV parsing
        if "\t" in line:
            parts = line.split("\t", 1)
        else:
            try:
                parts = next(csv.reader(io.StringIO(line)))
            except Exception:
                errors.append(f"Line {lineno}: cannot parse '{line[:50]}'")
                continue
        if len(parts) < 2:
            errors.append(f"Line {lineno}: missing reference column in '{line[:50]}'")
            continue
        pairs.append((parts[0], parts[1]))
    return pairs, errors


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

_PASS_FG = "#27ae60"
_FAIL_FG = "#c0392b"
_BG = "#f4f6f8"
_HEADER_BG = "#2c3e50"
_HEADER_FG = "#ecf0f1"
_BTN_PRIMARY = "#2980b9"
_BTN_SUCCESS = "#27ae60"
_BTN_DANGER = "#c0392b"
_BTN_NEUTRAL = "#7f8c8d"


class _Sidebar(tk.LabelFrame):
    """Left panel for metric management."""

    def __init__(self, parent: tk.Widget, ef: Evalframe, on_change) -> None:
        super().__init__(
            parent,
            text=" Metrics ",
            font=("Helvetica", 10, "bold"),
            bg=_BG,
            width=230,
        )
        self._ef = ef
        self._on_change = on_change
        self._builtin_vars: Dict[str, tk.BooleanVar] = {}
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        pad = {"padx": 10}

        # --- Built-in checkboxes ---
        tk.Label(
            self, text="Built-in metrics", font=("Helvetica", 9, "bold"),
            bg=_BG, fg="#555",
        ).pack(anchor=tk.W, pady=(10, 2), **pad)

        for name in BUILTIN_METRICS:
            var = tk.BooleanVar(value=True)
            self._builtin_vars[name] = var
            tk.Checkbutton(
                self, text=name, variable=var, bg=_BG,
                command=self._toggle_builtin,
            ).pack(anchor=tk.W, padx=22)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=8)

        # --- Custom metric entry ---
        tk.Label(
            self, text="Custom metric", font=("Helvetica", 9, "bold"),
            bg=_BG, fg="#555",
        ).pack(anchor=tk.W, pady=(0, 4), **pad)

        tk.Label(self, text="Name:", bg=_BG, font=("Helvetica", 9)
                 ).pack(anchor=tk.W, **pad)
        self._name_var = tk.StringVar()
        tk.Entry(self, textvariable=self._name_var, width=26
                 ).pack(fill=tk.X, **pad, pady=(0, 4))

        tk.Label(
            self, text="Expression  (pred, ref → bool/float):",
            bg=_BG, font=("Helvetica", 9), wraplength=210, justify=tk.LEFT,
        ).pack(anchor=tk.W, **pad)
        self._expr_text = tk.Text(
            self, height=3, width=26, font=("Courier", 9), wrap=tk.WORD
        )
        self._expr_text.insert(
            "1.0", "lambda pred, ref:\n    pred.strip() == ref.strip()"
        )
        self._expr_text.pack(fill=tk.X, **pad, pady=(0, 4))

        tk.Button(
            self, text="Add Metric", command=self._add_custom,
            bg=_BTN_PRIMARY, fg="white", relief=tk.FLAT, cursor="hand2",
        ).pack(fill=tk.X, **pad, pady=(0, 4))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=8)

        # --- Active metrics list with removal ---
        tk.Label(
            self, text="Active metrics", font=("Helvetica", 9, "bold"),
            bg=_BG, fg="#555",
        ).pack(anchor=tk.W, pady=(0, 2), **pad)

        list_frame = tk.Frame(self, bg=_BG)
        list_frame.pack(fill=tk.X, **pad)
        self._listbox = tk.Listbox(list_frame, height=7, font=("Courier", 9),
                                    selectmode=tk.SINGLE)
        sb = ttk.Scrollbar(list_frame, command=self._listbox.yview)
        self._listbox.configure(yscrollcommand=sb.set)
        self._listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Button(
            self, text="Remove Selected", command=self._remove_selected,
            bg=_BTN_DANGER, fg="white", relief=tk.FLAT, cursor="hand2",
        ).pack(fill=tk.X, **pad, pady=4)

        self.refresh()

    # ------------------------------------------------------------------

    def _toggle_builtin(self) -> None:
        for name, var in self._builtin_vars.items():
            if var.get():
                if name not in self._ef.metrics():
                    self._ef.add_builtin(name)
            else:
                self._ef.remove_metric(name)
        self.refresh()
        self._on_change()

    def _add_custom(self) -> None:
        name = self._name_var.get().strip()
        expr = self._expr_text.get("1.0", tk.END).strip()
        if not name:
            messagebox.showerror("Error", "Metric name cannot be empty.", parent=self)
            return
        try:
            fn = _compile_metric(expr)
        except ValueError as exc:
            messagebox.showerror("Invalid expression", str(exc), parent=self)
            return
        self._ef.add_metric(name, fn)
        self.refresh()
        self._on_change()
        messagebox.showinfo("Added", f"Custom metric '{name}' added.", parent=self)

    def _remove_selected(self) -> None:
        sel = self._listbox.curselection()
        if not sel:
            messagebox.showwarning(
                "No selection", "Select a metric to remove.", parent=self
            )
            return
        name = self._listbox.get(sel[0])
        self._ef.remove_metric(name)
        if name in self._builtin_vars:
            self._builtin_vars[name].set(False)
        self.refresh()
        self._on_change()

    def refresh(self) -> None:
        self._listbox.delete(0, tk.END)
        for m in self._ef.metrics():
            self._listbox.insert(tk.END, m)


# ---------------------------------------------------------------------------

class _SingleTab(tk.Frame):
    """Tab for single (prediction, reference) evaluation."""

    def __init__(self, parent: tk.Widget, ef: Evalframe) -> None:
        super().__init__(parent, bg=_BG)
        self._ef = ef
        self._last_results: Optional[Dict[str, EvalResult]] = None
        self._build()

    def _build(self) -> None:
        # Input
        inp = tk.LabelFrame(
            self, text=" Input ", bg=_BG, font=("Helvetica", 10, "bold")
        )
        inp.pack(fill=tk.X, padx=10, pady=(10, 4))
        inp.columnconfigure(1, weight=1)

        for row, (label, attr) in enumerate(
            [("Prediction:", "_pred_var"), ("Reference:", "_ref_var")]
        ):
            setattr(self, attr, tk.StringVar())
            tk.Label(
                inp, text=label, bg=_BG, font=("Helvetica", 10, "bold"), width=12, anchor=tk.E,
            ).grid(row=row, column=0, padx=(8, 2), pady=6, sticky=tk.E)
            tk.Entry(
                inp, textvariable=getattr(self, attr),
                font=("Helvetica", 10),
            ).grid(row=row, column=1, padx=(2, 8), pady=6, sticky=tk.EW)

        # Action row
        actions = tk.Frame(self, bg=_BG)
        actions.pack(fill=tk.X, padx=10, pady=4)
        tk.Button(
            actions, text="  Evaluate  ", command=self._run,
            bg=_BTN_SUCCESS, fg="white", font=("Helvetica", 11, "bold"),
            relief=tk.FLAT, cursor="hand2", pady=5,
        ).pack(side=tk.LEFT)
        tk.Button(
            actions, text="Clear", command=self._clear,
            bg=_BTN_NEUTRAL, fg="white", relief=tk.FLAT, cursor="hand2",
        ).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(
            actions, text="Export JSON", command=self._export,
            bg=_BTN_NEUTRAL, fg="white", relief=tk.FLAT, cursor="hand2",
        ).pack(side=tk.RIGHT)

        # Results treeview
        res_frame = tk.LabelFrame(
            self, text=" Results ", bg=_BG, font=("Helvetica", 10, "bold")
        )
        res_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        cols = ("metric", "score", "passed")
        self._tree = ttk.Treeview(
            res_frame, columns=cols, show="headings", height=12
        )
        for col, header, width in zip(
            cols, ("Metric", "Score", "Passed"), (200, 140, 100)
        ):
            self._tree.heading(col, text=header)
            self._tree.column(col, width=width, anchor=tk.CENTER)
        self._tree.tag_configure("pass", foreground=_PASS_FG)
        self._tree.tag_configure("fail", foreground=_FAIL_FG)

        vsb = ttk.Scrollbar(res_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------------------------------------------------------

    def _run(self) -> None:
        if not self._ef.metrics():
            messagebox.showwarning(
                "No metrics", "Enable or add at least one metric.", parent=self
            )
            return
        pred = self._pred_var.get()
        ref = self._ref_var.get()
        results = self._ef.evaluate(pred, ref)
        self._last_results = results

        for item in self._tree.get_children():
            self._tree.delete(item)
        for mname, r in results.items():
            score_str = (
                f"{r.score:.4f}" if isinstance(r.score, float) else str(r.score)
            )
            self._tree.insert(
                "", tk.END,
                values=(mname, score_str, "✓  pass" if r.passed else "✗  fail"),
                tags=("pass" if r.passed else "fail",),
            )

    def _clear(self) -> None:
        self._pred_var.set("")
        self._ref_var.set("")
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._last_results = None

    def _export(self) -> None:
        if self._last_results is None:
            messagebox.showwarning("No results", "Run an evaluation first.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            parent=self,
        )
        if path:
            Evalframe.save_results(self._last_results, path)
            messagebox.showinfo("Saved", f"Results written to:\n{path}", parent=self)

    def on_metrics_changed(self) -> None:
        """Re-run evaluation when the active metric set changes."""
        if self._last_results is not None:
            self._run()


# ---------------------------------------------------------------------------

class _BatchTab(tk.Frame):
    """Tab for batch (prediction, reference) evaluation."""

    def __init__(self, parent: tk.Widget, ef: Evalframe) -> None:
        super().__init__(parent, bg=_BG)
        self._ef = ef
        self._batch_results: List[Dict[str, EvalResult]] = []
        self._build()

    def _build(self) -> None:
        tk.Label(
            self,
            text=(
                "Enter one pair per line:  prediction<TAB>reference\n"
                "Comma-separated values are also accepted.  "
                "Blank lines are ignored."
            ),
            bg=_BG, fg="#555", font=("Helvetica", 9), justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=12, pady=(8, 2))

        # Input area
        inp_frame = tk.LabelFrame(
            self, text=" Input Pairs ", bg=_BG, font=("Helvetica", 10, "bold")
        )
        inp_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 4))

        self._text = tk.Text(
            inp_frame, height=7, font=("Courier", 10), wrap=tk.NONE
        )
        self._text.insert(
            "1.0",
            "The quick brown fox\tThe quick brown fox\n"
            "Hello world\tHello there\n"
            "Paris is the capital of France\tParis\n",
        )
        ysb = ttk.Scrollbar(inp_frame, command=self._text.yview)
        xsb = ttk.Scrollbar(inp_frame, orient=tk.HORIZONTAL, command=self._text.xview)
        self._text.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self._text.grid(row=0, column=0, sticky=tk.NSEW, padx=4, pady=4)
        ysb.grid(row=0, column=1, sticky=tk.NS)
        xsb.grid(row=1, column=0, sticky=tk.EW)
        inp_frame.rowconfigure(0, weight=1)
        inp_frame.columnconfigure(0, weight=1)

        # Buttons
        btn_row = tk.Frame(self, bg=_BG)
        btn_row.pack(fill=tk.X, padx=10, pady=4)
        tk.Button(
            btn_row, text="Load CSV / TSV", command=self._load_file,
            bg=_BTN_PRIMARY, fg="white", relief=tk.FLAT, cursor="hand2",
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            btn_row, text="  Run Batch  ", command=self._run,
            bg=_BTN_SUCCESS, fg="white", font=("Helvetica", 10, "bold"),
            relief=tk.FLAT, cursor="hand2",
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            btn_row, text="Export Results", command=self._export,
            bg=_BTN_NEUTRAL, fg="white", relief=tk.FLAT, cursor="hand2",
        ).pack(side=tk.RIGHT)

        # Summary table
        sum_frame = tk.LabelFrame(
            self, text=" Summary ", bg=_BG, font=("Helvetica", 10, "bold")
        )
        sum_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        cols = ("metric", "pass_rate", "avg_score", "n")
        headers = ("Metric", "Pass Rate", "Avg Score", "N")
        widths = (200, 120, 120, 70)
        self._sum_tree = ttk.Treeview(
            sum_frame, columns=cols, show="headings", height=6
        )
        for col, h, w in zip(cols, headers, widths):
            self._sum_tree.heading(col, text=h)
            self._sum_tree.column(col, width=w, anchor=tk.CENTER)
        self._sum_tree.pack(fill=tk.X, padx=4, pady=4)

    # ------------------------------------------------------------------

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("CSV / TSV / Text", "*.csv *.tsv *.txt"),
                ("All files", "*.*"),
            ],
            parent=self,
        )
        if path:
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            self._text.delete("1.0", tk.END)
            self._text.insert("1.0", content)

    def _run(self) -> None:
        if not self._ef.metrics():
            messagebox.showwarning(
                "No metrics", "Enable or add at least one metric.", parent=self
            )
            return
        raw = self._text.get("1.0", tk.END)
        pairs, errors = _parse_pairs(raw)
        if errors:
            messagebox.showerror(
                "Parse errors",
                "\n".join(errors[:8])
                + ("\n…" if len(errors) > 8 else ""),
                parent=self,
            )
            return
        if not pairs:
            messagebox.showwarning("Empty input", "No valid pairs found.", parent=self)
            return

        self._batch_results = self._ef.batch_evaluate(pairs)
        summary = self._ef.summary(self._batch_results)

        for item in self._sum_tree.get_children():
            self._sum_tree.delete(item)
        for mname, stats in summary.items():
            avg = (
                f"{stats['avg_score']:.4f}"
                if stats["avg_score"] is not None
                else "—"
            )
            self._sum_tree.insert(
                "", tk.END,
                values=(
                    mname,
                    f"{stats['pass_rate']:.1%}",
                    avg,
                    stats["n"],
                ),
            )
        messagebox.showinfo(
            "Done",
            f"Evaluated {len(pairs)} pair(s) with "
            f"{len(self._ef.metrics())} metric(s).",
            parent=self,
        )

    def _export(self) -> None:
        if not self._batch_results:
            messagebox.showwarning("No results", "Run a batch first.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            parent=self,
        )
        if path:
            Evalframe.save_results(self._batch_results, path)
            messagebox.showinfo("Saved", f"Results written to:\n{path}", parent=self)


# ---------------------------------------------------------------------------
# Top-level window
# ---------------------------------------------------------------------------


class EvalframeApp:
    """Root application window that wires together all panels."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("evalframe")
        self.root.configure(bg=_BG)
        self.root.geometry("1000x700")
        self.root.minsize(820, 580)

        self._ef = Evalframe(include_builtins=True)
        self._build()

    def _build(self) -> None:
        # ---- Header bar ----
        header = tk.Frame(self.root, bg=_HEADER_BG, height=52)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header, text="evalframe",
            font=("Helvetica", 20, "bold"), bg=_HEADER_BG, fg=_HEADER_FG,
        ).pack(side=tk.LEFT, padx=18, pady=10)
        tk.Label(
            header, text="LLM evaluation framework",
            font=("Helvetica", 10), bg=_HEADER_BG, fg="#95a5a6",
        ).pack(side=tk.LEFT, pady=16)

        # ---- Body: sidebar + notebook ----
        body = tk.Frame(self.root, bg=_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._single_tab = _SingleTab(None, self._ef)   # built below
        self._sidebar = _Sidebar(body, self._ef, on_change=self._on_metrics_changed)
        self._sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self._sidebar.pack_propagate(False)

        nb = ttk.Notebook(body)
        nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._single_tab = _SingleTab(nb, self._ef)
        nb.add(self._single_tab, text="  Single  ")

        self._batch_tab = _BatchTab(nb, self._ef)
        nb.add(self._batch_tab, text="  Batch  ")

    def _on_metrics_changed(self) -> None:
        self._single_tab.on_metrics_changed()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def launch() -> None:
    """Launch the evalframe desktop application."""
    root = tk.Tk()
    EvalframeApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
