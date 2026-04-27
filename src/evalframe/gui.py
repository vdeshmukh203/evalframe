"""Tkinter GUI for interactive evalframe evaluation.

Launch with::

    evalframe-gui          # console-script entry point
    python -m evalframe.gui
"""
from __future__ import annotations

import csv
import io
from typing import Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ImportError as _tk_err:
    raise ImportError(
        "The evalframe GUI requires tkinter. "
        "Install it with your system package manager "
        "(e.g. 'apt install python3-tk' on Debian/Ubuntu)."
    ) from _tk_err

from .frame import BUILTIN_METRICS, Evalframe


# ---------------------------------------------------------------------------
# Colour palette (accessible, works on light and dark OS themes)
# ---------------------------------------------------------------------------
_PASS_FG = "#1a7a1a"
_FAIL_FG = "#b30000"
_HEADER_BG = "#2d5fa6"
_HEADER_FG = "#ffffff"


class _SinglePane(ttk.Frame):
    """Evaluate one prediction/reference pair."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, padding=12)
        self._ef = Evalframe()
        self._metric_vars: Dict[str, tk.BooleanVar] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)

        # --- Input fields ---
        ttk.Label(self, text="Prediction:").grid(row=0, column=0, sticky="nw", pady=(0, 4))
        self._pred_box = scrolledtext.ScrolledText(self, height=4, wrap=tk.WORD)
        self._pred_box.grid(row=0, column=1, sticky="ew", pady=(0, 4))

        ttk.Label(self, text="Reference:").grid(row=1, column=0, sticky="nw", pady=(0, 8))
        self._ref_box = scrolledtext.ScrolledText(self, height=4, wrap=tk.WORD)
        self._ref_box.grid(row=1, column=1, sticky="ew", pady=(0, 8))

        # --- Built-in metric checkboxes ---
        metric_frame = ttk.LabelFrame(self, text="Built-in metrics", padding=8)
        metric_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        for i, name in enumerate(BUILTIN_METRICS):
            var = tk.BooleanVar(value=True)
            self._metric_vars[name] = var
            ttk.Checkbutton(metric_frame, text=name, variable=var).grid(
                row=0, column=i, padx=6, sticky="w"
            )

        # --- Custom metric entry ---
        custom_frame = ttk.LabelFrame(self, text="Custom metric (Python lambda)", padding=8)
        custom_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        custom_frame.columnconfigure(1, weight=1)
        ttk.Label(custom_frame, text="Name:").grid(row=0, column=0, sticky="w")
        self._custom_name = ttk.Entry(custom_frame, width=14)
        self._custom_name.grid(row=0, column=1, sticky="w", padx=(4, 8))
        ttk.Label(custom_frame, text="fn(pred, ref):").grid(row=0, column=2, sticky="w")
        self._custom_fn = ttk.Entry(custom_frame, width=40)
        self._custom_fn.grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Button(custom_frame, text="Add", command=self._add_custom).grid(
            row=0, column=4, padx=(4, 0)
        )

        # --- Run button ---
        ttk.Button(self, text="Run Evaluation", command=self._run).grid(
            row=4, column=0, columnspan=2, pady=8
        )

        # --- Results table ---
        result_frame = ttk.LabelFrame(self, text="Results", padding=4)
        result_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(0, 4))
        self.rowconfigure(5, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        cols = ("Metric", "Score", "Passed")
        self._tree = ttk.Treeview(result_frame, columns=cols, show="headings", height=7)
        for col in cols:
            self._tree.heading(col, text=col)
            self._tree.column(col, width=140, anchor="center")
        self._tree.column("Metric", anchor="w", width=180)
        self._tree.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(result_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")

        self._tree.tag_configure("pass", foreground=_PASS_FG)
        self._tree.tag_configure("fail", foreground=_FAIL_FG)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _add_custom(self) -> None:
        name = self._custom_name.get().strip()
        expr = self._custom_fn.get().strip()
        if not name or not expr:
            messagebox.showwarning("Missing input", "Provide both a name and a lambda expression.")
            return
        try:
            fn = eval(f"lambda pred, ref: {expr}", {}, {})  # noqa: S307
            if not callable(fn):
                raise TypeError
        except Exception as exc:
            messagebox.showerror("Invalid expression", str(exc))
            return
        self._ef.add_metric(name, fn)
        messagebox.showinfo("Added", f"Custom metric '{name}' registered.")
        self._custom_name.delete(0, tk.END)
        self._custom_fn.delete(0, tk.END)

    def _run(self) -> None:
        pred = self._pred_box.get("1.0", tk.END).strip()
        ref = self._ref_box.get("1.0", tk.END).strip()

        # Sync selected built-ins
        for name, var in self._metric_vars.items():
            if var.get() and name not in self._ef.metrics():
                self._ef.add_builtin(name)
            elif not var.get() and name in self._ef.metrics():
                self._ef.remove_metric(name)

        if not self._ef.metrics():
            messagebox.showwarning("No metrics", "Select at least one metric.")
            return

        results = self._ef.evaluate(pred, ref)
        self._tree.delete(*self._tree.get_children())
        for mname, er in results.items():
            tag = "pass" if er.passed else "fail"
            score_str = f"{er.score:.4f}" if isinstance(er.score, float) else str(er.score)
            passed_str = "✓ Yes" if er.passed else "✗ No"
            self._tree.insert("", tk.END, values=(mname, score_str, passed_str), tags=(tag,))


class _BatchPane(ttk.Frame):
    """Batch evaluation: paste CSV or load a file, show aggregate summary."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, padding=12)
        self._ef = Evalframe()
        self._metric_vars: Dict[str, tk.BooleanVar] = {}
        self._last_results: Optional[List[Dict]] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)

        # --- Built-in metric checkboxes ---
        metric_frame = ttk.LabelFrame(self, text="Built-in metrics", padding=8)
        metric_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for i, name in enumerate(BUILTIN_METRICS):
            var = tk.BooleanVar(value=True)
            self._metric_vars[name] = var
            ttk.Checkbutton(metric_frame, text=name, variable=var).grid(
                row=0, column=i, padx=6, sticky="w"
            )

        # --- Input area ---
        hint = ttk.LabelFrame(
            self,
            text='Input — paste or load CSV with columns "prediction,reference"',
            padding=4,
        )
        hint.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        hint.columnconfigure(0, weight=1)
        hint.rowconfigure(0, weight=1)

        self._input_box = scrolledtext.ScrolledText(hint, height=8, wrap=tk.WORD)
        self._input_box.grid(row=0, column=0, sticky="ew")
        self._input_box.insert("1.0", "prediction,reference\n")

        btn_row = ttk.Frame(self)
        btn_row.grid(row=2, column=0, sticky="w", pady=4)
        ttk.Button(btn_row, text="Load CSV…", command=self._load_csv).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Run Batch", command=self._run).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Export Results…", command=self._export).pack(side="left")

        # --- Summary table ---
        summary_frame = ttk.LabelFrame(self, text="Summary (per metric)", padding=4)
        summary_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 4))
        self.rowconfigure(3, weight=1)
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)

        cols = ("Metric", "Pass Rate", "Avg Score", "N")
        self._summary_tree = ttk.Treeview(
            summary_frame, columns=cols, show="headings", height=6
        )
        for col in cols:
            self._summary_tree.heading(col, text=col)
            self._summary_tree.column(col, width=140, anchor="center")
        self._summary_tree.column("Metric", anchor="w", width=180)
        self._summary_tree.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(summary_frame, orient="vertical", command=self._summary_tree.yview)
        self._summary_tree.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")

        self._summary_tree.tag_configure("good", foreground=_PASS_FG)
        self._summary_tree.tag_configure("bad", foreground=_FAIL_FG)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _load_csv(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                content = fh.read()
        except OSError as exc:
            messagebox.showerror("File error", str(exc))
            return
        self._input_box.delete("1.0", tk.END)
        self._input_box.insert("1.0", content)

    def _parse_pairs(self) -> List[Tuple[str, str]]:
        text = self._input_box.get("1.0", tk.END).strip()
        reader = csv.DictReader(io.StringIO(text))
        pairs: List[Tuple[str, str]] = []
        for row in reader:
            pred = row.get("prediction", "").strip()
            ref = row.get("reference", "").strip()
            pairs.append((pred, ref))
        return pairs

    def _run(self) -> None:
        # Sync selected built-ins
        for name, var in self._metric_vars.items():
            if var.get() and name not in self._ef.metrics():
                self._ef.add_builtin(name)
            elif not var.get() and name in self._ef.metrics():
                self._ef.remove_metric(name)

        if not self._ef.metrics():
            messagebox.showwarning("No metrics", "Select at least one metric.")
            return

        try:
            pairs = self._parse_pairs()
        except Exception as exc:
            messagebox.showerror("Parse error", str(exc))
            return

        if not pairs:
            messagebox.showwarning("No data", "No valid rows found in the input.")
            return

        self._last_results = self._ef.batch_evaluate(pairs)
        summary = self._ef.summary(self._last_results)

        self._summary_tree.delete(*self._summary_tree.get_children())
        for mname, info in summary.items():
            pr = info["pass_rate"]
            avg = f"{info['avg_score']:.4f}" if info["avg_score"] is not None else "—"
            tag = "good" if pr >= 0.5 else "bad"
            self._summary_tree.insert(
                "",
                tk.END,
                values=(mname, f"{pr:.2%}", avg, info["n"]),
                tags=(tag,),
            )

    def _export(self) -> None:
        if not self._last_results:
            messagebox.showwarning("No results", "Run the batch first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        csv_text = self._ef.to_csv(self._last_results)
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(csv_text)
            messagebox.showinfo("Saved", f"Results written to:\n{path}")
        except OSError as exc:
            messagebox.showerror("Save error", str(exc))


# ---------------------------------------------------------------------------
# Application window
# ---------------------------------------------------------------------------

class App(tk.Tk):
    """Root window for the evalframe GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.title("evalframe — LLM Evaluation Tool")
        self.minsize(740, 560)
        self._build_ui()

    def _build_ui(self) -> None:
        # Header bar
        header = tk.Frame(self, bg=_HEADER_BG)
        header.pack(fill="x")
        tk.Label(
            header,
            text="evalframe",
            font=("Helvetica", 18, "bold"),
            bg=_HEADER_BG,
            fg=_HEADER_FG,
            padx=12,
            pady=8,
        ).pack(side="left")
        tk.Label(
            header,
            text="Lightweight LLM Evaluation Framework",
            font=("Helvetica", 10),
            bg=_HEADER_BG,
            fg="#c8d8f8",
            padx=4,
        ).pack(side="left", pady=8)

        # Notebook (tabs)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        nb.add(_SinglePane(nb), text="  Single Pair  ")
        nb.add(_BatchPane(nb), text="  Batch Evaluation  ")


def main() -> None:
    """Entry point — opens the evalframe GUI window."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
