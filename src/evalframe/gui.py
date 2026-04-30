"""Tkinter desktop GUI for interactive evalframe evaluation."""
from __future__ import annotations

import csv
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List

from evalframe.frame import BUILTIN_METRICS, Evalframe


def launch_gui() -> None:
    """Launch the evalframe desktop GUI.

    Opens a two-tab window:

    * **Single Evaluation** — enter one prediction/reference pair, tick the
      desired metrics, and view a per-metric pass/fail report.
    * **Batch Evaluation** — load a CSV file (columns: ``prediction``,
      ``reference``), run all selected metrics, browse results in a table,
      view aggregate summary statistics, and export to CSV.

    Examples
    --------
    From Python::

        from evalframe.gui import launch_gui
        launch_gui()

    From the command line (after ``pip install evalframe``)::

        evalframe-gui
    """
    app = _App()
    app.mainloop()


class _App(tk.Tk):
    """Top-level application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("evalframe — LLM Output Evaluator")
        self.geometry("900x640")
        self.minsize(700, 480)
        self._build_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        tab_single = ttk.Frame(nb)
        nb.add(tab_single, text="  Single Evaluation  ")
        self._build_single_tab(tab_single)

        tab_batch = ttk.Frame(nb)
        nb.add(tab_batch, text="  Batch Evaluation  ")
        self._build_batch_tab(tab_batch)

    # ------------------------------------------------------------------
    # Single tab
    # ------------------------------------------------------------------

    def _build_single_tab(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent)
        left.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)

        ttk.Label(left, text="Prediction", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w"
        )
        self._pred_box = scrolledtext.ScrolledText(left, height=6, wrap="word")
        self._pred_box.pack(fill="x", pady=(2, 8))

        ttk.Label(left, text="Reference", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w"
        )
        self._ref_box = scrolledtext.ScrolledText(left, height=6, wrap="word")
        self._ref_box.pack(fill="x", pady=(2, 8))

        ttk.Label(left, text="Metrics", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w"
        )
        self._single_vars: Dict[str, tk.BooleanVar] = {}
        mframe = ttk.Frame(left)
        mframe.pack(anchor="w", pady=(2, 0))
        for name in BUILTIN_METRICS:
            var = tk.BooleanVar(value=True)
            self._single_vars[name] = var
            ttk.Checkbutton(mframe, text=name, variable=var).pack(anchor="w")

        ttk.Button(left, text="Evaluate", command=self._run_single).pack(
            pady=12, anchor="w"
        )

        right = ttk.LabelFrame(parent, text="Results")
        right.pack(
            side="right", fill="both", expand=True, padx=(4, 8), pady=8
        )
        self._single_out = scrolledtext.ScrolledText(
            right, state="disabled", wrap="word", font=("TkFixedFont", 10)
        )
        self._single_out.pack(fill="both", expand=True, padx=4, pady=4)

    def _run_single(self) -> None:
        pred = self._pred_box.get("1.0", "end").strip()
        ref = self._ref_box.get("1.0", "end").strip()
        if not pred or not ref:
            messagebox.showwarning(
                "Input required",
                "Please enter both a prediction and a reference.",
            )
            return

        ev = Evalframe()
        for name, var in self._single_vars.items():
            if var.get():
                ev.add_builtin(name)

        if not ev.metrics():
            messagebox.showwarning(
                "No metrics selected", "Please select at least one metric."
            )
            return

        results = ev.evaluate(pred, ref)
        lines: List[str] = []
        for mname, r in results.items():
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  {mname:<16}  score = {str(r.score):<10}  [{status}]")

        self._single_out.config(state="normal")
        self._single_out.delete("1.0", "end")
        self._single_out.insert("end", "\n".join(lines))
        self._single_out.config(state="disabled")

    # ------------------------------------------------------------------
    # Batch tab
    # ------------------------------------------------------------------

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        # --- File picker row ---
        row1 = ttk.Frame(parent)
        row1.pack(fill="x", padx=8, pady=(8, 2))
        ttk.Label(row1, text="CSV (columns: prediction, reference):").pack(side="left")
        self._csv_path = tk.StringVar()
        ttk.Entry(row1, textvariable=self._csv_path, width=42).pack(
            side="left", padx=4
        )
        ttk.Button(row1, text="Browse…", command=self._browse_csv).pack(side="left")
        ttk.Button(row1, text="Run Batch", command=self._run_batch).pack(
            side="left", padx=8
        )

        # --- Metric checkboxes ---
        row2 = ttk.Frame(parent)
        row2.pack(fill="x", padx=8, pady=(2, 4))
        ttk.Label(row2, text="Metrics:").pack(side="left")
        self._batch_vars: Dict[str, tk.BooleanVar] = {}
        for name in BUILTIN_METRICS:
            var = tk.BooleanVar(value=True)
            self._batch_vars[name] = var
            ttk.Checkbutton(row2, text=name, variable=var).pack(side="left", padx=4)

        # --- Results treeview ---
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill="both", expand=True, padx=8, pady=2)
        self._tree = ttk.Treeview(tree_frame, show="headings")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(
            tree_frame, orient="horizontal", command=self._tree.xview
        )
        self._tree.configure(
            yscrollcommand=vsb.set, xscrollcommand=hsb.set
        )
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        # --- Summary ---
        sum_frame = ttk.LabelFrame(parent, text="Summary")
        sum_frame.pack(fill="x", padx=8, pady=4)
        self._summary_box = tk.Text(
            sum_frame,
            height=5,
            state="disabled",
            wrap="word",
            font=("TkFixedFont", 9),
        )
        self._summary_box.pack(fill="x", padx=4, pady=4)

        # --- Export button ---
        bot = ttk.Frame(parent)
        bot.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(
            bot, text="Export Results CSV…", command=self._export_batch
        ).pack(side="right")

        self._batch_rows: List[dict] = []
        self._batch_cols: List[str] = []

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self._csv_path.set(path)

    def _run_batch(self) -> None:
        path = self._csv_path.get().strip()
        if not path:
            messagebox.showwarning("No file selected", "Please select a CSV file first.")
            return

        try:
            pairs = Evalframe.load_pairs_csv(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Load error", str(exc))
            return

        if not pairs:
            messagebox.showinfo("Empty file", "The CSV file contains no data rows.")
            return

        ev = Evalframe()
        for name, var in self._batch_vars.items():
            if var.get():
                ev.add_builtin(name)

        if not ev.metrics():
            messagebox.showwarning(
                "No metrics selected", "Please select at least one metric."
            )
            return

        metrics = ev.metrics()
        results = ev.batch_evaluate(pairs)
        summary = ev.summary(results)

        # Build treeview columns
        score_cols = [f"{m}_score" for m in metrics]
        pass_cols = [f"{m}_pass" for m in metrics]
        self._batch_cols = ["#", "prediction", "reference"] + score_cols + pass_cols

        self._tree.delete(*self._tree.get_children())
        self._tree["columns"] = self._batch_cols
        for col in self._batch_cols:
            self._tree.heading(col, text=col)
            width = 160 if col in ("prediction", "reference") else 80
            self._tree.column(col, width=width, minwidth=50, anchor="center")

        self._batch_rows = []
        for i, (res, (pred, ref)) in enumerate(zip(results, pairs), start=1):
            row: dict = {"#": i, "prediction": pred, "reference": ref}
            for m in metrics:
                r = res.get(m)
                if r is None:
                    row[f"{m}_score"] = "N/A"
                    row[f"{m}_pass"] = "N/A"
                else:
                    try:
                        row[f"{m}_score"] = round(float(r.score), 4)
                    except (TypeError, ValueError):
                        row[f"{m}_score"] = str(r.score)
                    row[f"{m}_pass"] = "Y" if r.passed else "N"
            self._batch_rows.append(row)
            self._tree.insert(
                "", "end", values=[row[c] for c in self._batch_cols]
            )

        # Populate summary
        lines: List[str] = []
        for m, s in summary.items():
            avg = f"{s['avg_score']:.4f}" if s["avg_score"] is not None else "N/A"
            lines.append(
                f"  {m:<16}  pass_rate={s['pass_rate']:.4f}  "
                f"avg_score={avg}  n={s['n']}"
            )
        self._summary_box.config(state="normal")
        self._summary_box.delete("1.0", "end")
        self._summary_box.insert("end", "\n".join(lines))
        self._summary_box.config(state="disabled")

    def _export_batch(self) -> None:
        if not self._batch_rows:
            messagebox.showwarning(
                "No results", "Run a batch evaluation before exporting."
            )
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._batch_cols)
            writer.writeheader()
            writer.writerows(self._batch_rows)
        messagebox.showinfo("Exported", f"Results saved to:\n{path}")
