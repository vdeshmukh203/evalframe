"""Web-based interactive GUI for evalframe.

Launch with::

    python -m evalframe.gui        # module invocation
    evalframe-gui                  # installed entry-point

The server binds to ``127.0.0.1:5000`` by default and opens the system
browser automatically.  Pass ``--help`` for all options.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Timer
from typing import Any, Dict

from evalframe.frame import BUILTIN_METRICS, Evalframe

# ---------------------------------------------------------------------------
# Embedded single-file HTML application (no external runtime dependencies)
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>evalframe</title>
<style>
*,*::before,*::after{box-sizing:border-box}
body{font-family:system-ui,sans-serif;margin:0;background:#f8fafc;color:#1e293b}
header{background:#1e40af;color:#fff;padding:1rem 1.5rem;display:flex;align-items:center;gap:1rem}
header h1{margin:0;font-size:1.4rem;letter-spacing:-.02em}
header span{font-size:.85rem;opacity:.65}
.container{max-width:960px;margin:2rem auto;padding:0 1.2rem}
.card{background:#fff;border-radius:8px;padding:1.5rem;margin-bottom:1.4rem;
      box-shadow:0 1px 4px rgba(0,0,0,.08)}
h2{font-size:1rem;font-weight:700;margin:0 0 1rem;color:#1e40af;
   text-transform:uppercase;letter-spacing:.06em}
label{display:block;font-size:.83rem;font-weight:600;margin-bottom:.3rem;color:#334155}
textarea,input[type=text]{width:100%;padding:.55rem .7rem;border:1px solid #cbd5e1;
  border-radius:5px;font-family:monospace;font-size:.88rem;resize:vertical;
  transition:border-color .15s}
textarea:focus,input[type=text]:focus{outline:none;border-color:#3b82f6}
.row{display:flex;gap:1rem}
.col{flex:1;min-width:0}
.metric-grid{display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;margin-bottom:1rem}
.metric-grid label{font-weight:400;display:flex;align-items:center;gap:.35rem;cursor:pointer;
  font-size:.88rem;color:#334155}
.custom-row{display:flex;gap:.6rem;margin-bottom:.8rem;align-items:flex-end;flex-wrap:wrap}
.custom-row .field{flex:1;min-width:120px}
.custom-row .field-wide{flex:2;min-width:180px}
button{background:#1e40af;color:#fff;border:none;border-radius:6px;
  padding:.55rem 1.3rem;font-size:.9rem;cursor:pointer;font-weight:600;
  transition:background .15s}
button:hover{background:#1d3a9a}
button.secondary{background:#64748b}
button.secondary:hover{background:#475569}
button.danger{background:#dc2626}
button.danger:hover{background:#b91c1c}
.tabs{display:flex;border-bottom:2px solid #e2e8f0;margin-bottom:1.4rem}
.tab{padding:.55rem 1.2rem;cursor:pointer;border:none;background:none;
  font-size:.92rem;color:#64748b;border-bottom:3px solid transparent;
  margin-bottom:-2px;font-weight:500;transition:color .15s}
.tab.active{color:#1e40af;border-bottom-color:#1e40af;font-weight:700}
.panel{display:none}
.panel.active{display:block}
table{width:100%;border-collapse:collapse;font-size:.88rem;margin-top:.5rem}
thead th{text-align:left;padding:.5rem .6rem;background:#f1f5f9;
  border-bottom:2px solid #e2e8f0;font-size:.78rem;text-transform:uppercase;
  letter-spacing:.05em;color:#64748b}
tbody td{padding:.45rem .6rem;border-bottom:1px solid #f1f5f9;vertical-align:top}
tbody tr:last-child td{border-bottom:none}
.pass{color:#15803d;font-weight:700}
.fail{color:#b91c1c;font-weight:700}
.err-msg{color:#9a3412;font-style:italic;font-size:.78rem;display:block;margin-top:.15rem}
.tag{display:inline-flex;align-items:center;gap:.3rem;background:#eff6ff;
  color:#1e40af;border:1px solid #bfdbfe;border-radius:20px;padding:.2rem .7rem;
  font-size:.78rem;margin:.15rem}
.tag button{background:none;border:none;color:#94a3b8;cursor:pointer;
  padding:0;font-size:.85rem;line-height:1;font-weight:700}
.tag button:hover{color:#dc2626;background:none}
.stat-grid{display:flex;flex-wrap:wrap;gap:.8rem;margin-bottom:1rem}
.stat{background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;
  padding:.8rem 1.1rem;min-width:130px}
.stat-name{font-size:.72rem;color:#64748b;text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:.2rem}
.stat-val{font-size:1.7rem;font-weight:800;color:#1e40af;line-height:1}
.stat-sub{font-size:.75rem;color:#94a3b8;margin-top:.15rem}
#msg{padding:.6rem 1rem;border-radius:5px;font-size:.88rem;margin-bottom:1rem;display:none}
#msg.error{background:#fef2f2;color:#b91c1c;border:1px solid #fecaca}
#msg.info{background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe}
.truncate{max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.score-cell{font-family:monospace}
</style>
</head>
<body>
<header>
  <h1>evalframe</h1>
  <span>Interactive LLM Evaluation Dashboard</span>
</header>

<div class="container">
  <div class="tabs">
    <button class="tab active" onclick="switchTab('single',this)">Single Pair</button>
    <button class="tab" onclick="switchTab('batch',this)">Batch</button>
  </div>

  <!-- ===== SINGLE PAIR PANEL ===== -->
  <div id="panel-single" class="panel active">
    <div class="card">
      <h2>Input</h2>
      <div class="row">
        <div class="col">
          <label for="pred">Prediction (model output)</label>
          <textarea id="pred" rows="4"
            placeholder="e.g. The capital of France is Paris."></textarea>
        </div>
        <div class="col">
          <label for="ref">Reference (ground truth)</label>
          <textarea id="ref" rows="4"
            placeholder="e.g. Paris"></textarea>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Metrics</h2>
      <div class="metric-grid" id="s-builtins">
        <label><input type="checkbox" value="exact_match"> exact_match</label>
        <label><input type="checkbox" value="contains" checked> contains</label>
        <label><input type="checkbox" value="prefix_match"> prefix_match</label>
        <label><input type="checkbox" value="f1" checked> f1</label>
        <label><input type="checkbox" value="rouge1" checked> rouge1</label>
      </div>

      <div style="margin-bottom:.6rem;font-size:.83rem;font-weight:600;color:#334155">
        Custom metric
      </div>
      <div class="custom-row">
        <div class="field">
          <label>Name</label>
          <input type="text" id="s-cname" placeholder="my_metric">
        </div>
        <div class="field-wide">
          <label>Lambda body&nbsp;<span style="font-weight:400;color:#94a3b8">(pred, ref available)</span></label>
          <input type="text" id="s-cbody" placeholder="pred.lower() == ref.lower()">
        </div>
        <button class="secondary" onclick="addCustom('s')">Add</button>
      </div>
      <div id="s-custom-tags"></div>
      <br>
      <button onclick="runSingle()">&#9654; Evaluate</button>
    </div>

    <div id="s-results"></div>
  </div>

  <!-- ===== BATCH PANEL ===== -->
  <div id="panel-batch" class="panel">
    <div class="card">
      <h2>Batch Input</h2>
      <label>
        One pair per line &mdash; tab-separated or pipe-separated
        (<code>prediction | reference</code>)
      </label>
      <textarea id="b-input" rows="9"
        placeholder="The capital of France is Paris.	Paris
Paris is the capital of France.	Paris
The cat sat on the mat.	cat sat on mat"></textarea>
    </div>

    <div class="card">
      <h2>Metrics</h2>
      <div class="metric-grid" id="b-builtins">
        <label><input type="checkbox" value="exact_match"> exact_match</label>
        <label><input type="checkbox" value="contains" checked> contains</label>
        <label><input type="checkbox" value="prefix_match"> prefix_match</label>
        <label><input type="checkbox" value="f1" checked> f1</label>
        <label><input type="checkbox" value="rouge1" checked> rouge1</label>
      </div>
      <button onclick="runBatch()">&#9654; Run Batch</button>
    </div>

    <div id="b-results"></div>
  </div>
</div>

<script>
// -----------------------------------------------------------------------
// State
// -----------------------------------------------------------------------
const customMetrics = {s: {}, b: {}};

// -----------------------------------------------------------------------
// Tab switching
// -----------------------------------------------------------------------
function switchTab(name, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
}

// -----------------------------------------------------------------------
// Custom metric management
// -----------------------------------------------------------------------
function addCustom(ns) {
  const name = document.getElementById(ns + '-cname').value.trim();
  const body = document.getElementById(ns + '-cbody').value.trim();
  if (!name) { alert('Provide a metric name.'); return; }
  if (!body) { alert('Provide a lambda body.'); return; }
  customMetrics[ns][name] = body;
  document.getElementById(ns + '-cname').value = '';
  document.getElementById(ns + '-cbody').value = '';
  renderTags(ns);
}

function removeCustom(ns, name) {
  delete customMetrics[ns][name];
  renderTags(ns);
}

function renderTags(ns) {
  const el = document.getElementById(ns + '-custom-tags');
  el.innerHTML = Object.entries(customMetrics[ns]).map(([n, b]) =>
    '<span class="tag">' +
    '<span><b>' + escHtml(n) + '</b>: ' + escHtml(b) + '</span>' +
    '<button onclick="removeCustom(\'' + ns + '\',\'' + escAttr(n) + '\')">&times;</button>' +
    '</span>'
  ).join('');
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function escAttr(s) {
  return s.replace(/'/g,"\\'");
}

// -----------------------------------------------------------------------
// Selected builtins
// -----------------------------------------------------------------------
function selectedBuiltins(groupId) {
  return [...document.querySelectorAll('#' + groupId + ' input:checked')].map(i => i.value);
}

// -----------------------------------------------------------------------
// Single evaluation
// -----------------------------------------------------------------------
async function runSingle() {
  const prediction = document.getElementById('pred').value;
  const reference  = document.getElementById('ref').value;
  const builtins   = selectedBuiltins('s-builtins');
  const custom     = customMetrics['s'];

  if (!builtins.length && !Object.keys(custom).length) {
    showMsg('s-results', 'Select at least one metric.', 'error');
    return;
  }

  const resp = await fetch('/api/evaluate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prediction, reference, builtins, custom}),
  });
  const data = await resp.json();

  if (data.error) { showMsg('s-results', data.error, 'error'); return; }
  renderSingleResults(data.results, 's-results');
}

function renderSingleResults(results, targetId) {
  let html = '<div class="card"><h2>Results</h2><table>';
  html += '<thead><tr><th>Metric</th><th>Score</th><th>Passed</th></tr></thead><tbody>';
  for (const [metric, r] of Object.entries(results)) {
    const cls  = r.passed ? 'pass' : 'fail';
    const mark = r.passed ? '&#10003; Yes' : '&#10007; No';
    const scoreStr = r.score === null
      ? '<em style="color:#94a3b8">n/a</em>'
      : '<span class="score-cell">' + escHtml(String(r.score)) + '</span>';
    const errStr = r.error
      ? '<span class="err-msg">&#9888; ' + escHtml(r.error) + '</span>'
      : '';
    html += '<tr><td>' + escHtml(metric) + '</td><td>' + scoreStr + errStr +
            '</td><td class="' + cls + '">' + mark + '</td></tr>';
  }
  html += '</tbody></table></div>';
  document.getElementById(targetId).innerHTML = html;
}

// -----------------------------------------------------------------------
// Batch evaluation
// -----------------------------------------------------------------------
async function runBatch() {
  const raw      = document.getElementById('b-input').value.trim();
  const builtins = selectedBuiltins('b-builtins');

  if (!builtins.length) {
    showMsg('b-results', 'Select at least one metric.', 'error');
    return;
  }
  if (!raw) {
    showMsg('b-results', 'Enter at least one prediction/reference pair.', 'error');
    return;
  }

  const pairs = raw.split('\\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)
    .map(line => {
      const sep = line.indexOf('\\t') !== -1 ? '\\t' : '|';
      const idx = line.indexOf(sep);
      if (idx === -1) return [line, ''];
      return [line.slice(0, idx).trim(), line.slice(idx + sep.length).trim()];
    });

  const resp = await fetch('/api/batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pairs, builtins}),
  });
  const data = await resp.json();

  if (data.error) { showMsg('b-results', data.error, 'error'); return; }
  renderBatchResults(data, 'b-results');
}

function renderBatchResults(data, targetId) {
  // Summary cards
  let html = '<div class="card"><h2>Summary</h2><div class="stat-grid">';
  for (const [metric, s] of Object.entries(data.summary)) {
    const pct = (s.pass_rate * 100).toFixed(1);
    const avg = s.avg_score !== null ? s.avg_score.toFixed(4) : '—';
    html += '<div class="stat">' +
      '<div class="stat-name">' + escHtml(metric) + '</div>' +
      '<div class="stat-val">' + pct + '%</div>' +
      '<div class="stat-sub">pass rate &middot; n=' + s.n + '</div>' +
      '<div class="stat-sub">avg score: ' + avg + '</div>' +
      '</div>';
  }
  html += '</div></div>';

  // Per-row table
  const metricNames = Object.keys(data.results[0] || {});
  html += '<div class="card"><h2>Per-pair Results</h2><div style="overflow-x:auto"><table>';
  html += '<thead><tr><th>#</th><th>Prediction</th><th>Reference</th>';
  metricNames.forEach(m => { html += '<th>' + escHtml(m) + '</th>'; });
  html += '</tr></thead><tbody>';

  data.results.forEach((row, i) => {
    const first = Object.values(row)[0];
    html += '<tr><td>' + (i + 1) + '</td>' +
      '<td><span class="truncate" title="' + escHtml(first ? first.prediction : '') + '">' +
      escHtml(first ? first.prediction : '') + '</span></td>' +
      '<td><span class="truncate" title="' + escHtml(first ? first.reference : '') + '">' +
      escHtml(first ? first.reference : '') + '</span></td>';
    metricNames.forEach(m => {
      const r = row[m];
      if (!r) { html += '<td>&#8212;</td>'; return; }
      const cls = r.passed ? 'pass' : 'fail';
      const val = r.score === null ? '<em style="color:#94a3b8">err</em>'
                                   : '<span class="score-cell">' + escHtml(String(r.score)) + '</span>';
      html += '<td class="' + cls + '">' + val + '</td>';
    });
    html += '</tr>';
  });
  html += '</tbody></table></div></div>';
  document.getElementById(targetId).innerHTML = html;
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------
function showMsg(targetId, text, type) {
  document.getElementById(targetId).innerHTML =
    '<div id="msg" class="' + type + '" style="display:block">' + escHtml(text) + '</div>';
}
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler serving the evalframe GUI and JSON API."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # suppress per-request console noise

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw)  # type: ignore[return-value]

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        dispatch = {
            "/api/evaluate": self._handle_evaluate,
            "/api/batch": self._handle_batch,
        }
        handler = dispatch.get(self.path)
        if handler is None:
            self.send_error(404)
            return
        handler()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_evalframe(
        self,
        builtins: list,
        custom: dict,
    ) -> Evalframe:
        """Construct an Evalframe from lists of built-in names and custom lambdas."""
        ef = Evalframe()
        for name in builtins:
            try:
                ef.add_builtin(name)
            except ValueError:
                pass
        for name, body in custom.items():
            try:
                # eval is intentional here: the GUI is a local dev tool and
                # the lambda body is provided interactively by the user.
                fn = eval(f"lambda pred, ref: {body}")  # noqa: S307,PGH001
                ef.add_metric(name, fn)
            except Exception as exc:
                raise ValueError(
                    f"Invalid lambda body for custom metric {name!r}: {exc}"
                ) from exc
        return ef

    def _handle_evaluate(self) -> None:
        try:
            req = self._read_json()
            prediction: str = req.get("prediction", "")
            reference: str = req.get("reference", "")
            ef = self._build_evalframe(
                req.get("builtins", []),
                req.get("custom", {}),
            )
            raw = ef.evaluate(prediction, reference)
            results = {
                k: {
                    "score": v.score,
                    "passed": v.passed,
                    "error": v.error,
                }
                for k, v in raw.items()
            }
            self._send_json({"results": results})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _handle_batch(self) -> None:
        try:
            req = self._read_json()
            pairs: list = req.get("pairs", [])
            ef = self._build_evalframe(req.get("builtins", []), {})
            batch = ef.batch_evaluate(pairs)
            summary = ef.summary(batch)
            serialized = [
                {
                    k: {
                        "score": v.score,
                        "passed": v.passed,
                        "prediction": v.prediction,
                        "reference": v.reference,
                        "error": v.error,
                    }
                    for k, v in row.items()
                }
                for row in batch
            ]
            self._send_json({"results": serialized, "summary": summary})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    host: str = "127.0.0.1",
    port: int = 5000,
    open_browser: bool = True,
) -> None:
    """Start the evalframe web GUI server.

    Parameters
    ----------
    host:
        Hostname to listen on (default ``"127.0.0.1"``).
    port:
        TCP port to listen on (default ``5000``).
    open_browser:
        When ``True`` (default), open the system browser automatically after
        a short delay.
    """
    server = HTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    print(f"evalframe GUI  →  {url}   (Ctrl-C to quit)")
    if open_browser:
        Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    """Entry point for the ``evalframe-gui`` command-line tool."""
    parser = argparse.ArgumentParser(
        description="Launch the evalframe interactive web GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=5000, help="TCP port")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        dest="no_browser",
        help="Do not open the browser automatically",
    )
    args = parser.parse_args()
    run(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
