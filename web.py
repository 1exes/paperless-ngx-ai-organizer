"""
Paperless-NGX Organizer - WebUI
Flask-based dashboard with embedded HTML/CSS/JS.
Runs as daemon thread alongside autopilot.
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
import logging
from datetime import datetime
from collections import deque

from flask import Flask, Response, request, jsonify

log = logging.getLogger("organizer")

# Shared state - updated by autopilot loop
_webui_status: dict = {
    "running": False,
    "cycle": 0,
    "started_at": None,
    "last_cycle_at": None,
    "seen_new": 0,
    "candidates": 0,
    "updated": 0,
    "failed": 0,
    "skipped_duplicates": 0,
    "skipped_organized": 0,
    "poll_errors": 0,
    "maintenance_runs": 0,
    "reviews_resolved": 0,
}

# References set by start_webui()
_run_db = None
_log_file = None
_env_file = None

# SSE log broadcasting
_log_subscribers: list[deque] = []
_log_lock = threading.Lock()


class _LogBroadcastHandler(logging.Handler):
    """Captures log records and broadcasts to SSE subscribers."""
    def emit(self, record):
        try:
            msg = self.format(record)
            with _log_lock:
                dead = []
                for q in _log_subscribers:
                    try:
                        q.append(msg)
                        if len(q) > 500:
                            q.popleft()
                    except Exception:
                        dead.append(q)
                for d in dead:
                    _log_subscribers.remove(d)
        except Exception:
            pass


def _create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    @app.route("/")
    def index():
        return Response(_HTML_TEMPLATE, content_type="text/html; charset=utf-8")

    @app.route("/api/status")
    def api_status():
        status = dict(_webui_status)
        if status.get("started_at"):
            elapsed = time.time() - status["started_at"]
            h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
            status["uptime"] = f"{h}h {m}m"
        else:
            status["uptime"] = "-"
        return jsonify(status)

    @app.route("/api/stats")
    def api_stats():
        if not _run_db:
            return jsonify({"error": "DB not ready"}), 503
        days = request.args.get("days", 30, type=int)
        return jsonify(_run_db.get_processing_stats(days=days))

    @app.route("/api/calibration")
    def api_calibration():
        if not _run_db:
            return jsonify({"error": "DB not ready"}), 503
        return jsonify(_run_db.get_confidence_calibration())

    @app.route("/api/reviews")
    def api_reviews():
        if not _run_db:
            return jsonify({"error": "DB not ready"}), 503
        limit = request.args.get("limit", 50, type=int)
        return jsonify(_run_db.list_open_reviews(limit=limit))

    @app.route("/api/reviews/<int:review_id>")
    def api_review_detail(review_id):
        if not _run_db:
            return jsonify({"error": "DB not ready"}), 503
        rev = _run_db.get_review_with_suggestion(review_id)
        if not rev:
            return jsonify({"error": "not found"}), 404
        if rev.get("suggestion_json"):
            try:
                rev["suggestion"] = json.loads(rev["suggestion_json"])
            except Exception:
                rev["suggestion"] = {}
            del rev["suggestion_json"]
        return jsonify(rev)

    @app.route("/api/reviews/<int:review_id>/approve", methods=["POST"])
    def api_review_approve(review_id):
        if not _run_db:
            return jsonify({"error": "DB not ready"}), 503
        ok = _run_db.close_review(review_id)
        if ok:
            log.info(f"WEBUI: Review #{review_id} approved")
            return jsonify({"ok": True})
        return jsonify({"error": "not found or already closed"}), 404

    @app.route("/api/reviews/<int:review_id>/reject", methods=["POST"])
    def api_review_reject(review_id):
        if not _run_db:
            return jsonify({"error": "DB not ready"}), 503
        ok = _run_db.close_review(review_id)
        if ok:
            log.info(f"WEBUI: Review #{review_id} rejected")
            return jsonify({"ok": True})
        return jsonify({"error": "not found or already closed"}), 404

    @app.route("/api/log")
    def api_log():
        lines = request.args.get("lines", 200, type=int)
        lines = min(lines, 2000)
        if not _log_file or not os.path.exists(_log_file):
            return jsonify({"lines": [], "total": 0})
        try:
            with open(_log_file, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            tail = all_lines[-lines:]
            return jsonify({"lines": [l.rstrip() for l in tail], "total": len(all_lines)})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/log/stream")
    def api_log_stream():
        def generate():
            q = deque(maxlen=500)
            with _log_lock:
                _log_subscribers.append(q)
            try:
                while True:
                    while q:
                        line = q.popleft()
                        yield f"data: {json.dumps(line, ensure_ascii=False)}\n\n"
                    time.sleep(0.5)
            except GeneratorExit:
                pass
            finally:
                with _log_lock:
                    if q in _log_subscribers:
                        _log_subscribers.remove(q)
        return Response(generate(), content_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/config")
    def api_config():
        if not _env_file or not os.path.exists(_env_file):
            return jsonify({"error": ".env not found"}), 404
        secrets = {"PAPERLESS_TOKEN", "LLM_API_KEY"}
        result = []
        try:
            with open(_env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
                    masked = key in secrets and bool(val)
                    result.append({
                        "key": key,
                        "value": "***" if masked else val,
                        "secret": masked,
                    })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify(result)

    @app.route("/api/config", methods=["POST"])
    def api_config_save():
        if not _env_file or not os.path.exists(_env_file):
            return jsonify({"error": ".env not found"}), 404
        changes = request.get_json(silent=True)
        if not changes or not isinstance(changes, dict):
            return jsonify({"error": "invalid payload"}), 400
        # Read current .env
        try:
            with open(_env_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        applied = []
        live_keys = {
            "LOG_LEVEL", "AUTOPILOT_INTERVAL_SEC", "AUTOPILOT_CLEANUP_EVERY_CYCLES",
            "AUTOPILOT_DUPLICATE_SCAN_EVERY_CYCLES", "AUTOPILOT_REVIEW_RESOLVE_EVERY_CYCLES",
            "AUTOPILOT_MAX_NEW_DOCS_PER_CYCLE", "MAX_TAGS_PER_DOC", "LLM_TEMPERATURE",
            "LLM_TIMEOUT", "LLM_MAX_TOKENS", "LLM_KEEP_ALIVE", "QUIET_HOURS_START",
            "QUIET_HOURS_END",
        }

        for key, new_val in changes.items():
            if not re.match(r'^[A-Z][A-Z0-9_]*$', key):
                continue
            # Do not allow changing secrets via WebUI
            if key in ("PAPERLESS_TOKEN", "LLM_API_KEY"):
                continue
            found = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("#") or "=" not in stripped:
                    continue
                lk = stripped.split("=", 1)[0].strip()
                if lk == key:
                    lines[i] = f"{key}={new_val}\n"
                    found = True
                    break
            if not found:
                lines.append(f"{key}={new_val}\n")
            applied.append(key)
            # Live-apply where possible
            if key in live_keys:
                os.environ[key] = str(new_val)

        try:
            with open(_env_file, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        log.info(f"WEBUI: Config updated: {', '.join(applied)}")
        return jsonify({"ok": True, "applied": applied, "needs_restart": [
            k for k in applied if k not in live_keys
        ]})

    return app


def start_webui(run_db, log_file: str, env_file: str, port: int = 5580):
    """Start Flask WebUI as daemon thread."""
    global _run_db, _log_file, _env_file
    _run_db = run_db
    _log_file = log_file
    _env_file = env_file

    # Install broadcast handler
    handler = _LogBroadcastHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger("organizer").addHandler(handler)

    app = _create_app()

    # Suppress Flask request logging
    wlog = logging.getLogger("werkzeug")
    wlog.setLevel(logging.WARNING)

    def _run():
        try:
            app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
        except Exception as exc:
            log.error(f"WEBUI: Flask crashed: {exc}")

    t = threading.Thread(target=_run, name="webui", daemon=True)
    t.start()
    log.info(f"WEBUI: Dashboard on http://0.0.0.0:{port}")
    return t


# ── HTML Template ──────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paperless Organizer</title>
<style>
:root {
  --bg: #1a1b26; --bg2: #24283b; --bg3: #2f3347;
  --fg: #c0caf5; --fg2: #a9b1d6; --dim: #565f89;
  --accent: #7aa2f7; --green: #9ece6a; --red: #f7768e;
  --orange: #ff9e64; --yellow: #e0af68; --cyan: #7dcfff;
  --border: #3b4261; --radius: 8px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
a { color: var(--accent); text-decoration: none; }

/* Layout */
.header { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
.header h1 { font-size: 18px; font-weight: 600; color: var(--accent); }
.header .version { color: var(--dim); font-size: 12px; }
.tabs { display: flex; gap: 2px; margin-left: auto; }
.tab { padding: 8px 18px; border-radius: var(--radius) var(--radius) 0 0; cursor: pointer; color: var(--dim); font-weight: 500; transition: all .15s; }
.tab:hover { color: var(--fg2); background: var(--bg3); }
.tab.active { color: var(--accent); background: var(--bg); border: 1px solid var(--border); border-bottom-color: var(--bg); }
.content { padding: 24px; max-width: 1400px; margin: 0 auto; }
.panel { display: none; }
.panel.active { display: block; }

/* Cards */
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
.card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; }
.card .label { color: var(--dim); font-size: 12px; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px; }
.card .value { font-size: 28px; font-weight: 700; }
.card .value.green { color: var(--green); }
.card .value.red { color: var(--red); }
.card .value.accent { color: var(--accent); }
.card .value.orange { color: var(--orange); }

/* Status dot */
.status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.status-dot.on { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-dot.off { background: var(--red); }

/* Tables */
table { width: 100%; border-collapse: collapse; background: var(--bg2); border-radius: var(--radius); overflow: hidden; }
th { background: var(--bg3); text-align: left; padding: 10px 14px; font-weight: 600; color: var(--fg2); font-size: 12px; text-transform: uppercase; letter-spacing: .5px; }
td { padding: 10px 14px; border-top: 1px solid var(--border); }
tr:hover td { background: var(--bg3); }

/* Confidence bars */
.bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.bar-label { width: 80px; color: var(--fg2); font-weight: 500; text-transform: capitalize; }
.bar-track { flex: 1; height: 24px; background: var(--bg3); border-radius: 4px; overflow: hidden; position: relative; }
.bar-fill { height: 100%; border-radius: 4px; transition: width .5s; }
.bar-text { position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: 12px; font-weight: 600; color: var(--fg); }

/* Buttons */
.btn { padding: 6px 16px; border: none; border-radius: 4px; cursor: pointer; font-weight: 500; font-size: 13px; transition: opacity .15s; }
.btn:hover { opacity: .85; }
.btn-green { background: var(--green); color: #1a1b26; }
.btn-red { background: var(--red); color: #1a1b26; }
.btn-accent { background: var(--accent); color: #1a1b26; }

/* Log viewer */
#log-container { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 12px; line-height: 1.6; max-height: 70vh; overflow-y: auto; white-space: pre-wrap; word-break: break-all; color: var(--fg2); }
#log-container .log-line { padding: 1px 0; }
#log-container .log-error { color: var(--red); }
#log-container .log-warning { color: var(--orange); }
#log-container .log-info { color: var(--fg2); }

/* Config */
.config-group { margin-bottom: 24px; }
.config-group h3 { color: var(--accent); margin-bottom: 12px; font-size: 14px; }
.config-row { display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--border); }
.config-row label { width: 320px; font-family: monospace; color: var(--fg2); font-size: 13px; }
.config-row input { flex: 1; background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; padding: 6px 10px; color: var(--fg); font-family: monospace; font-size: 13px; }
.config-row input:focus { outline: none; border-color: var(--accent); }
.config-row .badge { font-size: 10px; padding: 2px 6px; border-radius: 3px; font-weight: 600; }
.badge-live { background: var(--green); color: #1a1b26; }
.badge-restart { background: var(--orange); color: #1a1b26; }
.badge-secret { background: var(--dim); color: var(--bg); }

/* Section box */
.section { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 24px; }
.section h2 { font-size: 16px; margin-bottom: 16px; color: var(--fg); }

/* Review detail */
.review-detail { background: var(--bg3); border-radius: var(--radius); padding: 16px; margin-top: 8px; }
.review-detail pre { font-family: monospace; font-size: 12px; color: var(--fg2); white-space: pre-wrap; }

/* Toast */
.toast { position: fixed; bottom: 24px; right: 24px; padding: 12px 20px; border-radius: var(--radius); font-weight: 500; z-index: 999; opacity: 0; transition: opacity .3s; }
.toast.show { opacity: 1; }
.toast.ok { background: var(--green); color: #1a1b26; }
.toast.err { background: var(--red); color: #1a1b26; }

/* Toolbar */
.toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.toolbar select, .toolbar input { background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; padding: 6px 10px; color: var(--fg); font-size: 13px; }

/* Responsive */
@media (max-width: 768px) {
  .header { flex-wrap: wrap; padding: 10px 16px; }
  .tabs { width: 100%; }
  .tab { flex: 1; text-align: center; padding: 8px 8px; font-size: 12px; }
  .cards { grid-template-columns: repeat(2, 1fr); }
  .content { padding: 16px; }
  .config-row { flex-wrap: wrap; }
  .config-row label { width: 100%; }
}
</style>
</head>
<body>

<div class="header">
  <h1>Paperless Organizer</h1>
  <span class="version" id="version-info"></span>
  <div class="tabs">
    <div class="tab active" onclick="switchTab('dashboard')">Dashboard</div>
    <div class="tab" onclick="switchTab('reviews')">Reviews</div>
    <div class="tab" onclick="switchTab('logs')">Logs</div>
    <div class="tab" onclick="switchTab('config')">Einstellungen</div>
  </div>
</div>

<div class="content">

<!-- DASHBOARD -->
<div class="panel active" id="panel-dashboard">
  <div class="cards" id="status-cards"></div>

  <div class="section">
    <h2>Verarbeitungs-Statistiken (30 Tage)</h2>
    <div class="cards" id="stats-cards"></div>
    <div id="top-errors"></div>
  </div>

  <div class="section">
    <h2>Confidence-Kalibrierung</h2>
    <div id="calibration-bars"></div>
  </div>
</div>

<!-- REVIEWS -->
<div class="panel" id="panel-reviews">
  <div class="toolbar">
    <span id="review-count" style="color:var(--dim)"></span>
    <button class="btn btn-accent" onclick="loadReviews()">Aktualisieren</button>
  </div>
  <table id="reviews-table">
    <thead><tr><th>ID</th><th>Dok-ID</th><th>Grund</th><th>Erstellt</th><th>Aktionen</th></tr></thead>
    <tbody id="reviews-body"></tbody>
  </table>
  <div id="review-detail-area"></div>
</div>

<!-- LOGS -->
<div class="panel" id="panel-logs">
  <div class="toolbar">
    <label style="color:var(--dim)">Zeilen:</label>
    <select id="log-lines" onchange="loadLog()">
      <option value="100">100</option>
      <option value="200" selected>200</option>
      <option value="500">500</option>
      <option value="1000">1000</option>
    </select>
    <label style="color:var(--dim);margin-left:12px">
      <input type="checkbox" id="log-live" onchange="toggleLive()" checked> Live-Stream
    </label>
    <label style="color:var(--dim);margin-left:12px">
      <input type="checkbox" id="log-autoscroll" checked> Auto-Scroll
    </label>
    <button class="btn btn-accent" onclick="loadLog()">Neu laden</button>
  </div>
  <div id="log-container"></div>
</div>

<!-- CONFIG -->
<div class="panel" id="panel-config">
  <div class="toolbar">
    <button class="btn btn-green" onclick="saveConfig()">Speichern</button>
    <span id="config-status" style="color:var(--dim)"></span>
  </div>
  <div id="config-area"></div>
</div>

</div>

<div class="toast" id="toast"></div>

<script>
// ── State ──
let currentTab = 'dashboard';
let autoRefreshTimer = null;
let sseSource = null;
let configData = [];

// ── Tab switching ──
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');
  document.querySelectorAll('.tab')[['dashboard','reviews','logs','config'].indexOf(tab)].classList.add('active');
  if (tab === 'dashboard') loadDashboard();
  else if (tab === 'reviews') loadReviews();
  else if (tab === 'logs') loadLog();
  else if (tab === 'config') loadConfig();
}

// ── Toast ──
function toast(msg, ok=true) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show ' + (ok ? 'ok' : 'err');
  setTimeout(() => t.className = 'toast', 3000);
}

// ── API helper ──
async function api(path, opts) {
  try {
    const r = await fetch('/api/' + path, opts);
    if (!r.ok) throw new Error(r.statusText);
    return await r.json();
  } catch(e) {
    console.error('API error:', path, e);
    return null;
  }
}

// ── Dashboard ──
async function loadDashboard() {
  const [status, stats, cal] = await Promise.all([
    api('status'), api('stats'), api('calibration')
  ]);
  if (status) renderStatus(status);
  if (stats) renderStats(stats);
  if (cal) renderCalibration(cal);
}

function renderStatus(s) {
  const dot = s.running ? '<span class="status-dot on"></span>Aktiv' : '<span class="status-dot off"></span>Gestoppt';
  document.getElementById('version-info').innerHTML = dot + (s.uptime !== '-' ? ' · ' + s.uptime : '');
  const cards = [
    {label: 'Zyklus', value: s.cycle, cls: 'accent'},
    {label: 'Laufzeit', value: s.uptime, cls: 'accent'},
    {label: 'Neue Dokumente', value: s.seen_new, cls: 'green'},
    {label: 'Verarbeitet', value: s.candidates, cls: 'green'},
    {label: 'Aktualisiert', value: s.updated, cls: 'green'},
    {label: 'Fehler', value: s.failed, cls: s.failed > 0 ? 'red' : 'green'},
    {label: 'Poll-Fehler', value: s.poll_errors, cls: s.poll_errors > 0 ? 'orange' : 'green'},
    {label: 'Reviews gelöst', value: s.reviews_resolved, cls: 'accent'},
  ];
  document.getElementById('status-cards').innerHTML = cards.map(c =>
    `<div class="card"><div class="label">${c.label}</div><div class="value ${c.cls}">${c.value}</div></div>`
  ).join('');
}

function renderStats(s) {
  const rate = s.success_rate != null ? s.success_rate.toFixed(1) + '%' : '-';
  const cards = [
    {label: 'Dokumente (30T)', value: s.total_docs, cls: 'accent'},
    {label: 'Erfolgsrate', value: rate, cls: 'green'},
    {label: 'Runs', value: s.total_runs, cls: 'accent'},
    {label: 'Offene Reviews', value: s.open_reviews, cls: s.open_reviews > 0 ? 'orange' : 'green'},
    {label: 'Regelbasiert', value: s.rule_based, cls: 'cyan'},
    {label: 'Prior-Fallback', value: s.prior_fallback, cls: 'accent'},
  ];
  document.getElementById('stats-cards').innerHTML = cards.map(c =>
    `<div class="card"><div class="label">${c.label}</div><div class="value ${c.cls}">${c.value}</div></div>`
  ).join('');
  // Top errors
  const errHtml = s.top_errors && s.top_errors.length > 0
    ? '<table><thead><tr><th>Fehler</th><th style="width:80px">Anzahl</th></tr></thead><tbody>' +
      s.top_errors.map(e => `<tr><td style="color:var(--red)">${esc(e.error)}</td><td>${e.count}</td></tr>`).join('') +
      '</tbody></table>'
    : '<p style="color:var(--dim)">Keine Fehler</p>';
  document.getElementById('top-errors').innerHTML = '<h3 style="margin:16px 0 8px;font-size:14px">Top-Fehler</h3>' + errHtml;
}

function renderCalibration(cal) {
  const levels = ['high', 'medium', 'low'];
  const colors = {'high': 'var(--green)', 'medium': 'var(--orange)', 'low': 'var(--red)'};
  let html = '';
  for (const lv of levels) {
    const d = cal[lv];
    if (!d) continue;
    const pct = d.accuracy || 0;
    html += `<div class="bar-row">
      <span class="bar-label">${lv}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${pct}%;background:${colors[lv]}"></div>
        <span class="bar-text">${pct}% (${d.total} Docs, ${d.corrected} korrigiert)</span>
      </div>
    </div>`;
  }
  document.getElementById('calibration-bars').innerHTML = html || '<p style="color:var(--dim)">Keine Daten</p>';
}

// ── Reviews ──
async function loadReviews() {
  const revs = await api('reviews');
  if (!revs) return;
  document.getElementById('review-count').textContent = revs.length + ' offene Reviews';
  const body = document.getElementById('reviews-body');
  if (revs.length === 0) {
    body.innerHTML = '<tr><td colspan="5" style="color:var(--dim);text-align:center">Keine offenen Reviews</td></tr>';
    return;
  }
  body.innerHTML = revs.map(r => `<tr>
    <td>${r.id}</td>
    <td><a href="#" onclick="showReview(${r.id});return false">#${r.doc_id}</a></td>
    <td>${esc(r.reason)}</td>
    <td>${r.created_at}</td>
    <td>
      <button class="btn btn-green" onclick="reviewAction(${r.id},'approve')">Approve</button>
      <button class="btn btn-red" onclick="reviewAction(${r.id},'reject')">Reject</button>
    </td>
  </tr>`).join('');
}

async function showReview(id) {
  const rev = await api('reviews/' + id);
  if (!rev) return;
  const area = document.getElementById('review-detail-area');
  const sug = rev.suggestion || {};
  area.innerHTML = `<div class="review-detail">
    <h3 style="margin-bottom:8px">Review #${rev.id} - Dokument #${rev.doc_id}</h3>
    <pre>${esc(JSON.stringify(sug, null, 2))}</pre>
    <div style="margin-top:12px">
      <button class="btn btn-green" onclick="reviewAction(${rev.id},'approve')">Approve</button>
      <button class="btn btn-red" onclick="reviewAction(${rev.id},'reject')">Reject</button>
    </div>
  </div>`;
}

async function reviewAction(id, action) {
  const r = await fetch('/api/reviews/' + id + '/' + action, {method: 'POST'});
  if (r.ok) {
    toast(action === 'approve' ? 'Review genehmigt' : 'Review verworfen');
    loadReviews();
    document.getElementById('review-detail-area').innerHTML = '';
  } else {
    toast('Fehler: ' + r.statusText, false);
  }
}

// ── Logs ──
function loadLog() {
  const lines = document.getElementById('log-lines').value;
  api('log?lines=' + lines).then(data => {
    if (!data) return;
    const container = document.getElementById('log-container');
    container.innerHTML = data.lines.map(l => `<div class="log-line ${logClass(l)}">${esc(l)}</div>`).join('');
    if (document.getElementById('log-autoscroll').checked) {
      container.scrollTop = container.scrollHeight;
    }
  });
  if (document.getElementById('log-live').checked) startSSE();
}

function logClass(line) {
  if (/error|fehler|exception|traceback/i.test(line)) return 'log-error';
  if (/warning|warnung/i.test(line)) return 'log-warning';
  return 'log-info';
}

function toggleLive() {
  if (document.getElementById('log-live').checked) startSSE();
  else stopSSE();
}

function startSSE() {
  stopSSE();
  sseSource = new EventSource('/api/log/stream');
  sseSource.onmessage = function(e) {
    const line = JSON.parse(e.data);
    const container = document.getElementById('log-container');
    const div = document.createElement('div');
    div.className = 'log-line ' + logClass(line);
    div.textContent = line;
    container.appendChild(div);
    // Limit displayed lines
    while (container.children.length > 2000) container.removeChild(container.firstChild);
    if (document.getElementById('log-autoscroll').checked) {
      container.scrollTop = container.scrollHeight;
    }
  };
}

function stopSSE() {
  if (sseSource) { sseSource.close(); sseSource = null; }
}

// ── Config ──
const CONFIG_GROUPS = {
  'Paperless-NGX': ['PAPERLESS_URL', 'PAPERLESS_TOKEN', 'OWNER_ID'],
  'LLM': ['LLM_URL', 'LLM_MODEL', 'LLM_API_KEY', 'LLM_SYSTEM_PROMPT', 'LLM_KEEP_ALIVE',
           'LLM_CONNECT_TIMEOUT', 'LLM_TIMEOUT', 'LLM_COMPACT_TIMEOUT', 'LLM_RETRY_COUNT',
           'LLM_MAX_TOKENS', 'LLM_COMPACT_MAX_TOKENS', 'LLM_TEMPERATURE',
           'LLM_VERIFY_ON_LOW_CONFIDENCE', 'LLM_FALLBACK_MODEL', 'LLM_FALLBACK_AFTER_ERRORS'],
  'Autopilot': ['AUTOPILOT_INTERVAL_SEC', 'AUTOPILOT_CONTEXT_REFRESH_CYCLES',
                'AUTOPILOT_START_WITH_AUTO_ORGANIZE', 'AUTOPILOT_RECHECK_ALL_ON_START',
                'AUTOPILOT_CLEANUP_EVERY_CYCLES', 'AUTOPILOT_DUPLICATE_SCAN_EVERY_CYCLES',
                'AUTOPILOT_REVIEW_RESOLVE_EVERY_CYCLES', 'AUTOPILOT_MAX_NEW_DOCS_PER_CYCLE',
                'QUIET_HOURS_START', 'QUIET_HOURS_END'],
  'Organizer': ['DEFAULT_DRY_RUN', 'AGENT_WORKERS', 'MAX_TAGS_PER_DOC', 'ENFORCE_TAG_TAXONOMY',
                'ALLOW_NEW_TAGS', 'ALLOW_NEW_CORRESPONDENTS', 'DELETE_UNUSED_CORRESPONDENTS',
                'ALLOW_NEW_STORAGE_PATHS', 'RECHECK_ALL_DOCS_IN_AUTO', 'AUTO_CLEANUP_AFTER_ORGANIZE'],
  'Learning': ['LEARNING_EXAMPLE_LIMIT', 'LEARNING_MAX_EXAMPLES', 'ENABLE_LEARNING_PRIORS',
               'LEARNING_PRIOR_MAX_HINTS', 'LEARNING_PRIOR_MIN_SAMPLES', 'LEARNING_PRIOR_MIN_RATIO',
               'LEARNING_PRIOR_ENABLE_TAG_SUGGESTION', 'RULE_BASED_MIN_SAMPLES', 'RULE_BASED_MIN_RATIO'],
  'Review & Tags': ['REVIEW_TAG_NAME', 'AUTO_APPLY_REVIEW_TAG', 'REVIEW_ON_MEDIUM_CONFIDENCE',
                     'AUTO_CREATE_TAXONOMY_TAGS', 'KEEP_UNUSED_TAXONOMY_TAGS', 'MAX_TOTAL_TAGS'],
  'Logging & Sonstiges': ['LOG_LEVEL', 'LIVE_WATCH_INTERVAL_SEC', 'LIVE_WATCH_CONTEXT_REFRESH_CYCLES',
                           'LIVE_WATCH_COMPACT_FIRST', 'ENABLE_WEB_HINTS', 'WEB_HINT_MAX_ENTITIES',
                           'SKIP_RECENT_LLM_ERRORS_MINUTES', 'SKIP_RECENT_LLM_ERRORS_THRESHOLD'],
};

const LIVE_KEYS = new Set([
  'LOG_LEVEL', 'AUTOPILOT_INTERVAL_SEC', 'AUTOPILOT_CLEANUP_EVERY_CYCLES',
  'AUTOPILOT_DUPLICATE_SCAN_EVERY_CYCLES', 'AUTOPILOT_REVIEW_RESOLVE_EVERY_CYCLES',
  'AUTOPILOT_MAX_NEW_DOCS_PER_CYCLE', 'MAX_TAGS_PER_DOC', 'LLM_TEMPERATURE',
  'LLM_TIMEOUT', 'LLM_MAX_TOKENS', 'LLM_KEEP_ALIVE', 'QUIET_HOURS_START', 'QUIET_HOURS_END',
]);

async function loadConfig() {
  configData = await api('config');
  if (!configData) return;
  const area = document.getElementById('config-area');
  const byKey = {};
  configData.forEach(c => byKey[c.key] = c);
  let html = '';
  for (const [group, keys] of Object.entries(CONFIG_GROUPS)) {
    const rows = keys.filter(k => byKey[k]).map(k => {
      const c = byKey[k];
      const badge = c.secret ? '<span class="badge badge-secret">SECRET</span>'
        : LIVE_KEYS.has(k) ? '<span class="badge badge-live">LIVE</span>'
        : '<span class="badge badge-restart">RESTART</span>';
      const disabled = c.secret ? 'disabled' : '';
      return `<div class="config-row">
        <label>${k} ${badge}</label>
        <input type="text" data-key="${k}" value="${esc(c.value)}" ${disabled}>
      </div>`;
    }).join('');
    if (rows) html += `<div class="config-group"><h3>${group}</h3>${rows}</div>`;
  }
  // Remaining keys not in any group
  const grouped = new Set(Object.values(CONFIG_GROUPS).flat());
  const remaining = configData.filter(c => !grouped.has(c.key));
  if (remaining.length) {
    html += '<div class="config-group"><h3>Weitere</h3>' + remaining.map(c => {
      const badge = c.secret ? '<span class="badge badge-secret">SECRET</span>'
        : '<span class="badge badge-restart">RESTART</span>';
      const disabled = c.secret ? 'disabled' : '';
      return `<div class="config-row">
        <label>${c.key} ${badge}</label>
        <input type="text" data-key="${c.key}" value="${esc(c.value)}" ${disabled}>
      </div>`;
    }).join('') + '</div>';
  }
  area.innerHTML = html;
}

async function saveConfig() {
  const inputs = document.querySelectorAll('#config-area input[data-key]:not([disabled])');
  const byKey = {};
  configData.forEach(c => byKey[c.key] = c.value);
  const changes = {};
  inputs.forEach(inp => {
    const k = inp.dataset.key;
    if (inp.value !== byKey[k]) changes[k] = inp.value;
  });
  if (Object.keys(changes).length === 0) {
    toast('Keine Änderungen');
    return;
  }
  const r = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(changes),
  });
  if (r.ok) {
    const data = await r.json();
    const restart = data.needs_restart || [];
    if (restart.length > 0) {
      toast('Gespeichert. Neustart nötig für: ' + restart.join(', '));
    } else {
      toast('Gespeichert & live angewendet');
    }
    loadConfig();
  } else {
    toast('Speichern fehlgeschlagen', false);
  }
}

// ── Helpers ──
function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Auto-refresh ──
function startAutoRefresh() {
  if (autoRefreshTimer) clearInterval(autoRefreshTimer);
  autoRefreshTimer = setInterval(() => {
    if (currentTab === 'dashboard') loadDashboard();
  }, 10000);
}

// ── Init ──
loadDashboard();
startAutoRefresh();
</script>
</body>
</html>"""
