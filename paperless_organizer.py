"""
Paperless-NGX Organizer - Unified Terminal App
Vereint: main.py, cleanup.py, cleanup_correspondents.py, cleanup_doctypes.py,
         batch_delete_tags.py, find_duplicates.py, fast_cleanup.py

Nutzt lokales LLM (Ollama/LM Studio/OpenAI-kompatibel) statt Claude CLI.
Interaktive Rich Terminal-UI mit Menuesystem.
"""

from __future__ import annotations

__version__ = "2.0.0"

import os
import sys
import json
import re
import time
import logging
import logging.handlers
import sqlite3
import threading
import difflib
import unicodedata
import signal
import requests
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode, urlparse, urlunparse
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.logging import RichHandler

load_dotenv()
console = Console()

# â”€â”€ Logging-Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "organizer.log")
STATE_DB_FILE = os.path.join(LOG_DIR, "organizer_state.db")
TAXONOMY_FILE = os.path.join(LOG_DIR, "taxonomy_tags.json")
LEARNING_PROFILE_FILE = os.path.join(LOG_DIR, "learning_profile.json")
LEARNING_EXAMPLES_FILE = os.path.join(LOG_DIR, "learning_examples.jsonl")

_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        RichHandler(console=console, show_path=False, markup=True, rich_tracebacks=True),
        _file_handler,
    ],
)
log = logging.getLogger("organizer")


class LocalStateDB:
    """Simple local SQLite storage for run/document history."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        self._harden_permissions()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _harden_permissions(self):
        # Best-effort: keep DB readable/writable only for local user account where supported.
        try:
            if os.path.exists(self.db_path):
                os.chmod(self.db_path, 0o600)
        except Exception:
            pass

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    action TEXT NOT NULL,
                    dry_run INTEGER NOT NULL,
                    llm_model TEXT,
                    llm_url TEXT,
                    summary_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    doc_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    title_before TEXT,
                    title_after TEXT,
                    tags_before TEXT,
                    tags_after TEXT,
                    correspondent_before TEXT,
                    correspondent_after TEXT,
                    error_text TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tag_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    doc_id INTEGER,
                    action TEXT NOT NULL,
                    tag_name TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    doc_id INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    suggestion_json TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            """)

    def start_run(self, action: str, dry_run: bool, llm_model: str, llm_url: str) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (started_at, action, dry_run, llm_model, llm_url)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now, action, int(dry_run), llm_model, llm_url),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, summary: dict):
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET ended_at = ?, summary_json = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), json.dumps(summary, ensure_ascii=False), run_id),
            )

    def record_document(self, run_id: int, doc_id: int, status: str,
                        document: dict, suggestion: dict | None = None, error: str = ""):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    run_id, doc_id, status, title_before, title_after,
                    tags_before, tags_after, correspondent_before, correspondent_after,
                    error_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    doc_id,
                    status,
                    document.get("title", ""),
                    (suggestion or {}).get("title", ""),
                    json.dumps(document.get("tags") or []),
                    json.dumps((suggestion or {}).get("tags") or []),
                    str(document.get("correspondent") or ""),
                    str((suggestion or {}).get("correspondent") or ""),
                    error[:500],
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def record_tag_event(self, run_id: int | None, doc_id: int | None, action: str, tag_name: str, detail: str):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tag_events (run_id, doc_id, action, tag_name, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    doc_id,
                    action,
                    tag_name,
                    detail[:500],
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def enqueue_review(self, run_id: int | None, doc_id: int, reason: str, suggestion: dict | None):
        now = datetime.now().isoformat(timespec="seconds")
        payload = json.dumps(suggestion or {}, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM review_queue WHERE doc_id = ? AND status = 'open' ORDER BY id DESC LIMIT 1",
                (doc_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE review_queue
                    SET run_id = ?, reason = ?, suggestion_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (run_id, reason[:800], payload, now, int(existing[0])),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO review_queue (run_id, doc_id, reason, suggestion_json, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'open', ?, ?)
                    """,
                    (run_id, doc_id, reason[:800], payload, now, now),
                )

    def list_open_reviews(self, limit: int = 50) -> list[dict]:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, doc_id, reason, created_at, updated_at
                FROM review_queue
                WHERE status = 'open'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_review_with_suggestion(self, review_id: int) -> dict | None:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, doc_id, suggestion_json FROM review_queue WHERE id = ?",
                (review_id,),
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def close_review(self, review_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE review_queue SET status = 'resolved', updated_at = ? WHERE id = ? AND status = 'open'",
                (datetime.now().isoformat(timespec="seconds"), review_id),
            )
            return cur.rowcount > 0

    def count_recent_document_statuses(self, doc_id: int, statuses: list[str], within_minutes: int) -> int:
        if not statuses or within_minutes <= 0:
            return 0
        since = (datetime.now() - timedelta(minutes=within_minutes)).isoformat(timespec="seconds")
        placeholders = ",".join("?" for _ in statuses)
        params = [doc_id, *statuses, since]
        query = (
            f"SELECT COUNT(*) FROM documents "
            f"WHERE doc_id = ? AND status IN ({placeholders}) AND created_at >= ?"
        )
        with self._lock, self._connect() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
            return int((row or [0])[0])

    def purge_old_runs(self, keep_days: int = 90) -> dict:
        """Delete runs and documents older than keep_days. Returns counts of deleted rows."""
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            # Get run IDs to delete
            old_run_ids = [
                row[0] for row in
                conn.execute("SELECT id FROM runs WHERE started_at < ?", (cutoff,)).fetchall()
            ]
            if not old_run_ids:
                return {"runs": 0, "documents": 0, "tag_events": 0, "reviews": 0}
            placeholders = ",".join("?" for _ in old_run_ids)
            docs_deleted = conn.execute(
                f"DELETE FROM documents WHERE run_id IN ({placeholders})", old_run_ids
            ).rowcount
            tags_deleted = conn.execute(
                f"DELETE FROM tag_events WHERE run_id IN ({placeholders})", old_run_ids
            ).rowcount
            reviews_deleted = conn.execute(
                "DELETE FROM review_queue WHERE status = 'resolved' AND updated_at < ?", (cutoff,)
            ).rowcount
            runs_deleted = conn.execute(
                f"DELETE FROM runs WHERE id IN ({placeholders})", old_run_ids
            ).rowcount
            return {"runs": runs_deleted, "documents": docs_deleted,
                    "tag_events": tags_deleted, "reviews": reviews_deleted}

    def generate_monthly_report(self, year: int, month: int) -> dict:
        """Monatlichen Report aus SQLite-Daten generieren."""
        start = f"{year:04d}-{month:02d}-01T00:00:00"
        if month == 12:
            end = f"{year + 1:04d}-01-01T00:00:00"
        else:
            end = f"{year:04d}-{month + 1:02d}-01T00:00:00"

        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row

            # Anzahl Laeufe
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM runs WHERE started_at >= ? AND started_at < ?",
                (start, end),
            ).fetchone()
            total_runs = row["cnt"] if row else 0

            # Verarbeitete Dokumente + Erfolgsquote
            rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM documents "
                "WHERE created_at >= ? AND created_at < ? GROUP BY status",
                (start, end),
            ).fetchall()
            status_counts = {r["status"]: r["cnt"] for r in rows}
            total_docs = sum(status_counts.values())
            success_docs = status_counts.get("ok", 0) + status_counts.get("applied", 0)
            success_rate = (success_docs / total_docs * 100) if total_docs > 0 else 0.0

            # Neue Korrespondenten (vorher leer, nachher gesetzt)
            new_corrs = conn.execute(
                "SELECT DISTINCT correspondent_after FROM documents "
                "WHERE created_at >= ? AND created_at < ? "
                "AND (correspondent_before IS NULL OR correspondent_before = '') "
                "AND correspondent_after IS NOT NULL AND correspondent_after != ''",
                (start, end),
            ).fetchall()
            new_correspondents = [r["correspondent_after"] for r in new_corrs]

            # Geloeschte Tags
            deleted_tags = conn.execute(
                "SELECT tag_name, COUNT(*) AS cnt FROM tag_events "
                "WHERE action = 'delete' AND created_at >= ? AND created_at < ? "
                "GROUP BY tag_name ORDER BY cnt DESC",
                (start, end),
            ).fetchall()
            deleted_tag_list = [{"name": r["tag_name"], "count": r["cnt"]} for r in deleted_tags]

            # Offene Reviews (aktueller Stand, nicht zeitraumgebunden)
            open_reviews = conn.execute(
                "SELECT id, doc_id, reason, updated_at FROM review_queue "
                "WHERE status = 'open' ORDER BY updated_at DESC LIMIT 20",
            ).fetchall()
            open_review_list = [dict(r) for r in open_reviews]

            # Fehler-Uebersicht
            errors = conn.execute(
                "SELECT error_text, COUNT(*) AS cnt FROM documents "
                "WHERE status = 'error' AND created_at >= ? AND created_at < ? "
                "AND error_text != '' "
                "GROUP BY error_text ORDER BY cnt DESC LIMIT 10",
                (start, end),
            ).fetchall()
            error_list = [{"error": r["error_text"], "count": r["cnt"]} for r in errors]

        return {
            "year": year,
            "month": month,
            "total_runs": total_runs,
            "total_docs": total_docs,
            "success_docs": success_docs,
            "success_rate": success_rate,
            "status_counts": status_counts,
            "new_correspondents": new_correspondents,
            "deleted_tags": deleted_tag_list,
            "open_reviews": open_review_list,
            "errors": error_list,
        }



# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CONFIG
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# --- LLM (lokaler Server: Ollama/LM Studio/OpenAI-kompatibel) ---
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:1234/v1/chat/completions").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", "").strip()
LLM_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "").strip()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_CONNECT_TIMEOUT = int(os.getenv("LLM_CONNECT_TIMEOUT", "8"))
LLM_COMPACT_TIMEOUT = int(os.getenv("LLM_COMPACT_TIMEOUT", "60"))
LLM_COMPACT_TIMEOUT_RETRY = int(os.getenv("LLM_COMPACT_TIMEOUT_RETRY", str(max(LLM_COMPACT_TIMEOUT + 25, 75))))
LLM_RETRY_COUNT = int(os.getenv("LLM_RETRY_COUNT", "2"))
LLM_COMPACT_MAX_TOKENS = int(os.getenv("LLM_COMPACT_MAX_TOKENS", "320"))
LLM_COMPACT_PROMPT_MAX_PATHS = int(os.getenv("LLM_COMPACT_PROMPT_MAX_PATHS", "20"))
LLM_COMPACT_PROMPT_MAX_TAGS = int(os.getenv("LLM_COMPACT_PROMPT_MAX_TAGS", "24"))
LLM_VERIFY_ON_LOW_CONFIDENCE = os.getenv("LLM_VERIFY_ON_LOW_CONFIDENCE", "0").strip().lower() in ("1", "true", "yes", "on")

# --- Paperless-NGX ---
OWNER_ID = int(os.getenv("OWNER_ID", "4"))

# --- Erlaubte Dokumenttypen (20 Stueck) ---
ALLOWED_DOC_TYPES = [
    "Vertrag", "Rechnung", "Bescheinigung", "Information", "Kontoauszug",
    "Zeugnis", "Angebot", "Kuendigung", "Mahnung", "Versicherungspolice",
    "Steuerbescheid", "Arztbericht", "Gehaltsabrechnung", "Bestellung",
    "Korrespondenz", "Dokumentation", "Lizenz", "Formular", "Urkunde",
    "Bewerbung",
]

# --- Tag-Whitelist (nicht loeschen) ---
TAG_WHITELIST = {"kfz", "service", "termin", "aufhebungsvertrag", "ausbildung", "fachinformatiker"}

# --- Tag-Loeschregeln ---
TAG_DELETE_THRESHOLD = int(os.getenv("TAG_DELETE_THRESHOLD", "0"))
TAG_ENGLISH_THRESHOLD = int(os.getenv("TAG_ENGLISH_THRESHOLD", "0"))
NON_TAXONOMY_DELETE_THRESHOLD = int(os.getenv("NON_TAXONOMY_DELETE_THRESHOLD", "5"))
DELETE_USED_TAGS = os.getenv("DELETE_USED_TAGS", "0").strip().lower() in ("1", "true", "yes", "on")

# --- Korrespondenten-Dedupe (generisch, konservativ) ---
CORRESPONDENT_LEGAL_TOKENS = {
    "gmbh", "ag", "mbh", "kg", "kgaa", "ug", "eg", "e", "ev", "e.v", "ggmbh",
    "inc", "ltd", "llc", "corp", "co", "company", "corporation", "sarl", "sa",
}
CORRESPONDENT_STOPWORDS = {
    "der", "die", "das", "und", "im", "in", "am", "an", "auf", "mit", "von", "zu", "zum", "zur",
    "des", "dem", "den", "for", "of", "the", "via", "team", "support",
}

# --- Tag-Farbpalette (20 Farben) ---
TAG_COLORS = [
    "#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#2980b9", "#27ae60", "#c0392b",
    "#8e44ad", "#16a085", "#d35400", "#2c3e50", "#f1c40f",
    "#e91e63", "#00bcd4", "#ff5722", "#607d8b", "#795548",
]

# --- Worker / Retry ---
MAX_WORKERS = 5
MAX_RETRIES = 3
AGENT_WORKERS = int(os.getenv("AGENT_WORKERS", "1"))

WRITE_LOCK = threading.Lock()

# --- Governance / Automation ---
MAX_TAGS_PER_DOC = int(os.getenv("MAX_TAGS_PER_DOC", "4"))
ALLOW_NEW_TAGS = os.getenv("ALLOW_NEW_TAGS", "0").strip().lower() in ("1", "true", "yes", "on")
ALLOW_NEW_STORAGE_PATHS = os.getenv("ALLOW_NEW_STORAGE_PATHS", "0").strip().lower() in ("1", "true", "yes", "on")
AUTO_CLEANUP_AFTER_ORGANIZE = os.getenv("AUTO_CLEANUP_AFTER_ORGANIZE", "1").strip().lower() in ("1", "true", "yes", "on")
TAG_MATCH_THRESHOLD = float(os.getenv("TAG_MATCH_THRESHOLD", "0.88"))
MAX_PROMPT_TAG_CHOICES = int(os.getenv("MAX_PROMPT_TAG_CHOICES", "120"))
ENFORCE_TAG_TAXONOMY = os.getenv("ENFORCE_TAG_TAXONOMY", "1").strip().lower() in ("1", "true", "yes", "on")
AUTO_CREATE_TAXONOMY_TAGS = os.getenv("AUTO_CREATE_TAXONOMY_TAGS", "0").strip().lower() in ("1", "true", "yes", "on")
MAX_TOTAL_TAGS = int(os.getenv("MAX_TOTAL_TAGS", "100"))
KEEP_UNUSED_TAXONOMY_TAGS = os.getenv("KEEP_UNUSED_TAXONOMY_TAGS", "1").strip().lower() in ("1", "true", "yes", "on")
RECHECK_ALL_DOCS_IN_AUTO = os.getenv("RECHECK_ALL_DOCS_IN_AUTO", "0").strip().lower() in ("1", "true", "yes", "on")
REVIEW_TAG_NAME = os.getenv("REVIEW_TAG_NAME", "Manuell-Pruefen")
AUTO_APPLY_REVIEW_TAG = os.getenv("AUTO_APPLY_REVIEW_TAG", "1").strip().lower() in ("1", "true", "yes", "on")
REVIEW_ON_MEDIUM_CONFIDENCE = os.getenv("REVIEW_ON_MEDIUM_CONFIDENCE", "0").strip().lower() in ("1", "true", "yes", "on")
USE_ARCHIVE_SERIAL_NUMBER = os.getenv("USE_ARCHIVE_SERIAL_NUMBER", "0").strip().lower() in ("1", "true", "yes", "on")
CORRESPONDENT_MATCH_THRESHOLD = float(os.getenv("CORRESPONDENT_MATCH_THRESHOLD", "0.86"))
ALLOW_NEW_CORRESPONDENTS = os.getenv("ALLOW_NEW_CORRESPONDENTS", "0").strip().lower() in ("1", "true", "yes", "on")
DELETE_UNUSED_CORRESPONDENTS = os.getenv("DELETE_UNUSED_CORRESPONDENTS", "0").strip().lower() in ("1", "true", "yes", "on")
CORRESPONDENT_MERGE_MIN_GROUP_DOCS = int(os.getenv("CORRESPONDENT_MERGE_MIN_GROUP_DOCS", "2"))
CORRESPONDENT_MERGE_MIN_NAME_SIMILARITY = float(os.getenv("CORRESPONDENT_MERGE_MIN_NAME_SIMILARITY", "0.96"))
LEARNING_EXAMPLE_LIMIT = int(os.getenv("LEARNING_EXAMPLE_LIMIT", "3"))
LEARNING_MAX_EXAMPLES = int(os.getenv("LEARNING_MAX_EXAMPLES", "2000"))
ENABLE_LEARNING_PRIORS = os.getenv("ENABLE_LEARNING_PRIORS", "1").strip().lower() in ("1", "true", "yes", "on")
LEARNING_PRIOR_MAX_HINTS = int(os.getenv("LEARNING_PRIOR_MAX_HINTS", "2"))
LEARNING_PRIOR_MIN_SAMPLES = int(os.getenv("LEARNING_PRIOR_MIN_SAMPLES", "3"))
LEARNING_PRIOR_MIN_RATIO = float(os.getenv("LEARNING_PRIOR_MIN_RATIO", "0.70"))
LEARNING_PRIOR_ENABLE_TAG_SUGGESTION = os.getenv("LEARNING_PRIOR_ENABLE_TAG_SUGGESTION", "0").strip().lower() in ("1", "true", "yes", "on")
LIVE_WATCH_INTERVAL_SEC = int(os.getenv("LIVE_WATCH_INTERVAL_SEC", "45"))
LIVE_WATCH_CONTEXT_REFRESH_CYCLES = int(os.getenv("LIVE_WATCH_CONTEXT_REFRESH_CYCLES", "5"))
LIVE_WATCH_COMPACT_FIRST = os.getenv("LIVE_WATCH_COMPACT_FIRST", "1").strip().lower() in ("1", "true", "yes", "on")
MAX_CONTEXT_EMPLOYER_HINTS = int(os.getenv("MAX_CONTEXT_EMPLOYER_HINTS", "20"))
WORK_CORR_EMPLOYER_MIN_DOCS = int(os.getenv("WORK_CORR_EMPLOYER_MIN_DOCS", "8"))
AUTOPILOT_INTERVAL_SEC = int(os.getenv("AUTOPILOT_INTERVAL_SEC", "45"))
AUTOPILOT_CONTEXT_REFRESH_CYCLES = int(os.getenv("AUTOPILOT_CONTEXT_REFRESH_CYCLES", "5"))
AUTOPILOT_START_WITH_AUTO_ORGANIZE = os.getenv("AUTOPILOT_START_WITH_AUTO_ORGANIZE", "1").strip().lower() in ("1", "true", "yes", "on")
AUTOPILOT_RECHECK_ALL_ON_START = os.getenv("AUTOPILOT_RECHECK_ALL_ON_START", "0").strip().lower() in ("1", "true", "yes", "on")
AUTOPILOT_CLEANUP_EVERY_CYCLES = int(os.getenv("AUTOPILOT_CLEANUP_EVERY_CYCLES", "10"))
AUTOPILOT_DUPLICATE_SCAN_EVERY_CYCLES = int(os.getenv("AUTOPILOT_DUPLICATE_SCAN_EVERY_CYCLES", "0"))
AUTOPILOT_REVIEW_RESOLVE_EVERY_CYCLES = int(os.getenv("AUTOPILOT_REVIEW_RESOLVE_EVERY_CYCLES", "15"))
AUTOPILOT_MAX_NEW_DOCS_PER_CYCLE = int(os.getenv("AUTOPILOT_MAX_NEW_DOCS_PER_CYCLE", "25"))
WATCH_RECONNECT_ERROR_THRESHOLD = int(os.getenv("WATCH_RECONNECT_ERROR_THRESHOLD", "3"))
WATCH_ERROR_BACKOFF_BASE_SEC = int(os.getenv("WATCH_ERROR_BACKOFF_BASE_SEC", "10"))
WATCH_ERROR_BACKOFF_MAX_SEC = int(os.getenv("WATCH_ERROR_BACKOFF_MAX_SEC", "180"))
SKIP_RECENT_LLM_ERRORS_MINUTES = int(os.getenv("SKIP_RECENT_LLM_ERRORS_MINUTES", "240"))
SKIP_RECENT_LLM_ERRORS_THRESHOLD = int(os.getenv("SKIP_RECENT_LLM_ERRORS_THRESHOLD", "1"))

# --- Optional web hints for unknown brands/entities ---
ENABLE_WEB_HINTS = os.getenv("ENABLE_WEB_HINTS", "0").strip().lower() in ("1", "true", "yes", "on")
WEB_HINT_TIMEOUT = int(os.getenv("WEB_HINT_TIMEOUT", "6"))
WEB_HINT_MAX_ENTITIES = int(os.getenv("WEB_HINT_MAX_ENTITIES", "2"))
WEB_HINT_CACHE: dict[str, str] = {}
WEB_HINT_CACHE_LOCK = threading.Lock()

KNOWN_BRAND_HINTS = {
    "elevenlabs": "ElevenLabs ist ein KI-Voice/Audio SaaS Anbieter (Abo-Dienst, meist privat/IT/SaaS).",
    "github": "GitHub ist ein Entwickler- und SaaS-Dienst (oft IT/SaaS, nicht automatisch Arbeit).",
    "jetbrains": "JetBrains ist ein Software-Aboanbieter (IT/SaaS).",
    "openai": "OpenAI ist ein KI/SaaS Anbieter (IT/SaaS).",
    "google cloud": "Google Cloud ist ein Cloud/SaaS Anbieter (normalerweise privat/IT/Finanzen, nicht Arbeitgeber).",
    "google": "Google ist ein Technologieanbieter (Kontext pruefen, nicht automatisch Arbeitgeber).",
    "microsoft": "Microsoft ist ein Software/Cloud Anbieter (Azure, M365 etc.).",
}

EMPLOYER_HINTS = {"wbs training ag", "wbs", "msg systems ag", "msg"}

CORRESPONDENT_ALIASES = {
    "ihk dresden": "IHK Dresden",
    "ihk thresien": "IHK Dresden",
    "ihk dresen": "IHK Dresden",
    "industrie- und handelskammer dresden": "IHK Dresden",
    "industrie und handelskammer dresden": "IHK Dresden",
    "industrie und handelskammer thresien": "IHK Dresden",
    "ihk": "IHK Dresden",
    "wbs training ag": "WBS TRAINING AG",
    "msg systems ag": "msg systems ag",
    "google cloud": "Google Cloud",
}

SPELLING_FIXES = {
    "thresien": "dresden",
    "dresen": "dresden",
    "drseden": "dresden",
    "industriekammer": "industrie- und handelskammer",
    "handelskammer": "handelskammer",
    "kuendigung": "kuendigung",
}

TITLE_SPELLING_FIXES = {
    "thresien": "Dresden",
    "dresen": "Dresden",
    "drseden": "Dresden",
    "industriekammer": "Industrie- und Handelskammer",
}

VENDOR_GUARDRAILS = {
    "google cloud": {
        "correspondent": "Google Cloud",
        "path_preferences": ["Freizeit/IT", "Finanzen/Rechnungen", "Finanzen", "IT"],
    },
    "github": {
        "correspondent": "GitHub",
        "path_preferences": ["Freizeit/IT", "Finanzen/Rechnungen", "Finanzen", "IT"],
    },
    "elevenlabs": {
        "correspondent": "ElevenLabs",
        "path_preferences": ["Freizeit/IT", "Finanzen/Rechnungen", "Finanzen", "IT"],
    },
    "jetbrains": {
        "correspondent": "JetBrains",
        "path_preferences": ["Freizeit/IT", "Finanzen/Rechnungen", "Finanzen", "IT"],
    },
    "openai": {
        "correspondent": "OpenAI",
        "path_preferences": ["Freizeit/IT", "Finanzen/Rechnungen", "Finanzen", "IT"],
    },
}

PRIVATE_VEHICLE_HINTS = ["vw polo", "volkswagen polo", "polo", "golf polo"]
COMPANY_VEHICLE_HINTS = ["toyota"]
HEALTH_HINTS = [
    "allergie", "allergen", "arzt", "klinikum", "testzentrum", "sars-cov-2", "covid",
    "krankenkasse", "arbeitsunfaehigkeit", "arbeitsunfÃ¤higkeit", "krankschreibung", "anamnese",
]
SCHOOL_HINTS = [
    "berufliches schulzentrum", "oberschule", "abschlusszeugnis", "zeugnis", "pruefung",
    "prÃ¼fung", "fahrschule", "fahrerlaubnispruefung", "fahrerlaubnisprÃ¼fung", "anmeldebestaetigung",
    "anmeldebestÃ¤tigung",
]
EVENT_TICKET_HINTS = [
    "event ticket", "openair", "konzert", "festival", "ticket.io", "eintrittskarte",
    "party", "birthday party", "club", "veranstaltung",
]
TRANSPORT_TICKET_HINTS = [
    "fahrkarte", "deutschlandticket", "deutsche bahn", "bahn", "verkehrsverbund", "omnibus", "bus",
    "zugticket", "bahnticket",
]


def _validate_config():
    """Validate configuration at startup and warn about potential issues."""
    warnings = []
    if not os.getenv("PAPERLESS_URL"):
        warnings.append("PAPERLESS_URL nicht gesetzt - Verbindung wird fehlschlagen")
    if not os.getenv("PAPERLESS_TOKEN"):
        warnings.append("PAPERLESS_TOKEN nicht gesetzt - Authentifizierung wird fehlschlagen")
    if LLM_TEMPERATURE < 0.0 or LLM_TEMPERATURE > 2.0:
        warnings.append(f"LLM_TEMPERATURE={LLM_TEMPERATURE} ausserhalb sinnvollem Bereich (0.0-2.0)")
    if LLM_TIMEOUT < 10:
        warnings.append(f"LLM_TIMEOUT={LLM_TIMEOUT}s sehr kurz - Timeouts wahrscheinlich")
    if MAX_TAGS_PER_DOC < 1 or MAX_TAGS_PER_DOC > 20:
        warnings.append(f"MAX_TAGS_PER_DOC={MAX_TAGS_PER_DOC} ausserhalb sinnvollem Bereich (1-20)")
    if LEARNING_PRIOR_MIN_RATIO < 0.3 or LEARNING_PRIOR_MIN_RATIO > 1.0:
        warnings.append(f"LEARNING_PRIOR_MIN_RATIO={LEARNING_PRIOR_MIN_RATIO} ausserhalb sinnvollem Bereich (0.3-1.0)")
    if CORRESPONDENT_MATCH_THRESHOLD < 0.5 or CORRESPONDENT_MATCH_THRESHOLD > 1.0:
        warnings.append(f"CORRESPONDENT_MATCH_THRESHOLD={CORRESPONDENT_MATCH_THRESHOLD} ausserhalb sinnvollem Bereich (0.5-1.0)")
    if AUTOPILOT_INTERVAL_SEC < 5:
        warnings.append(f"AUTOPILOT_INTERVAL_SEC={AUTOPILOT_INTERVAL_SEC}s zu kurz - Server-Ueberlastung moeglich")
    url = os.getenv("LLM_URL", "")
    if url and not url.startswith(("http://", "https://")):
        warnings.append(f"LLM_URL='{url}' hat kein http(s):// Prefix")
    for w in warnings:
        log.warning(f"[yellow]Config:[/yellow] {w}")
    return len(warnings) == 0


_validate_config()


@dataclass
class DecisionContext:
    """Collected runtime context for better document decisions."""
    employer_names: set[str] = field(default_factory=set)          # normalized names
    provider_names: set[str] = field(default_factory=set)          # normalized keys
    top_work_paths: list[str] = field(default_factory=list)
    top_private_paths: list[str] = field(default_factory=list)
    profile_employment_lines: list[str] = field(default_factory=list)
    profile_private_vehicles: list[str] = field(default_factory=list)
    profile_company_vehicles: list[str] = field(default_factory=list)
    profile_context_text: str = ""
    notes: list[str] = field(default_factory=list)


def _safe_iso_date(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 10:
        candidate = text[:10]
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate
        except ValueError:
            return ""
    return ""


class LearningProfile:
    """Persistent lightweight learning profile for jobs/vehicles and stable context."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.data: dict = {}
        self.load()

    def _default_data(self) -> dict:
        return {
            "owner": "Edgar Richter",
            "jobs": [
                {"company": "msg systems ag", "start": "2022-01-01", "end": "2025-07-31", "source": "seed"},
                {"company": "WBS TRAINING AG", "start": "2025-08-01", "end": "", "source": "seed"},
            ],
            "vehicles": {
                "private": ["VW Polo"],
                "company": ["Toyota"],
            },
            "notes": ["DRK-Mitglied", "AOK PLUS"],
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        }

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self.data = raw
                else:
                    self.data = self._default_data()
            except Exception:
                self.data = self._default_data()
        else:
            self.data = self._default_data()
            self.save()

        self.data.setdefault("jobs", [])
        self.data.setdefault("vehicles", {})
        self.data["vehicles"].setdefault("private", [])
        self.data["vehicles"].setdefault("company", [])
        self.data.setdefault("notes", [])
        self.data.setdefault("owner", "Edgar Richter")
        self.data.setdefault("last_updated", datetime.now().isoformat(timespec="seconds"))

    def save(self):
        with self._lock:
            self.data["last_updated"] = datetime.now().isoformat(timespec="seconds")
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

    def employer_names(self) -> set[str]:
        names = set()
        for job in self.data.get("jobs", []):
            company = _normalize_text(str(job.get("company", "")))
            if not company:
                continue
            names.add(_normalize_tag_name(company))
        return names

    def private_vehicle_hints(self) -> list[str]:
        hints = list(PRIVATE_VEHICLE_HINTS)
        hints.extend(self.data.get("vehicles", {}).get("private", []))
        return list(dict.fromkeys(_normalize_tag_name(h) for h in hints if _normalize_text(h)))

    def company_vehicle_hints(self) -> list[str]:
        hints = list(COMPANY_VEHICLE_HINTS)
        hints.extend(self.data.get("vehicles", {}).get("company", []))
        return list(dict.fromkeys(_normalize_tag_name(h) for h in hints if _normalize_text(h)))

    def employment_lines(self) -> list[str]:
        jobs = self.data.get("jobs", [])
        normalized = []
        for job in jobs:
            company = _normalize_text(str(job.get("company", "")))
            if not company:
                continue
            start = _safe_iso_date(str(job.get("start", "")))
            end = _safe_iso_date(str(job.get("end", "")))
            if start and end:
                line = f"{start} bis {end}: {company}"
            elif start:
                line = f"seit {start}: {company}"
            else:
                line = company
            normalized.append((start or "0000-00-00", line))
        normalized.sort(key=lambda x: x[0], reverse=True)
        return [line for _, line in normalized[:10]]

    def prompt_context_text(self) -> str:
        owner = _normalize_text(str(self.data.get("owner", "Edgar Richter")))
        jobs = "; ".join(self.employment_lines()) or "keine bekannten Beschaeftigungen"
        private_vehicles = ", ".join(self.data.get("vehicles", {}).get("private", [])) or "keine"
        company_vehicles = ", ".join(self.data.get("vehicles", {}).get("company", [])) or "keine"
        notes = ", ".join(_normalize_text(n) for n in self.data.get("notes", []) if _normalize_text(n)) or "keine"
        if len(notes) > 240:
            notes = notes[:240] + "..."
        return (
            f"Person: {owner}.\n"
            f"Beschaeftigungsverlauf: {jobs}.\n"
            f"Privatfahrzeuge: {private_vehicles}.\n"
            f"Firmenfahrzeuge: {company_vehicles}.\n"
            f"Weitere Hinweise: {notes}."
        )

    @staticmethod
    def _extract_vehicle_candidates(text: str) -> list[str]:
        if not text:
            return []
        cleaned = _strip_diacritics(text.lower())
        out = []
        has_model_brand = set()

        pattern_with_model = r"\b(toyota|volkswagen|vw|audi|bmw|mercedes|skoda|ford|renault|opel|seat|tesla|hyundai|kia)\s+([a-z0-9\-]{2,20})\b"
        for m in re.finditer(pattern_with_model, cleaned):
            brand = m.group(1)
            model = m.group(2)
            if brand == "vw":
                brand = "volkswagen"
            has_model_brand.add(brand)
            candidate = _normalize_text(f"{brand} {model}").title()
            if candidate and candidate not in out:
                out.append(candidate)

        pattern_brand_only = r"\b(toyota|volkswagen|vw|audi|bmw|mercedes|skoda|ford|renault|opel|seat|tesla|hyundai|kia)\b"
        for m in re.finditer(pattern_brand_only, cleaned):
            brand = m.group(1)
            if brand == "vw":
                brand = "volkswagen"
            if brand in has_model_brand:
                continue
            candidate = _normalize_text(brand).title()
            if candidate and candidate not in out:
                out.append(candidate)
        return out[:8]

    def _learn_job(self, company: str, start_date: str):
        company_clean = _normalize_text(company)
        if not company_clean:
            return
        jobs = self.data.get("jobs", [])
        wanted_norm = _normalize_tag_name(company_clean)
        existing = None
        for job in jobs:
            if _normalize_tag_name(str(job.get("company", ""))) == wanted_norm:
                existing = job
                break

        if existing:
            if start_date:
                old_start = _safe_iso_date(str(existing.get("start", "")))
                if not old_start or start_date < old_start:
                    existing["start"] = start_date
            if not existing.get("company"):
                existing["company"] = company_clean
            return

        jobs.append({"company": company_clean, "start": start_date, "end": "", "source": "auto"})
        self.data["jobs"] = jobs

        # If a new company starts, close the previously open job (if present).
        if start_date:
            new_start = datetime.strptime(start_date, "%Y-%m-%d")
            for job in jobs:
                same = _normalize_tag_name(str(job.get("company", ""))) == wanted_norm
                if same:
                    continue
                old_end = _safe_iso_date(str(job.get("end", "")))
                old_start = _safe_iso_date(str(job.get("start", "")))
                if old_end or not old_start:
                    continue
                try:
                    old_start_dt = datetime.strptime(old_start, "%Y-%m-%d")
                except ValueError:
                    continue
                if old_start_dt <= new_start:
                    job["end"] = (new_start - timedelta(days=1)).strftime("%Y-%m-%d")

        if len(jobs) > 25:
            jobs.sort(
                key=lambda j: _safe_iso_date(str(j.get("start", ""))) or "0000-00-00",
                reverse=True,
            )
            self.data["jobs"] = jobs[:25]

    def _learn_vehicle(self, candidates: list[str], company_vehicle: bool):
        if not candidates:
            return
        vehicle_key = "company" if company_vehicle else "private"
        target = self.data.setdefault("vehicles", {}).setdefault(vehicle_key, [])
        known_norm = {_normalize_tag_name(v) for v in target}
        for cand in candidates:
            if _normalize_tag_name(cand) in known_norm:
                continue
            target.append(cand)
            known_norm.add(_normalize_tag_name(cand))

    def learn_from_document(self, document: dict, suggestion: dict):
        text = " ".join([
            str(document.get("title", "")),
            str(document.get("original_file_name", "")),
            str(document.get("content", ""))[:6000],
            str(suggestion.get("title", "")),
            str(suggestion.get("correspondent", "")),
            str(suggestion.get("storage_path", "")),
            " ".join(str(t) for t in (suggestion.get("tags") or [])),
        ]).lower()

        created = _safe_iso_date(str(document.get("created", "")))
        corr = _normalize_text(str(suggestion.get("correspondent", "")))
        path_lower = _normalize_text(str(suggestion.get("storage_path", ""))).lower()
        doc_type_lower = _normalize_text(str(suggestion.get("document_type", ""))).lower()
        tags_lower = {_normalize_tag_name(str(t)) for t in (suggestion.get("tags") or [])}

        employment_keywords = (
            "arbeitsvertrag",
            "arbeitsverhaltnis",
            "arbeitsverhältnis",
            "eintritt",
            "anstellung",
            "arbeitsbeginn",
            "gehaltsabrechnung",
        )
        corr_norm = _normalize_tag_name(corr)
        known_employer = corr_norm in EMPLOYER_HINTS or corr_norm in self.employer_names()
        explicit_employment = any(k in _strip_diacritics(text) for k in employment_keywords)
        strong_doc_type = doc_type_lower in {"vertrag", "gehaltsabrechnung", "zeugnis"}
        has_work_tag = ("arbeit" in tags_lower or "wbs" in tags_lower or "msg" in tags_lower)
        if corr and (
            (known_employer and path_lower.startswith("arbeit/") and (strong_doc_type or has_work_tag))
            or (explicit_employment and (strong_doc_type or path_lower.startswith("arbeit/")))
        ):
            self._learn_job(corr, created)

        vehicle_keywords = ("fahrzeug", "firmenwagen", "dienstwagen", "kfz", "zulassungsbescheinigung", "kennzeichen", "leasing")
        if any(k in _strip_diacritics(text) for k in vehicle_keywords):
            candidates = self._extract_vehicle_candidates(text)
            if candidates:
                company_signal = path_lower.startswith("arbeit/") or ("firmenwagen" in text) or ("dienstwagen" in text)
                private_signal = ("privatwagen" in text) or ("kfz-versicherung" in text)
                if company_signal and not private_signal:
                    self._learn_vehicle(candidates, company_vehicle=True)
                elif private_signal and not company_signal:
                    self._learn_vehicle(candidates, company_vehicle=False)
                else:
                    self._learn_vehicle(candidates, company_vehicle=path_lower.startswith("arbeit/"))


class LearningExamples:
    """Persistent few-shot memory for small models (JSONL)."""

    def __init__(self, path: str, max_examples: int = LEARNING_MAX_EXAMPLES):
        self.path = path
        self.max_examples = max(100, int(max_examples or 100))
        self._lock = threading.Lock()
        self._examples: list[dict] = []
        self.load()

    def load(self):
        self._examples = []
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        self._examples.append(row)
            if len(self._examples) > self.max_examples:
                self._examples = self._examples[-self.max_examples:]
        except Exception:
            self._examples = []

    @staticmethod
    def _tokens(text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z0-9]{3,}", _strip_diacritics((text or "").lower()))
        stop = {"und", "der", "die", "das", "von", "mit", "fuer", "for", "the", "and", "ein", "eine"}
        return {w for w in words if w not in stop}

    def append(self, document: dict, suggestion: dict):
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "doc_title": _normalize_text(str(document.get("title", ""))),
            "filename": _normalize_text(str(document.get("original_file_name", ""))),
            "correspondent": _normalize_text(str(suggestion.get("correspondent", ""))),
            "document_type": _normalize_text(str(suggestion.get("document_type", ""))),
            "storage_path": _normalize_text(str(suggestion.get("storage_path", ""))),
            "tags": [_normalize_text(str(t)) for t in (suggestion.get("tags") or []) if _normalize_text(str(t))],
        }
        if not row["correspondent"] and not row["document_type"] and not row["storage_path"]:
            return

        with self._lock:
            self._examples.append(row)
            if len(self._examples) > self.max_examples:
                self._examples = self._examples[-self.max_examples:]
            with open(self.path, "w", encoding="utf-8") as f:
                for entry in self._examples:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def select(self, document: dict, limit: int = LEARNING_EXAMPLE_LIMIT) -> list[dict]:
        if not self._examples:
            return []
        query = " ".join([
            str(document.get("title", "")),
            str(document.get("original_file_name", "")),
            str(document.get("content", ""))[:1200],
        ])
        q_tokens = self._tokens(query)
        if not q_tokens:
            return []

        scored = []
        for entry in self._examples[-500:]:
            text = " ".join([
                str(entry.get("doc_title", "")),
                str(entry.get("filename", "")),
                str(entry.get("correspondent", "")),
                str(entry.get("document_type", "")),
                str(entry.get("storage_path", "")),
                " ".join(entry.get("tags") or []),
            ])
            e_tokens = self._tokens(text)
            overlap = len(q_tokens & e_tokens)
            if overlap <= 0:
                continue
            scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        unique = []
        seen_keys = set()
        for _, entry in scored:
            key = (
                entry.get("correspondent", ""),
                entry.get("document_type", ""),
                entry.get("storage_path", ""),
                tuple(entry.get("tags") or []),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique.append(entry)
            if len(unique) >= max(0, int(limit)):
                break
        return unique

    def _build_correspondent_profiles(self) -> list[dict]:
        grouped: dict[str, dict] = {}
        for entry in self._examples:
            corr = _normalize_text(str(entry.get("correspondent", "")))
            if not corr:
                continue
            key = _normalize_tag_name(corr)
            if not key:
                continue
            row = grouped.setdefault(
                key,
                {
                    "correspondent": corr,
                    "count": 0,
                    "doc_types": defaultdict(int),
                    "paths": defaultdict(int),
                    "tags": defaultdict(int),
                },
            )
            row["count"] += 1
            doc_type = _normalize_text(str(entry.get("document_type", "")))
            path = _normalize_text(str(entry.get("storage_path", "")))
            if doc_type:
                row["doc_types"][doc_type] += 1
            if path:
                row["paths"][path] += 1
            for t in entry.get("tags") or []:
                tag_name = _normalize_text(str(t))
                if tag_name:
                    row["tags"][tag_name] += 1

        # Merge similar correspondent names (e.g. "Baader Bank" + "Baader Bank AG")
        grouped = self._merge_correspondent_variants(grouped)

        profiles = []
        for row in grouped.values():
            total = max(1, int(row["count"]))
            top_doc_type, top_doc_type_count = ("", 0)
            top_path, top_path_count = ("", 0)
            if row["doc_types"]:
                top_doc_type, top_doc_type_count = max(row["doc_types"].items(), key=lambda x: x[1])
            if row["paths"]:
                top_path, top_path_count = max(row["paths"].items(), key=lambda x: x[1])
            top_tags_sorted = sorted(row["tags"].items(), key=lambda x: x[1], reverse=True)[:4]
            tag_ratios = {name: (count / total) for name, count in top_tags_sorted}
            profiles.append(
                {
                    "correspondent": row["correspondent"],
                    "count": total,
                    "top_document_type": top_doc_type,
                    "document_type_ratio": (top_doc_type_count / total) if top_doc_type else 0.0,
                    "top_storage_path": top_path,
                    "storage_path_ratio": (top_path_count / total) if top_path else 0.0,
                    "top_tags": [name for name, _ in top_tags_sorted],
                    "tag_ratios": tag_ratios,
                }
            )
        return profiles

    @staticmethod
    def _merge_correspondent_variants(grouped: dict[str, dict], threshold: float = 0.85) -> dict[str, dict]:
        """Merge correspondent variants with similar names (e.g. 'Baader Bank' + 'Baader Bank AG')."""
        keys = list(grouped.keys())
        merged_into: dict[str, str] = {}  # maps absorbed key -> canonical key
        for i, key_a in enumerate(keys):
            if key_a in merged_into:
                continue
            for key_b in keys[i + 1:]:
                if key_b in merged_into:
                    continue
                ratio = difflib.SequenceMatcher(None, key_a, key_b).ratio()
                if ratio < threshold:
                    continue
                # Merge: keep the one with more samples as canonical
                row_a = grouped[key_a]
                row_b = grouped[key_b]
                if row_a["count"] >= row_b["count"]:
                    canonical_key, absorbed_key = key_a, key_b
                else:
                    canonical_key, absorbed_key = key_b, key_a
                canon = grouped[canonical_key]
                absorbed = grouped[absorbed_key]
                canon["count"] += absorbed["count"]
                for dt, cnt in absorbed["doc_types"].items():
                    canon["doc_types"][dt] += cnt
                for p, cnt in absorbed["paths"].items():
                    canon["paths"][p] += cnt
                for t, cnt in absorbed["tags"].items():
                    canon["tags"][t] += cnt
                merged_into[absorbed_key] = canonical_key
        for absorbed_key in merged_into:
            grouped.pop(absorbed_key, None)
        return grouped

    def routing_hints_for_document(self, document: dict, limit: int = LEARNING_PRIOR_MAX_HINTS) -> list[dict]:
        """
        Learns stable routing priors from confirmed examples.
        Example: many docs from "Scalable Capital" -> likely Kontoauszug + Finanzen/Bank.
        """
        if not ENABLE_LEARNING_PRIORS:
            return []
        if not self._examples:
            return []

        query = " ".join([
            str(document.get("title", "")),
            str(document.get("original_file_name", "")),
            str(document.get("content", ""))[:2200],
        ])
        query_norm = _normalize_tag_name(query)
        q_tokens = self._tokens(query)
        if not query_norm and not q_tokens:
            return []

        profiles = self._build_correspondent_profiles()
        scored = []
        for profile in profiles:
            count = int(profile.get("count", 0) or 0)
            if count < LEARNING_PRIOR_MIN_SAMPLES:
                continue
            corr = str(profile.get("correspondent", ""))
            corr_norm = _normalize_tag_name(corr)
            corr_tokens = self._tokens(corr)
            overlap = len(q_tokens & corr_tokens)
            exact = 1 if (corr_norm and corr_norm in query_norm) else 0
            if overlap <= 0 and exact == 0:
                continue
            score = float(overlap + (3 * exact)) + min(count, 20) * 0.08
            scored.append((score, profile))

        scored.sort(key=lambda x: x[0], reverse=True)
        unique = []
        seen = set()
        for _, profile in scored:
            corr = str(profile.get("correspondent", ""))
            corr_norm = _normalize_tag_name(corr)
            if not corr_norm or corr_norm in seen:
                continue
            seen.add(corr_norm)
            unique.append(profile)
            if len(unique) >= max(0, int(limit)):
                break
        return unique

# --- German-Character Detection ---
GERMAN_CHARS_RE = re.compile(r'[aeoeueAeOeUess]')


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PaperlessClient
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class PaperlessClient:
    """Vereinter Client fuer die Paperless-NGX REST API."""

    _CACHE_TTL_SEC = 120  # Master data cache lifetime

    _MIN_WRITE_INTERVAL = 0.15  # Minimum seconds between write operations

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token
        self._color_index = 0
        self._cache: dict[str, tuple[float, list]] = {}
        self._last_write_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        })

    def _rate_limit_write(self):
        """Ensure minimum interval between write operations."""
        now = time.monotonic()
        elapsed = now - self._last_write_time
        if elapsed < self._MIN_WRITE_INTERVAL:
            time.sleep(self._MIN_WRITE_INTERVAL - elapsed)
        self._last_write_time = time.monotonic()

    def _get_cached(self, endpoint: str) -> list:
        """Return cached result if still valid, otherwise fetch and cache."""
        now = time.monotonic()
        cached = self._cache.get(endpoint)
        if cached and (now - cached[0]) < self._CACHE_TTL_SEC:
            return cached[1]
        result = self._get_all(endpoint)
        self._cache[endpoint] = (now, result)
        return result

    def invalidate_cache(self, endpoint: str | None = None):
        """Clear cache for a specific endpoint or all."""
        if endpoint:
            self._cache.pop(endpoint, None)
        else:
            self._cache.clear()

    def _get_all(self, endpoint: str) -> list:
        """Alle Eintraege mit Paginierung laden."""
        results = []
        url = f"{self.url}/api/{endpoint}/?page_size=100"
        while url:
            url = url.replace("http://", "https://")
            last_exc = None
            resp = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = self.session.get(url, timeout=30)
                    resp.raise_for_status()
                    last_exc = None
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as exc:
                    last_exc = exc
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise
                except requests.exceptions.RequestException:
                    raise
            if resp is None:
                if last_exc:
                    raise last_exc
                raise RuntimeError(f"GET fehlgeschlagen: {url}")
            data = resp.json()
            results.extend(data.get("results", []))
            url = data.get("next")
        return results

    # --- Lesen ---
    def get_tags(self) -> list:
        return self._get_cached("tags")

    def get_correspondents(self) -> list:
        return self._get_cached("correspondents")

    def get_document_types(self) -> list:
        return self._get_cached("document_types")

    def get_storage_paths(self) -> list:
        return self._get_cached("storage_paths")

    def get_documents(self) -> list:
        return self._get_all("documents")  # Documents not cached - they change frequently

    def get_document(self, doc_id: int) -> dict:
        url = f"{self.url}/api/documents/{doc_id}/"
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except requests.exceptions.RequestException:
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Dokument konnte nicht geladen werden: {doc_id}")

    # --- Schreiben ---
    def _with_permissions(self, data: dict) -> dict:
        """Owner und Berechtigungen hinzufuegen."""
        data["owner"] = OWNER_ID
        data["set_permissions"] = {
            "view": {"users": [OWNER_ID], "groups": []},
            "change": {"users": [OWNER_ID], "groups": []},
        }
        return data

    def _next_color(self) -> str:
        color = TAG_COLORS[self._color_index % len(TAG_COLORS)]
        self._color_index += 1
        return color

    @staticmethod
    def _text_color_for_background(color: str) -> str:
        value = (color or "").strip().lower()
        if not value.startswith("#") or len(value) != 7:
            return "#ffffff"
        try:
            r = int(value[1:3], 16)
            g = int(value[3:5], 16)
            b = int(value[5:7], 16)
        except ValueError:
            return "#ffffff"
        # Simple luminance threshold for readable text.
        luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
        return "#000000" if luminance >= 170 else "#ffffff"

    def _write_with_retry(self, method: str, url: str, data: dict, timeout: int = 30) -> requests.Response:
        """Schreiboperation mit Retry bei transienten Fehlern."""
        self._rate_limit_write()
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = getattr(self.session, method)(url, json=data, timeout=timeout)
                resp.raise_for_status()
                return resp
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except requests.exceptions.RequestException:
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Schreiboperation fehlgeschlagen: {method.upper()} {url}")

    def update_document(self, doc_id: int, data: dict) -> dict:
        resp = self._write_with_retry("patch", f"{self.url}/api/documents/{doc_id}/", data)
        return resp.json()

    def create_tag(self, name: str, color: str | None = None, text_color: str | None = None) -> dict:
        color = (color or self._next_color()).strip().lower()
        text_color = (text_color or self._text_color_for_background(color)).strip().lower()
        data = {"name": name, "color": color, "text_color": text_color}
        resp = self._write_with_retry("post", f"{self.url}/api/tags/", self._with_permissions(data))
        self.invalidate_cache("tags")
        return resp.json()

    def create_correspondent(self, name: str) -> dict:
        resp = self._write_with_retry("post", f"{self.url}/api/correspondents/", self._with_permissions({"name": name}))
        self.invalidate_cache("correspondents")
        return resp.json()

    def create_document_type(self, name: str) -> dict:
        resp = self._write_with_retry("post", f"{self.url}/api/document_types/", self._with_permissions({"name": name}))
        self.invalidate_cache("document_types")
        return resp.json()

    def create_storage_path(self, name: str, path: str) -> dict:
        resp = self._write_with_retry("post", f"{self.url}/api/storage_paths/", self._with_permissions({"name": name, "path": path}))
        return resp.json()

    def get_next_asn(self) -> int:
        """Naechste freie Archivnummer."""
        resp = self.session.get(
            f"{self.url}/api/documents/?page_size=1&ordering=-archive_serial_number",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data["results"] and data["results"][0].get("archive_serial_number"):
            return data["results"][0]["archive_serial_number"] + 1
        return 1

    # --- Loeschen (einzeln mit Retry) ---
    def _delete_item(self, endpoint: str, item_id: int) -> bool:
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.delete(
                    f"{self.url}/api/{endpoint}/{item_id}/", timeout=15,
                )
                if resp.status_code != 204:
                    log.warning(f"DELETE {endpoint}/{item_id} unerwartet: Status {resp.status_code}")
                return resp.status_code == 204
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                time.sleep(2 * (attempt + 1))
        log.error(f"DELETE {endpoint}/{item_id} fehlgeschlagen nach {MAX_RETRIES} Versuchen: {last_exc}")
        return False

    def delete_tag(self, tag_id: int) -> bool:
        result = self._delete_item("tags", tag_id)
        if result:
            self.invalidate_cache("tags")
        return result

    def delete_correspondent(self, corr_id: int) -> bool:
        result = self._delete_item("correspondents", corr_id)
        if result:
            self.invalidate_cache("correspondents")
        return result

    def delete_document_type(self, type_id: int) -> bool:
        result = self._delete_item("document_types", type_id)
        if result:
            self.invalidate_cache("document_types")
        return result

    # --- Batch-Loeschung (parallel) ---
    def batch_delete(self, endpoint: str, items: list, label: str) -> int:
        """Parallele Loeschung mit ThreadPoolExecutor + Retry."""
        if not items:
            console.print(f"  Keine {label} zu loeschen.")
            return 0

        console.print(f"  Loesche {len(items)} {label}...")
        deleted = 0
        aborted = False
        pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        try:
            futures = {
                pool.submit(self._delete_item, endpoint, item["id"]): item
                for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    if future.result():
                        deleted += 1
                        console.print(f"    [green]Geloescht:[/green] {item['name']} (ID {item['id']})")
                    else:
                        console.print(f"    [red]Fehlgeschlagen:[/red] {item['name']} (ID {item['id']})")
                except Exception as e:
                    console.print(f"    [red]Fehler:[/red] {item['name']}: {e}")
        except KeyboardInterrupt:
            aborted = True
            log.warning("Batch-Loeschung durch Benutzer abgebrochen")
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if not aborted:
                pool.shutdown(wait=True, cancel_futures=False)

        console.print(f"  Fertig: {deleted}/{len(items)} {label} geloescht")
        return deleted


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LocalLLMAnalyzer
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def _apply_spelling_fixes(value: str) -> str:
    text = _normalize_text(value).lower()
    if not text:
        return ""
    for wrong, correct in SPELLING_FIXES.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text)
    return text


def _apply_title_spelling_fixes(value: str) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    for wrong, correct in TITLE_SPELLING_FIXES.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text, flags=re.IGNORECASE)
    return text


def _normalize_tag_name(value: str) -> str:
    return _apply_spelling_fixes(value)


def _normalize_correspondent_name(value: str) -> str:
    base = _apply_spelling_fixes(value)
    base = base.replace("  ", " ")
    return base


_COMPANY_SUFFIX_RE = re.compile(
    r"\s+(ag|gmbh|gbr|kg|ohg|e\.?\s?v\.?|se|mbh|co\.?\s?kg|gmbh\s?&\s?co\.?\s?kg|inc\.?|ltd\.?|llc|plc|corp\.?|s\.?a\.?)\.?\s*$",
    re.IGNORECASE,
)


def _canonicalize_correspondent_name(value: str) -> str:
    normalized = _normalize_correspondent_name(value)
    if not normalized:
        return ""
    # OCR / typo resilience for known patterns
    normalized = normalized.replace("thresien", "dresden")
    normalized = normalized.replace("industriekammer", "industrie- und handelskammer")
    # Direct alias lookup
    alias = CORRESPONDENT_ALIASES.get(normalized)
    if alias:
        return alias
    # Also try without company suffix (e.g., "Telekom Deutschland GmbH" -> "Telekom Deutschland")
    alias_lower = normalized.lower().strip()
    without_suffix = _COMPANY_SUFFIX_RE.sub("", alias_lower).strip()
    if without_suffix != alias_lower:
        alias = CORRESPONDENT_ALIASES.get(without_suffix)
        if alias:
            return alias
    return _normalize_text(value)


def _sanitize_suggestion_spelling(suggestion: dict):
    title = _apply_title_spelling_fixes(str(suggestion.get("title", "")))
    if title:
        suggestion["title"] = title

    corr = str(suggestion.get("correspondent", ""))
    if corr:
        suggestion["correspondent"] = _canonicalize_correspondent_name(corr)

    sanitized_tags = []
    for tag in suggestion.get("tags", []) or []:
        fixed = _apply_title_spelling_fixes(str(tag))
        if fixed:
            sanitized_tags.append(fixed)
    suggestion["tags"] = sanitized_tags


class TagTaxonomy:
    """Canonical tags + synonym mapping loaded from taxonomy_tags.json."""

    def __init__(self, path: str):
        self.path = path
        self.canonical_tags: list[str] = []
        self.synonym_to_canonical: dict[str, str] = {}
        self.metadata_by_canonical: dict[str, dict] = {}
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            self.canonical_tags = []
            self.synonym_to_canonical = {}
            self.metadata_by_canonical = {}
            log.warning(f"Taxonomie-Datei nicht gefunden: {self.path}")
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        canonical = []
        synonym_map = {}
        metadata_map = {}
        for entry in data.get("tags", []):
            name = _normalize_text(entry.get("name", ""))
            if not name:
                continue
            canonical.append(name)
            synonym_map[_normalize_tag_name(name)] = name
            for syn in entry.get("synonyms", []):
                syn_name = _normalize_tag_name(str(syn))
                if syn_name:
                    synonym_map[syn_name] = name
            metadata_map[name] = {
                "color": _normalize_text(str(entry.get("color", ""))) if entry.get("color") else "",
                "description": _normalize_text(str(entry.get("description", ""))) if entry.get("description") else "",
            }

        self.canonical_tags = sorted(set(canonical), key=lambda x: x.lower())
        self.synonym_to_canonical = synonym_map
        self.metadata_by_canonical = metadata_map

    def canonical_from_any(self, value: str) -> str | None:
        key = _normalize_tag_name(value)
        if not key:
            return None
        return self.synonym_to_canonical.get(key)

    def metadata_for_tag(self, canonical_name: str) -> dict:
        return dict(self.metadata_by_canonical.get(canonical_name, {}))

    def color_for_tag(self, canonical_name: str) -> str:
        meta = self.metadata_for_tag(canonical_name)
        color = _normalize_text(meta.get("color", ""))
        if color.startswith("#") and len(color) == 7:
            return color.lower()
        idx = sum(ord(ch) for ch in _normalize_tag_name(canonical_name)) % len(TAG_COLORS)
        return TAG_COLORS[idx]

    def description_for_tag(self, canonical_name: str) -> str:
        meta = self.metadata_for_tag(canonical_name)
        desc = _normalize_text(meta.get("description", ""))
        if desc:
            return desc
        return f"Kategorie fuer {canonical_name}."

    def prompt_tags(self, limit: int = MAX_PROMPT_TAG_CHOICES) -> list[str]:
        return self.canonical_tags[:max(1, limit)]


def _extract_keywords(text: str, limit: int = 6) -> list:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", text or "")
    skip = {"und", "oder", "mit", "fuer", "der", "die", "das", "invoice", "document"}
    counts = {}
    for word in words:
        key = word.lower()
        if key in skip:
            continue
        counts[key] = counts.get(key, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]]


def _get_brand_hint(text: str) -> str:
    haystack = (text or "").lower()
    for keyword, hint in KNOWN_BRAND_HINTS.items():
        if keyword in haystack:
            return hint
    return ""


def _fetch_web_hint(title: str, content: str) -> str:
    if not ENABLE_WEB_HINTS:
        return ""
    query_text = " ".join([title or "", *_extract_keywords(content, limit=5)]).strip()
    if not query_text:
        return ""
    cache_key = f"primary::{query_text.lower()}"
    with WEB_HINT_CACHE_LOCK:
        if cache_key in WEB_HINT_CACHE:
            return WEB_HINT_CACHE[cache_key]
    params = urlencode({"q": query_text, "format": "json", "no_html": 1, "skip_disambig": 1})
    url = f"https://api.duckduckgo.com/?{params}"
    result = ""
    try:
        resp = requests.get(url, timeout=WEB_HINT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        abstract = (data.get("AbstractText") or "").strip()
        heading = (data.get("Heading") or "").strip()
        if abstract:
            result = f"WEB-HINWEIS: {heading}: {abstract[:220]}"
    except Exception:
        result = ""
    with WEB_HINT_CACHE_LOCK:
        WEB_HINT_CACHE[cache_key] = result
    return result


def _fetch_entity_web_hint(entity: str) -> str:
    if not ENABLE_WEB_HINTS:
        return ""
    entity_clean = _normalize_text(entity)
    if not entity_clean:
        return ""
    cache_key = f"entity::{entity_clean.lower()}"
    with WEB_HINT_CACHE_LOCK:
        if cache_key in WEB_HINT_CACHE:
            return WEB_HINT_CACHE[cache_key]
    params = urlencode({"q": f"{entity_clean} company", "format": "json", "no_html": 1, "skip_disambig": 1})
    url = f"https://api.duckduckgo.com/?{params}"
    result = ""
    try:
        resp = requests.get(url, timeout=WEB_HINT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        abstract = (data.get("AbstractText") or "").strip()
        heading = (data.get("Heading") or "").strip()
        if abstract:
            result = f"{entity_clean}: {heading} - {abstract[:160]}"
    except Exception:
        result = ""
    with WEB_HINT_CACHE_LOCK:
        WEB_HINT_CACHE[cache_key] = result
    return result


def _web_search_entity(entity: str, max_results: int = 3) -> str:
    """Real web search via duckduckgo-search package with fallback to Instant Answers API."""
    if not ENABLE_WEB_HINTS:
        return ""
    entity_clean = _normalize_text(entity)
    if not entity_clean:
        return ""
    cache_key = f"websearch::{entity_clean.lower()}"
    with WEB_HINT_CACHE_LOCK:
        if cache_key in WEB_HINT_CACHE:
            return WEB_HINT_CACHE[cache_key]
    result = ""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(f"{entity_clean} Firma Unternehmen", max_results=max_results))
        if hits:
            parts = []
            for hit in hits[:max_results]:
                title = (hit.get("title") or "").strip()
                body = (hit.get("body") or "").strip()
                if title or body:
                    parts.append(f"{title}: {body[:120]}")
            if parts:
                result = f"{entity_clean}: " + " | ".join(parts)
    except ImportError:
        # Fallback: use old DuckDuckGo Instant Answers API
        result = _fetch_entity_web_hint(entity)
    except Exception:
        # Graceful fallback on any search error
        try:
            result = _fetch_entity_web_hint(entity)
        except Exception:
            result = ""
    with WEB_HINT_CACHE_LOCK:
        WEB_HINT_CACHE[cache_key] = result
    return result


# --- German month names for date parsing ---
_GERMAN_MONTHS = {
    "januar": 1, "februar": 2, "maerz": 3, "märz": 3, "april": 4,
    "mai": 5, "juni": 6, "juli": 7, "august": 8, "september": 9,
    "oktober": 10, "november": 11, "dezember": 12,
    "jan": 1, "feb": 2, "mär": 3, "mar": 3, "apr": 4, "mai": 5,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "dez": 12,
}


def _extract_document_date(document: dict) -> str:
    """Extract the most likely document date from content (German formats)."""
    content = str(document.get("content") or "")
    if not content:
        return ""
    # Check first 1500 chars where date usually appears
    text = content[:1500]

    # Pattern 1: dd.mm.yyyy (most common German format)
    dates_dot = re.findall(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", text)
    # Pattern 2: dd. Monat yyyy
    dates_named = re.findall(r"\b(\d{1,2})\.\s*([A-Za-zäöüÄÖÜ]+)\s+(\d{4})\b", text)
    # Pattern 3: yyyy-mm-dd (ISO format)
    dates_iso = re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)

    candidates = []
    for day, month, year in dates_dot:
        try:
            d = datetime(int(year), int(month), int(day))
            if 2000 <= d.year <= 2030:
                candidates.append(d)
        except (ValueError, OverflowError):
            pass
    for day, month_name, year in dates_named:
        month_num = _GERMAN_MONTHS.get(month_name.lower())
        if month_num:
            try:
                d = datetime(int(year), month_num, int(day))
                if 2000 <= d.year <= 2030:
                    candidates.append(d)
            except (ValueError, OverflowError):
                pass
    for year, month, day in dates_iso:
        try:
            d = datetime(int(year), int(month), int(day))
            if 2000 <= d.year <= 2030:
                candidates.append(d)
        except (ValueError, OverflowError):
            pass

    if not candidates:
        return ""
    # Return the most recent date (likely the document date, not some reference date)
    best = max(candidates)
    return best.strftime("%Y-%m-%d")


_GERMAN_MARKERS = {
    "und", "oder", "der", "die", "das", "ein", "eine", "ist", "sind", "wird", "wurde",
    "bei", "mit", "nach", "fuer", "ueber", "unter", "zwischen", "durch", "gegen",
    "rechnung", "vertrag", "sehr", "geehrte", "freundlichen", "gruessen", "bestaetigung",
    "versicherung", "kontoauszug", "bescheinigung", "mitteilung", "kuendigung",
    "antrag", "mietvertrag", "steuerbescheid",
}
_ENGLISH_MARKERS = {
    "the", "and", "for", "with", "this", "that", "from", "your", "have", "been",
    "invoice", "receipt", "agreement", "confirmation", "payment", "subscription",
    "account", "statement", "service", "please", "thank", "dear", "regards",
}


def _detect_language(text: str) -> str:
    """Detect document language ('de' or 'en') based on word frequency."""
    words = set(re.findall(r"\b[a-z]{3,}\b", text[:2000].lower()))
    de_count = len(words & _GERMAN_MARKERS)
    en_count = len(words & _ENGLISH_MARKERS)
    return "de" if de_count >= en_count else "en"


def _assess_ocr_quality(document: dict) -> tuple[str, float]:
    """Assess OCR quality of document content. Returns (quality_level, score) where quality_level is 'good', 'medium', or 'poor'."""
    content = str(document.get("content") or "")
    if not content:
        return ("poor", 0.0)

    total_chars = len(content)
    if total_chars < 50:
        return ("poor", 0.1)

    # Count various quality indicators
    alpha_chars = sum(1 for c in content if c.isalpha())
    digit_chars = sum(1 for c in content if c.isdigit())
    space_chars = sum(1 for c in content if c.isspace())
    special_chars = total_chars - alpha_chars - digit_chars - space_chars

    # Words check
    words = content.split()
    total_words = len(words)
    if total_words < 5:
        return ("poor", 0.15)

    # Ratio of alphabetic characters (good OCR has high ratio)
    alpha_ratio = alpha_chars / max(1, total_chars)
    # Ratio of special characters (bad OCR has lots of garbage)
    special_ratio = special_chars / max(1, total_chars)
    # Average word length (very short = garbage, very long = merged words)
    avg_word_len = sum(len(w) for w in words) / max(1, total_words)
    # Count words that look like real words (3+ alpha chars)
    real_words = sum(1 for w in words if len(w) >= 3 and sum(1 for c in w if c.isalpha()) >= 2)
    real_word_ratio = real_words / max(1, total_words)

    score = 0.0
    score += min(0.3, alpha_ratio * 0.4)  # More alpha = better
    score += min(0.2, (1.0 - special_ratio) * 0.25)  # Less special = better
    score += min(0.2, real_word_ratio * 0.25)  # More real words = better
    score += 0.15 if 3.0 <= avg_word_len <= 12.0 else 0.0  # Normal word length
    score += 0.15 if total_words >= 20 else (0.07 if total_words >= 10 else 0.0)

    if score >= 0.7:
        return ("good", score)
    elif score >= 0.4:
        return ("medium", score)
    else:
        return ("poor", score)


def _improve_title(title: str, document: dict) -> str:
    """Clean up and improve LLM-generated titles."""
    if not title:
        return title
    # Remove common file extensions the LLM might include
    title = re.sub(r"\.(pdf|png|jpg|jpeg|tiff?|docx?|xlsx?|csv)$", "", title, flags=re.IGNORECASE).strip()
    # Remove leading/trailing quotes
    title = title.strip("'\"")
    # Remove "Dokument:" or similar prefixes
    title = re.sub(r"^(Dokument|Document|Datei|File|Titel|Betreff|Subject)\s*[:|-]\s*", "", title, flags=re.IGNORECASE).strip()
    # Remove OCR artifacts: sequences of random characters
    title = re.sub(r"[|/\\]{2,}", " ", title)
    # Collapse multiple spaces
    title = re.sub(r"\s{2,}", " ", title).strip()
    # Remove repeated words (e.g., "Rechnung Rechnung")
    words = title.split()
    deduped = []
    for w in words:
        if not deduped or w.lower() != deduped[-1].lower():
            deduped.append(w)
    title = " ".join(deduped)
    # Remove trailing hyphens or dots
    title = title.rstrip("-. ")
    # Capitalize first letter
    if title and title[0].islower():
        title = title[0].upper() + title[1:]
    # Truncate very long titles
    if len(title) > 128:
        title = title[:125] + "..."
    return title


def _content_fingerprint(content: str) -> set[int]:
    """Create a set of trigram hashes for content similarity comparison (MinHash-like)."""
    if not content:
        return set()
    # Normalize: lowercase, remove special chars, collapse whitespace
    text = re.sub(r"[^a-z0-9äöüß\s]", " ", content.lower())
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) < 5:
        return set()
    # Create word trigrams and hash them
    trigrams = set()
    for i in range(len(words) - 2):
        trigram = " ".join(words[i:i + 3])
        trigrams.add(hash(trigram))
    return trigrams


def _content_similarity(fp_a: set[int], fp_b: set[int]) -> float:
    """Jaccard similarity between two content fingerprints."""
    if not fp_a or not fp_b:
        return 0.0
    intersection = len(fp_a & fp_b)
    union = len(fp_a | fp_b)
    return intersection / max(1, union)


_KNOWN_BLZ = {
    "10010010": "Postbank",
    "10020500": "Bank fuer Sozialwirtschaft",
    "10050000": "Landesbank Berlin",
    "10070000": "Deutsche Bank Berlin",
    "10070024": "Deutsche Bank",
    "10070848": "Commerzbank",
    "10090000": "Berliner Volksbank",
    "12030000": "Deutsche Kreditbank (DKB)",
    "20010020": "Postbank Hamburg",
    "20050550": "Haspa",
    "20070000": "Deutsche Bank Hamburg",
    "20070024": "Deutsche Bank",
    "25050000": "Nord/LB",
    "26050001": "Sparkasse Osnabrueck",
    "30010111": "SEB",
    "30020900": "Targobank",
    "30060601": "Deutsche Apotheker- und Aerztebank",
    "37010050": "Postbank Koeln",
    "43060967": "Volksbank Paderborn",
    "50010517": "ING",
    "50020200": "BHF-BANK",
    "50040000": "Commerzbank Frankfurt",
    "50050201": "Frankfurter Sparkasse",
    "50060400": "Evangelische Bank",
    "50070010": "Deutsche Bank Frankfurt",
    "50070024": "Deutsche Bank",
    "50080000": "Commerzbank",
    "50090500": "Sparda-Bank Hessen",
    "50310400": "Baader Bank",
    "51230800": "Volksbank Mittelhessen",
    "60020290": "UniCredit Bank - HypoVereinsbank",
    "60050101": "BW-Bank",
    "60070070": "Deutsche Bank Stuttgart",
    "68452290": "Sparkasse Markgraeflerland",
    "70010080": "Postbank Muenchen",
    "70020270": "UniCredit Bank - HypoVereinsbank",
    "70050000": "Bayerische Landesbank",
    "70070010": "Deutsche Bank Muenchen",
    "70070024": "Deutsche Bank",
    "70090100": "Volksbank Raiffeisenbank",
    "76010085": "Postbank Nuernberg",
    "82050000": "Sparkasse Gera-Greiz",
    "83050000": "Sparkasse Mittelthueringen",
    "85050100": "Sparkasse Chemnitz",
    "85050300": "Ostsaechsische Sparkasse Dresden",
    "86050200": "Sparkasse Leipzig",
    "10050006": "Landesbank Berlin - Berliner Sparkasse",
    "30050110": "Stadtsparkasse Duesseldorf",
    "76050101": "Sparkasse Nuernberg",
    "44050199": "Sparkasse Dortmund",
}


def _extract_document_entities(document: dict) -> list[str]:
    """Extract potential entity names from document text (email domains, URLs, IBAN banks)."""
    content = str(document.get("content") or "")
    title = str(document.get("title") or "")
    filename = str(document.get("original_file_name") or "")
    # Use last 500 chars (often letterhead/footer) + first 500 chars + title + filename
    text_parts = title + " " + filename + " " + content[:500] + " " + content[-500:] if len(content) > 500 else title + " " + filename + " " + content

    entities = []
    seen_norm = set()

    def _add_entity(name: str):
        norm = _normalize_tag_name(name)
        if norm and norm not in seen_norm and len(name) > 2:
            seen_norm.add(norm)
            entities.append(name)

    # Email domains -> company name
    for match in re.finditer(r"@([a-z0-9.-]+\.[a-z]{2,})", text_parts, re.IGNORECASE):
        domain = match.group(1).lower()
        # Skip generic email providers
        if domain.split(".")[-2] in ("gmail", "gmx", "yahoo", "outlook", "hotmail", "web", "t-online", "posteo", "protonmail"):
            continue
        # Extract company name from domain (e.g. "scalable.capital" -> "scalable capital")
        company = domain.rsplit(".", 1)[0].replace(".", " ").replace("-", " ").strip()
        if company:
            _add_entity(company)

    # URLs -> domain -> company name
    for match in re.finditer(r"https?://(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})", text_parts, re.IGNORECASE):
        domain = match.group(1).lower()
        company = domain.rsplit(".", 1)[0].replace(".", " ").replace("-", " ").strip()
        if company and len(company) > 2:
            _add_entity(company)

    # IBAN pattern -> resolve bank name from BLZ
    for match in re.finditer(r"\b([A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{0,4}\s?\d{0,2})\b", text_parts):
        iban = match.group(1).replace(" ", "")
        if len(iban) >= 16 and iban[:2] == "DE" and len(iban) >= 12:
            blz = iban[4:12]
            bank_name = _KNOWN_BLZ.get(blz)
            if bank_name:
                _add_entity(bank_name)
            else:
                _add_entity(f"Bank BLZ {blz}")
            break  # Only search for one IBAN

    return entities[:4]  # Limit to avoid too many searches


def _web_search_document_context(document: dict) -> str:
    """Build web search context for unrecognized documents by searching for extracted entities."""
    if not ENABLE_WEB_HINTS:
        return ""
    entities = _extract_document_entities(document)
    if not entities:
        return ""

    results = []
    for entity in entities[:WEB_HINT_MAX_ENTITIES]:
        hint = _web_search_entity(entity)
        if hint:
            results.append(hint)
    if not results:
        return ""
    return "WEBSUCHE: " + " | ".join(results)


def build_decision_context(documents: list, correspondents: list, storage_paths: list,
                           learning_profile: LearningProfile | None = None) -> DecisionContext:
    """
    Phase 1: collect current system data before making document decisions.
    Learns likely employers/vendors/paths from existing assignments.
    """
    context = DecisionContext()
    corr_by_id = {int(c["id"]): str(c.get("name", "")) for c in correspondents if c.get("id") is not None}
    path_by_id = {int(p["id"]): str(p.get("name", "")) for p in storage_paths if p.get("id") is not None}

    work_corr_counts = defaultdict(int)
    provider_counts = defaultdict(int)
    work_path_counts = defaultdict(int)
    private_path_counts = defaultdict(int)

    for doc in documents:
        corr_name = corr_by_id.get(int(doc.get("correspondent") or 0), "")
        path_name = path_by_id.get(int(doc.get("storage_path") or 0), "")
        corr_norm = _normalize_tag_name(corr_name)
        path_norm = _normalize_tag_name(path_name)

        if path_norm.startswith("arbeit/"):
            if corr_norm:
                work_corr_counts[corr_norm] += 1
            work_path_counts[path_name] += 1
        elif path_name:
            private_path_counts[path_name] += 1

        for vendor_key in VENDOR_GUARDRAILS:
            if vendor_key in corr_norm:
                provider_counts[vendor_key] += 1

    context.employer_names.update(EMPLOYER_HINTS)
    if learning_profile:
        context.employer_names.update(learning_profile.employer_names())
        context.profile_employment_lines = learning_profile.employment_lines()
        context.profile_private_vehicles = learning_profile.private_vehicle_hints()
        context.profile_company_vehicles = learning_profile.company_vehicle_hints()
        context.profile_context_text = learning_profile.prompt_context_text()
    for norm_name, count in sorted(work_corr_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= WORK_CORR_EMPLOYER_MIN_DOCS:
            context.employer_names.add(norm_name)

    for vendor_key in VENDOR_GUARDRAILS:
        context.provider_names.add(vendor_key)
    for vendor_key, count in provider_counts.items():
        if count > 0:
            context.provider_names.add(vendor_key)

    context.top_work_paths = [k for k, _ in sorted(work_path_counts.items(), key=lambda x: x[1], reverse=True)[:6]]
    context.top_private_paths = [k for k, _ in sorted(private_path_counts.items(), key=lambda x: x[1], reverse=True)[:6]]
    context.notes.append(f"docs={len(documents)}")
    context.notes.append(f"employers={len(context.employer_names)}")
    context.notes.append(f"providers={len(context.provider_names)}")
    if context.profile_employment_lines:
        context.notes.append(f"profile_jobs={len(context.profile_employment_lines)}")
    if context.profile_private_vehicles or context.profile_company_vehicles:
        context.notes.append(
            f"profile_vehicles={len(context.profile_private_vehicles)}+{len(context.profile_company_vehicles)}"
        )
    return context


def _collect_web_entity_hints(document: dict, current_corr: str = "") -> str:
    if not ENABLE_WEB_HINTS:
        return ""
    text = " ".join([
        str(document.get("title", "")),
        str(document.get("original_file_name", "")),
        str(document.get("content", ""))[:1500],
        current_corr,
    ]).lower()
    candidates = []
    for key in KNOWN_BRAND_HINTS:
        if key in text:
            candidates.append(key)
    if current_corr:
        candidates.append(current_corr)

    seen = set()
    hints = []
    for candidate in candidates:
        norm = _normalize_tag_name(candidate)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        hint = _web_search_entity(candidate)
        if hint:
            hints.append(hint)
        if len(hints) >= WEB_HINT_MAX_ENTITIES:
            break
    return " | ".join(hints)


def _find_vendor_key(text: str) -> str:
    haystack = (text or "").lower()
    for key in VENDOR_GUARDRAILS:
        if key in haystack:
            return key
    return ""


def _contains_any_hint(text: str, hints: list[str]) -> bool:
    haystack = (text or "").lower()
    return any(hint in haystack for hint in hints)


def _apply_vehicle_guardrails(document: dict, suggestion: dict, storage_paths: list,
                              decision_context: DecisionContext | None = None) -> list[str]:
    """
    User-specific vehicle guardrail:
    - VW Polo is private car
    - Toyota is company car
    """
    corrections = []
    text = " ".join([
        str(document.get("title", "")),
        str(document.get("original_file_name", "")),
        str(document.get("content", ""))[:3000],
        str(suggestion.get("title", "")),
        str(suggestion.get("storage_path", "")),
        " ".join(str(t) for t in (suggestion.get("tags") or [])),
    ]).lower()
    path_value = _normalize_text(str(suggestion.get("storage_path", "")))
    path_lower = path_value.lower()

    private_hints = PRIVATE_VEHICLE_HINTS
    company_hints = COMPANY_VEHICLE_HINTS
    if decision_context:
        if decision_context.profile_private_vehicles:
            private_hints = decision_context.profile_private_vehicles
        if decision_context.profile_company_vehicles:
            company_hints = decision_context.profile_company_vehicles

    has_private_vehicle = _contains_any_hint(text, private_hints)
    has_company_vehicle = _contains_any_hint(text, company_hints)

    if has_private_vehicle and path_lower.startswith("arbeit/"):
        safe_private_path = _pick_existing_storage_path(
            storage_paths,
            ["Auto/Service", "Auto/Versicherung", "Auto", "Persoenlich/Auto", "Freizeit/Auto"],
        )
        if safe_private_path:
            suggestion["storage_path"] = safe_private_path
            corrections.append(f"storage_path->{safe_private_path} (private vehicle)")

    if has_company_vehicle and path_lower and not path_lower.startswith("arbeit/"):
        work_preferences = []
        if decision_context:
            work_preferences.extend(decision_context.top_work_paths)
        work_preferences.extend(["Arbeit/WBS", "Arbeit/msg", "Arbeit"])
        safe_work_path = _pick_existing_storage_path(storage_paths, work_preferences)
        if safe_work_path:
            suggestion["storage_path"] = safe_work_path
            corrections.append(f"storage_path->{safe_work_path} (company vehicle)")

    return corrections


def _pick_existing_storage_path(storage_paths: list, preferences: list[str]) -> str:
    if not storage_paths:
        return ""
    normalized = [p.get("name", "") for p in storage_paths if p.get("name")]
    for pref in preferences:
        for candidate in normalized:
            if candidate.lower() == pref.lower():
                return candidate
    for pref in preferences:
        for candidate in normalized:
            if pref.lower() in candidate.lower() or candidate.lower() in pref.lower():
                return candidate
    return ""


def _get_correspondent_name_by_id(correspondents: list, corr_id: int | None) -> str:
    if not corr_id:
        return ""
    for corr in correspondents:
        if corr.get("id") == corr_id:
            return str(corr.get("name", ""))
    return ""


def _resolve_correspondent_from_name(correspondents: list, corr_name: str) -> tuple[int | None, str]:
    """
    Resolve correspondent by canonical alias + fuzzy match.
    Returns (id, resolved_name).
    """
    wanted = _canonicalize_correspondent_name(corr_name)
    if not wanted:
        return None, ""

    by_norm = {_normalize_correspondent_name(c.get("name", "")): c for c in correspondents}
    wanted_norm = _normalize_correspondent_name(wanted)
    if wanted_norm in by_norm:
        item = by_norm[wanted_norm]
        return int(item["id"]), str(item.get("name", wanted))

    close = difflib.get_close_matches(wanted_norm, list(by_norm.keys()), n=1, cutoff=CORRESPONDENT_MATCH_THRESHOLD)
    if close:
        item = by_norm[close[0]]
        return int(item["id"]), str(item.get("name", wanted))
    return None, wanted


def _effective_employer_hints(decision_context: DecisionContext | None = None) -> set[str]:
    hints = set(EMPLOYER_HINTS)
    if decision_context:
        hints.update(decision_context.employer_names)
    return hints


def _apply_vendor_guardrails(document: dict, suggestion: dict, correspondents: list, storage_paths: list,
                             decision_context: DecisionContext | None = None) -> list[str]:
    """Auto-correct common provider/employer misclassifications."""
    corrections = []
    current_corr_name = _get_correspondent_name_by_id(correspondents, document.get("correspondent"))
    text = " ".join([
        str(document.get("title", "")),
        str(document.get("original_file_name", "")),
        str(document.get("content", ""))[:3000],
        str(current_corr_name),
        str(suggestion.get("correspondent", "")),
    ]).lower()
    vendor_key = _find_vendor_key(text)
    if not vendor_key:
        return corrections

    suggested_corr = _normalize_tag_name(str(suggestion.get("correspondent", "")))
    current_corr = _normalize_tag_name(current_corr_name)
    guard = VENDOR_GUARDRAILS[vendor_key]
    employer_hints = _effective_employer_hints(decision_context)

    # If a provider is detected, do not allow forced employer reassignment.
    if suggested_corr in employer_hints and vendor_key not in employer_hints:
        suggestion["correspondent"] = guard["correspondent"]
        corrections.append(f"correspondent->{guard['correspondent']} (provider conflict)")
    elif not suggested_corr and current_corr in employer_hints:
        suggestion["correspondent"] = guard["correspondent"]
        corrections.append(f"correspondent->{guard['correspondent']} (empty+provider)")
    elif current_corr and vendor_key in current_corr and suggested_corr in employer_hints:
        suggestion["correspondent"] = current_corr_name
        corrections.append(f"correspondent->{current_corr_name} (keep current provider)")

    # Keep provider documents away from work paths.
    storage_path_value = _normalize_text(str(suggestion.get("storage_path", "")))
    if storage_path_value.lower().startswith("arbeit/"):
        safe_path = _pick_existing_storage_path(storage_paths, guard["path_preferences"])
        if safe_path:
            suggestion["storage_path"] = safe_path
            corrections.append(f"storage_path->{safe_path} (provider conflict)")

    # Optional web hint for auditability (only queried on detected conflicts/providers).
    web_provider = _fetch_entity_web_hint(guard["correspondent"])
    if web_provider:
        suggestion["reasoning"] = f"{suggestion.get('reasoning', '')} | Web: {web_provider}".strip(" |")
    return corrections


def _apply_topic_guardrails(document: dict, suggestion: dict, correspondents: list, storage_paths: list,
                            decision_context: DecisionContext | None = None) -> list[str]:
    """Generic content guardrails for health/school/ticket documents."""
    corrections = []
    current_corr_name = _get_correspondent_name_by_id(correspondents, document.get("correspondent"))
    suggested_corr = _normalize_tag_name(str(suggestion.get("correspondent", "")))
    current_corr = _normalize_tag_name(current_corr_name)
    employer_hints = _effective_employer_hints(decision_context)

    text = " ".join([
        str(document.get("title", "")),
        str(document.get("original_file_name", "")),
        str(document.get("content", ""))[:3500],
        str(suggestion.get("title", "")),
        str(suggestion.get("reasoning", "")),
    ]).lower()
    path_value = _normalize_text(str(suggestion.get("storage_path", "")))
    path_lower = path_value.lower()

    is_health = _contains_any_hint(text, HEALTH_HINTS)
    is_school = _contains_any_hint(text, SCHOOL_HINTS)
    is_event_ticket = _contains_any_hint(text, EVENT_TICKET_HINTS)
    is_transport_ticket = _contains_any_hint(text, TRANSPORT_TICKET_HINTS) and not is_event_ticket

    if is_health:
        safe_health_path = _pick_existing_storage_path(
            storage_paths,
            ["Gesundheit/Arzt", "Gesundheit/Krankenkasse", "Gesundheit", "Korrespondenz/Allgemein"],
        )
        if safe_health_path and (not path_lower or path_lower.startswith("arbeit/") or path_lower.startswith("korrespondenz/")):
            suggestion["storage_path"] = safe_health_path
            corrections.append(f"storage_path->{safe_health_path} (health)")
        if suggested_corr in employer_hints and current_corr_name and current_corr not in employer_hints:
            suggestion["correspondent"] = current_corr_name
            corrections.append(f"correspondent->{current_corr_name} (health context)")

    if is_school:
        safe_school_path = _pick_existing_storage_path(
            storage_paths,
            ["Ausbildung/Berufsschule", "Ausbildung/Schule", "Ausbildung", "Korrespondenz/Allgemein"],
        )
        if safe_school_path and (not path_lower or path_lower.startswith("arbeit/") or path_lower.startswith("korrespondenz/")):
            suggestion["storage_path"] = safe_school_path
            corrections.append(f"storage_path->{safe_school_path} (school)")
        if suggested_corr in employer_hints and current_corr_name and current_corr not in employer_hints:
            suggestion["correspondent"] = current_corr_name
            corrections.append(f"correspondent->{current_corr_name} (school context)")

    if is_event_ticket:
        safe_event_path = _pick_existing_storage_path(
            storage_paths,
            ["Freizeit/Events", "Freizeit", "Freizeit/Reisen", "Korrespondenz/Allgemein"],
        )
        if safe_event_path and (not path_lower or path_lower.startswith("korrespondenz/")):
            suggestion["storage_path"] = safe_event_path
            corrections.append(f"storage_path->{safe_event_path} (event ticket)")

    if is_transport_ticket:
        safe_transport_path = _pick_existing_storage_path(
            storage_paths,
            ["Finanzen/Rechnungen", "Finanzen", "Freizeit/Reisen", "Mobilitaet", "Korrespondenz/Allgemein"],
        )
        if safe_transport_path and (not path_lower or path_lower.startswith("korrespondenz/")):
            suggestion["storage_path"] = safe_transport_path
            corrections.append(f"storage_path->{safe_transport_path} (transport ticket)")

    return corrections


def _apply_learning_guardrails(suggestion: dict, storage_paths: list, learning_hints: list[dict] | None) -> list[str]:
    """
    Apply conservative priors learned from confirmed examples.
    Only nudges generic/empty outputs, does not hard-overwrite specific good predictions.
    """
    corrections = []
    if not ENABLE_LEARNING_PRIORS or not learning_hints:
        return corrections

    best = learning_hints[0]
    min_ratio = max(0.5, min(0.95, LEARNING_PRIOR_MIN_RATIO))
    top_corr = _normalize_text(str(best.get("correspondent", "")))
    top_type = _normalize_text(str(best.get("top_document_type", "")))
    top_path = _normalize_text(str(best.get("top_storage_path", "")))
    type_ratio = float(best.get("document_type_ratio", 0.0) or 0.0)
    path_ratio = float(best.get("storage_path_ratio", 0.0) or 0.0)
    top_tags = [str(t) for t in (best.get("top_tags") or []) if _normalize_text(str(t))]
    tag_ratios = best.get("tag_ratios") or {}

    corr_now = _normalize_text(str(suggestion.get("correspondent", "")))
    type_now = _normalize_text(str(suggestion.get("document_type", "")))
    path_now = _normalize_text(str(suggestion.get("storage_path", "")))

    if not corr_now and top_corr:
        suggestion["correspondent"] = top_corr
        corrections.append(f"correspondent->{top_corr} (learning prior)")

    if top_type and type_ratio >= min_ratio and (not type_now or type_now.lower() == "information"):
        suggestion["document_type"] = top_type
        corrections.append(f"document_type->{top_type} (learning prior {type_ratio:.0%})")

    if top_path and path_ratio >= min_ratio:
        path_is_generic = (not path_now) or path_now.lower().startswith("korrespondenz/allgemein")
        if path_is_generic:
            safe_path = _pick_existing_storage_path(storage_paths, [top_path])
            if safe_path:
                suggestion["storage_path"] = safe_path
                corrections.append(f"storage_path->{safe_path} (learning prior {path_ratio:.0%})")

    if LEARNING_PRIOR_ENABLE_TAG_SUGGESTION:
        tags_now = [_normalize_text(str(t)) for t in (suggestion.get("tags") or []) if _normalize_text(str(t))]
        tags_now_norm = {_normalize_tag_name(t) for t in tags_now}
        blocked_norm = {
            _normalize_tag_name(REVIEW_TAG_NAME),
            _normalize_tag_name("Duplikat"),
            _normalize_tag_name("Auto"),
        }
        added_tags = 0
        for tag in top_tags:
            if len(tags_now) >= MAX_TAGS_PER_DOC:
                break
            ratio = float(tag_ratios.get(tag, 0.0) or 0.0)
            if ratio < max(0.55, min_ratio - 0.1):
                continue
            norm = _normalize_tag_name(tag)
            if not norm or norm in tags_now_norm or norm in blocked_norm:
                continue
            tags_now.append(tag)
            tags_now_norm.add(norm)
            added_tags += 1
            if added_tags >= 2:
                break
        if added_tags:
            suggestion["tags"] = tags_now
            corrections.append(f"tags+{added_tags} (learning prior)")

    return corrections


RULE_BASED_MIN_SAMPLES = int(os.getenv("RULE_BASED_MIN_SAMPLES", "10"))
RULE_BASED_MIN_RATIO = float(os.getenv("RULE_BASED_MIN_RATIO", "0.80"))


def _try_rule_based_suggestion(document: dict, learning_hints: list[dict],
                               storage_paths: list) -> dict | None:
    """Fast path: skip LLM entirely for well-known correspondent patterns (10+ samples, >80% consistent)."""
    if not ENABLE_LEARNING_PRIORS or not learning_hints:
        return None
    best = learning_hints[0]
    count = int(best.get("count", 0) or 0)
    if count < RULE_BASED_MIN_SAMPLES:
        return None

    corr = _normalize_text(str(best.get("correspondent", "")))
    if not corr:
        return None

    type_ratio = float(best.get("document_type_ratio", 0.0) or 0.0)
    path_ratio = float(best.get("storage_path_ratio", 0.0) or 0.0)

    # Only use rule-based if BOTH doc_type and path are highly consistent
    if type_ratio < RULE_BASED_MIN_RATIO or path_ratio < RULE_BASED_MIN_RATIO:
        return None

    top_type = _normalize_text(str(best.get("top_document_type", "")))
    top_path = _normalize_text(str(best.get("top_storage_path", "")))
    if not top_type or not top_path:
        return None

    safe_path = _pick_existing_storage_path(storage_paths, [top_path])
    if not safe_path:
        return None

    top_tags = [str(t) for t in (best.get("top_tags") or []) if _normalize_text(str(t))]
    tag_ratios = best.get("tag_ratios") or {}
    rule_tags = []
    for tag in top_tags:
        ratio = float(tag_ratios.get(tag, 0.0) or 0.0)
        if ratio >= RULE_BASED_MIN_RATIO and len(rule_tags) < MAX_TAGS_PER_DOC:
            rule_tags.append(tag)

    return {
        "correspondent": corr,
        "title": document.get("title", ""),
        "document_type": top_type,
        "storage_path": safe_path,
        "tags": rule_tags or [],
        "confidence": "rule_based",
        "reasoning": f"Regelbasiert: {count} Beispiele, Typ {type_ratio:.0%}, Pfad {path_ratio:.0%}",
    }


def _build_suggestion_from_priors(document: dict, learning_hints: list[dict],
                                  storage_paths: list) -> dict | None:
    """Baut aus Learning-Priors eine Suggestion OHNE LLM-Aufruf (Fallback)."""
    if not learning_hints:
        return None
    best = learning_hints[0]
    count = int(best.get("count", 0) or 0)
    if count < LEARNING_PRIOR_MIN_SAMPLES:
        return None

    corr = _normalize_text(str(best.get("correspondent", "")))
    if not corr:
        return None

    suggestion: dict = {
        "correspondent": corr,
        "title": document.get("title", ""),
        "confidence": "prior_only",
    }

    top_type = _normalize_text(str(best.get("top_document_type", "")))
    type_ratio = float(best.get("document_type_ratio", 0.0) or 0.0)
    if top_type and type_ratio >= 0.60:
        suggestion["document_type"] = top_type

    top_path = _normalize_text(str(best.get("top_storage_path", "")))
    path_ratio = float(best.get("storage_path_ratio", 0.0) or 0.0)
    if top_path and path_ratio >= 0.60:
        safe_path = _pick_existing_storage_path(storage_paths, [top_path])
        if safe_path:
            suggestion["storage_path"] = safe_path

    top_tags = [str(t) for t in (best.get("top_tags") or []) if _normalize_text(str(t))]
    tag_ratios = best.get("tag_ratios") or {}
    prior_tags = []
    for tag in top_tags:
        ratio = float(tag_ratios.get(tag, 0.0) or 0.0)
        if ratio >= 0.50 and len(prior_tags) < MAX_TAGS_PER_DOC:
            prior_tags.append(tag)
    if prior_tags:
        suggestion["tags"] = prior_tags

    return suggestion


def _select_controlled_tags(suggested_tags: list, existing_tags: list, taxonomy: TagTaxonomy | None = None,
                            run_db: LocalStateDB | None = None, run_id: int | None = None,
                            doc_id: int | None = None) -> tuple[list, list]:
    """Select tags from existing catalog; optionally allow creation via env."""
    tag_lookup = {}
    for tag in existing_tags:
        normalized = _normalize_tag_name(tag.get("name", ""))
        if normalized:
            tag_lookup[normalized] = tag["name"]

    approved = []
    approved_norm = set()
    dropped = []
    seen = set()

    def add_approved_once(tag_name: str, raw_tag: str):
        normalized_approved = _normalize_tag_name(tag_name)
        if not normalized_approved:
            return
        if normalized_approved in approved_norm:
            dropped.append((raw_tag, f"duplikat->'{tag_name}'"))
            if run_db:
                run_db.record_tag_event(run_id, doc_id, "dropped_duplicate", raw_tag, f"duplicate canonical '{tag_name}'")
            return
        approved.append(tag_name)
        approved_norm.add(normalized_approved)

    for raw_tag in suggested_tags or []:
        normalized = _normalize_tag_name(raw_tag)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        canonical = None
        if taxonomy and ENFORCE_TAG_TAXONOMY:
            canonical = taxonomy.canonical_from_any(raw_tag)
            if canonical is None:
                canonical = taxonomy.canonical_from_any(normalized)
            if canonical is None:
                best_tax = difflib.get_close_matches(
                    normalized,
                    [_normalize_tag_name(t) for t in taxonomy.canonical_tags],
                    n=1,
                    cutoff=TAG_MATCH_THRESHOLD,
                )
                if best_tax:
                    canonical = taxonomy.canonical_from_any(best_tax[0])
            if canonical is None:
                dropped.append((raw_tag, "nicht in Taxonomie"))
                if run_db:
                    run_db.record_tag_event(run_id, doc_id, "blocked", raw_tag, "not in taxonomy")
                continue

        normalized_target = _normalize_tag_name(canonical or raw_tag)

        if normalized_target in tag_lookup:
            add_approved_once(tag_lookup[normalized_target], raw_tag)
            continue

        best = difflib.get_close_matches(normalized_target, list(tag_lookup.keys()), n=1, cutoff=TAG_MATCH_THRESHOLD)
        if best:
            add_approved_once(tag_lookup[best[0]], raw_tag)
            dropped.append((raw_tag, f"mapped to existing '{tag_lookup[best[0]]}'"))
            if run_db:
                run_db.record_tag_event(run_id, doc_id, "mapped", raw_tag, f"mapped to existing '{tag_lookup[best[0]]}'")
            continue

        if canonical and AUTO_CREATE_TAXONOMY_TAGS:
            add_approved_once(canonical, raw_tag)
            continue

        if ALLOW_NEW_TAGS and not taxonomy:
            add_approved_once(_normalize_text(raw_tag), raw_tag)
            continue

        dropped.append((raw_tag, "blocked by tag policy (existing-only)"))
        if run_db:
            run_db.record_tag_event(run_id, doc_id, "blocked", raw_tag, "blocked by tag policy (existing-only)")

    approved = approved[:MAX_TAGS_PER_DOC]
    return approved, dropped


def _resolve_path_id(path_value: str, storage_paths: list) -> int | None:
    if not path_value:
        return None
    key = path_value.lower()
    path_name_map = {p["name"].lower(): p["id"] for p in storage_paths}
    path_path_map = {p["path"].lower(): p["id"] for p in storage_paths}
    if key in path_name_map:
        return path_name_map[key]
    if key in path_path_map:
        return path_path_map[key]
    for p in storage_paths:
        if p["name"].lower().startswith(key) or key.startswith(p["name"].lower()):
            return p["id"]
        if p["path"].lower().startswith(key) or key in p["path"].lower():
            return p["id"]
    return None


def _collect_hard_review_reasons(document: dict, suggestion: dict, selected_tags: list, storage_paths: list,
                                 decision_context: DecisionContext | None = None) -> list[str]:
    reasons = []
    is_prior_only = str(suggestion.get("confidence", "")).strip().lower() == "prior_only"
    confidence = str(suggestion.get("confidence", "high")).strip().lower()
    if REVIEW_ON_MEDIUM_CONFIDENCE and confidence in ("low", "medium"):
        reasons.append(f"confidence={confidence}")

    if not _normalize_text(suggestion.get("title", "")) and not _normalize_text(document.get("title", "")):
        reasons.append("kein Titel")

    doc_type = _normalize_text(suggestion.get("document_type", ""))
    if not doc_type and not document.get("document_type"):
        if not is_prior_only:
            reasons.append("kein Dokumenttyp")
    elif doc_type and doc_type.lower() not in {t.lower() for t in ALLOWED_DOC_TYPES} and not document.get("document_type"):
        reasons.append(f"dokumenttyp-unbekannt:{doc_type}")

    if not selected_tags and not document.get("tags"):
        reasons.append("keine gueltigen Tags")

    path_value = _normalize_text(suggestion.get("storage_path", ""))
    if not path_value and not document.get("storage_path"):
        if not is_prior_only:
            reasons.append("kein Speicherpfad")
    elif path_value and _resolve_path_id(path_value, storage_paths) is None and not ALLOW_NEW_STORAGE_PATHS and not document.get("storage_path"):
        reasons.append(f"speicherpfad-unbekannt:{path_value}")

    text = " ".join([
        str(document.get("title", "")),
        str(document.get("original_file_name", "")),
        str(document.get("content", ""))[:3000],
    ]).lower()
    vendor_key = _find_vendor_key(text)
    suggested_corr = _normalize_tag_name(str(suggestion.get("correspondent", "")))
    employer_hints = _effective_employer_hints(decision_context)
    if vendor_key and suggested_corr in employer_hints:
        reasons.append(f"anbieter-arbeitskonflikt:{vendor_key}")
    if vendor_key and path_value.lower().startswith("arbeit/"):
        reasons.append(f"anbieter-arbeitsordner:{vendor_key}")

    # OCR quality check - flag poor OCR for review
    ocr_quality, _ = _assess_ocr_quality(document)
    if ocr_quality == "poor" and not is_prior_only:
        reasons.append("ocr-qualitaet-schlecht")

    return reasons


def _mark_document_for_review(paperless: PaperlessClient, document: dict, tags: list, reason: str):
    if not AUTO_APPLY_REVIEW_TAG:
        return

    review_tag_id = None
    for tag in tags:
        if _normalize_tag_name(tag.get("name", "")) == _normalize_tag_name(REVIEW_TAG_NAME):
            review_tag_id = tag["id"]
            break

    if review_tag_id is None:
        try:
            created = paperless.create_tag(REVIEW_TAG_NAME)
            tags.append(created)
            review_tag_id = created["id"]
        except Exception as exc:
            log.warning(f"Review-Tag konnte nicht erstellt werden: {exc}")
            return

    current = list(document.get("tags") or [])
    if review_tag_id in current:
        return
    current.append(review_tag_id)
    try:
        paperless.update_document(document["id"], {"tags": current})
        log.info(f"  Dokument #{document['id']} als Review markiert ({REVIEW_TAG_NAME}) - {reason}")
    except Exception as exc:
        log.warning(f"Review-Markierung fehlgeschlagen fuer #{document['id']}: {exc}")


def _http_error_detail(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    status = getattr(response, "status_code", "?")
    text = (getattr(response, "text", "") or "").strip()
    if text:
        return f"HTTP {status}: {text[:1000]}"
    return f"HTTP {status}"


def _apply_update_with_fallbacks(paperless: PaperlessClient, doc_id: int, update_data: dict) -> tuple[dict, list[str]]:
    """
    Update with guarded retries for common 400 causes.
    Keeps automation running instead of hard-failing on one invalid field.
    """
    payload = dict(update_data)
    notes: list[str] = []

    for attempt in range(1, 6):
        try:
            result = paperless.update_document(doc_id, payload)
            return result, notes
        except requests.exceptions.HTTPError as exc:
            detail = _http_error_detail(exc)
            lower = detail.lower()
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status != 400:
                raise RuntimeError(detail) from exc

            changed = False

            # Most frequent issue: duplicate/invalid ASN.
            if "archive_serial" in lower and "archive_serial_number" in payload:
                payload.pop("archive_serial_number", None)
                notes.append("retry_without_archive_serial_number")
                changed = True

            # Field-specific fallback if API rejects IDs.
            if "storage_path" in lower and "storage_path" in payload:
                payload.pop("storage_path", None)
                notes.append("retry_without_storage_path")
                changed = True
            if "correspondent" in lower and "correspondent" in payload:
                payload.pop("correspondent", None)
                notes.append("retry_without_correspondent")
                changed = True
            if "document_type" in lower and "document_type" in payload:
                payload.pop("document_type", None)
                notes.append("retry_without_document_type")
                changed = True
            if "tags" in lower and "tags" in payload:
                payload.pop("tags", None)
                notes.append("retry_without_tags")
                changed = True

            # Generic fallback once: minimal safe update (title + optional tags).
            if not changed and attempt == 1:
                minimal = {}
                if update_data.get("title"):
                    minimal["title"] = update_data["title"]
                if update_data.get("tags"):
                    minimal["tags"] = update_data["tags"]
                if minimal:
                    payload = minimal
                    notes.append("retry_minimal_payload")
                    changed = True

            if not changed:
                raise RuntimeError(detail) from exc

    raise RuntimeError("update failed after fallback retries")


class LocalLLMAnalyzer:
    """Analysiert Dokumente mit lokalem LLM-Endpunkt."""

    def __init__(self, url: str = LLM_URL, model: str = LLM_MODEL):
        self.url = self._normalize_url(url)
        self.model = (model or "").strip()

    @staticmethod
    def _normalize_url(url: str) -> str:
        parsed = urlparse((url or "").strip())
        path = (parsed.path or "").strip()
        if path in ("", "/"):
            if parsed.port == 11434:
                parsed = parsed._replace(path="/api/chat")
            else:
                parsed = parsed._replace(path="/v1/chat/completions")
            return urlunparse(parsed)
        return url

    def _auth_headers(self) -> dict | None:
        return {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else None

    def _candidate_model_urls(self) -> list[str]:
        candidates: list[str] = []
        if "/v1/chat/completions" in self.url:
            candidates.append(self.url.replace("/v1/chat/completions", "/v1/models"))
        if "/api/v1/chat" in self.url:
            candidates.append(self.url.replace("/api/v1/chat", "/v1/models"))
            candidates.append(self.url.replace("/api/v1/chat", "/api/v1/models"))
        if "/api/chat" in self.url:
            candidates.append(self.url.replace("/api/chat", "/api/tags"))
            candidates.append(self.url.replace("/api/chat", "/v1/models"))
        return list(dict.fromkeys(candidates))

    def _discover_model(self, headers: dict | None) -> str:
        preferred = [
            "qwen2.5:14b",
            "qwen2.5:7b",
            "qwen3:8b",
            "qwen2.5-coder:14b",
            "qwen2.5-coder:7b",
            "gemma3:12b",
            "gemma3:4b",
            "google/gemma-3-12b",
            "google/gemma-3-4b",
            "gemma3:latest",
            "llama3.1:8b",
            "mistral:7b",
            "phi-4:14b",
            "llama3.2:3b",
        ]
        discovered: list[str] = []
        for test_url in self._candidate_model_urls():
            try:
                resp = requests.get(test_url, headers=headers, timeout=5)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                models = data.get("data")
                if not isinstance(models, list):
                    models = data.get("models")
                if not isinstance(models, list):
                    continue
                for entry in models:
                    if not isinstance(entry, dict):
                        continue
                    model_id = entry.get("id") or entry.get("name") or entry.get("key") or entry.get("model")
                    if isinstance(model_id, str) and model_id.strip():
                        discovered.append(model_id.strip())
            except (requests.exceptions.RequestException, ValueError):
                continue
        if not discovered:
            return ""
        unique = list(dict.fromkeys(discovered))
        lower_map = {m.lower(): m for m in unique}
        for pref in preferred:
            hit = lower_map.get(pref.lower())
            if hit:
                return hit
        return unique[0]

    def verify_connection(self) -> bool:
        """Prueft ob der konfigurierte LLM-Server erreichbar ist."""
        log.info(f"Teste LLM-Verbindung: {self.url}")
        headers = self._auth_headers()
        try:
            for test_url in self._candidate_model_urls():
                resp = requests.get(test_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    if not self.model:
                        self.model = self._discover_model(headers)
                    model_label = self.model or "Server-Default"
                    log.info(f"[green]LLM verbunden[/green] - Modell: {model_label}")
                    return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.RequestException:
            pass
        # Fallback: Endpoint direkt mit kleinem Probe-Payload pruefen.
        try:
            probe = requests.post(
                self.url,
                headers=headers,
                json=self._build_payload("ping"),
                timeout=5,
            )
            if probe.status_code < 500 and probe.status_code != 404:
                if not self.model:
                    self.model = self._discover_model(headers)
                model_label = self.model or "Server-Default"
                log.info(f"[green]LLM-Endpunkt erreichbar[/green] - Modell: {model_label}")
                return True
        except requests.exceptions.RequestException:
            pass
        log.error(f"LLM-Server nicht erreichbar! ({self.url})")
        console.print("[yellow]Bitte LLM-Server starten und Modell laden.[/yellow]")
        return False

    def _parse_json_response(self, text: str) -> dict:
        """JSON aus LLM-Antwort extrahieren (```json``` Fences strippen)."""
        text = text.strip()
        # JSON-Fence entfernen
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        # Finde JSON-Objekt
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        return json.loads(text)

    def analyze(self, document: dict, existing_tags: list,
                existing_correspondents: list, existing_types: list,
                existing_paths: list, taxonomy: TagTaxonomy | None = None,
                decision_context: DecisionContext | None = None,
                few_shot_examples: list[dict] | None = None,
                learning_hints: list[dict] | None = None,
                compact_mode: bool = False,
                read_timeout_override: int | None = None,
                web_hint_override: str | None = None) -> dict:
        """Analysiert ein Dokument und gibt Organisationsvorschlag zurueck."""

        # Top-Korrespondenten nach Dokumentanzahl (Token sparen)
        top_corrs = sorted(existing_correspondents, key=lambda c: c.get("document_count", 0), reverse=True)
        max_corr_choices = 12 if compact_mode else 35
        corr_names = [c["name"] for c in top_corrs[:max_corr_choices]]
        if taxonomy and ENFORCE_TAG_TAXONOMY and taxonomy.canonical_tags:
            if compact_mode:
                max_tag_choices = max(8, min(MAX_PROMPT_TAG_CHOICES, LLM_COMPACT_PROMPT_MAX_TAGS))
            else:
                max_tag_choices = min(MAX_PROMPT_TAG_CHOICES, 90)
            tag_choices = taxonomy.prompt_tags(max_tag_choices)
        else:
            top_tags = sorted(existing_tags, key=lambda t: t.get("document_count", 0), reverse=True)
            if compact_mode:
                max_tag_choices = max(8, min(MAX_PROMPT_TAG_CHOICES, LLM_COMPACT_PROMPT_MAX_TAGS))
            else:
                max_tag_choices = min(MAX_PROMPT_TAG_CHOICES, 90)
            tag_choices = [t["name"] for t in top_tags[:max_tag_choices]]

        current_tags = [
            t["name"] for t in existing_tags
            if t["id"] in (document.get("tags") or [])
        ]
        current_corr = next(
            (c["name"] for c in existing_correspondents
             if c["id"] == document.get("correspondent")), "")
        current_type = next(
            (t["name"] for t in existing_types
             if t["id"] == document.get("document_type")), "")
        current_path = next(
            (p["name"] for p in existing_paths
             if p["id"] == document.get("storage_path")), "")

        # Token-effizient: Header + Footer enthalten oft die wichtigsten Infos
        content = document.get("content") or ""
        content_len = len(content)
        if compact_mode:
            if content_len > 1200:
                # Header (Briefkopf) + Footer (Signatur/Impressum)
                head = content[:400]
                tail = content[-200:] if content_len > 800 else ""
                content_preview = head + f"\n[...{content_len} Zeichen, Kompaktmodus...]\n" + tail
            else:
                content_preview = content[:600]
        else:
            if content_len > 5000:
                head = content[:1000]
                tail = content[-400:]
                content_preview = head + f"\n[...{content_len} Zeichen insgesamt...]\n" + tail
            elif content_len > 2000:
                content_preview = content[:1200] + f"\n[...{content_len} Zeichen...]\n" + content[-300:]
            else:
                content_preview = content

        brand_hint = _get_brand_hint(f"{document.get('title', '')} {document.get('original_file_name', '')} {content_preview[:1000]}")
        if web_hint_override:
            web_hint = web_hint_override
        elif ENABLE_WEB_HINTS and not compact_mode:
            # Use real web search for entity hints, keep old primary hint as supplement
            web_hint_primary = _fetch_web_hint(document.get("title", ""), content_preview)
            web_hint_entities = _collect_web_entity_hints(document, current_corr=current_corr)
            # Also try real web search for the current correspondent if unknown
            web_search_corr = ""
            if current_corr and current_corr not in KNOWN_BRAND_HINTS:
                web_search_corr = _web_search_entity(current_corr)
            web_hint = " | ".join([h for h in [web_hint_primary, web_hint_entities, web_search_corr] if h])
        else:
            web_hint = ""

        if decision_context:
            employer_list = sorted(decision_context.employer_names)[:max(1, MAX_CONTEXT_EMPLOYER_HINTS)]
            provider_list = sorted(decision_context.provider_names)[:10]
            employers_info = ", ".join(employer_list) if employer_list else "keine"
            providers_info = ", ".join(provider_list) if provider_list else "keine"
        else:
            employers_info = "keine"
            providers_info = "keine"
        work_paths_info = ", ".join(decision_context.top_work_paths) if decision_context else "keine"
        private_paths_info = ", ".join(decision_context.top_private_paths) if decision_context else "keine"
        profile_context = (
            decision_context.profile_context_text
            if decision_context and decision_context.profile_context_text
            else (
                "Person: Edgar Richter.\n"
                "Beschaeftigungsverlauf: seit 2025-08-01 WBS TRAINING AG; davor msg systems ag.\n"
                "Privatfahrzeuge: VW Polo.\n"
                "Firmenfahrzeuge: Toyota.\n"
                "Weitere Hinweise: DRK-Mitglied, AOK PLUS."
            )
        )
        examples_text = "keine"
        if few_shot_examples and not compact_mode:
            lines = []
            for idx, ex in enumerate(few_shot_examples[:LEARNING_EXAMPLE_LIMIT], 1):
                lines.append(
                    f"{idx}) Titel~{ex.get('doc_title', '?')} -> "
                    f"Korr={ex.get('correspondent', '?')}, "
                    f"Typ={ex.get('document_type', '?')}, "
                    f"Pfad={ex.get('storage_path', '?')}, "
                    f"Tags={', '.join(ex.get('tags') or []) or 'keine'}"
                )
            examples_text = "\n".join(lines)

        learning_hint_text = "keine"
        if learning_hints and not compact_mode:
            lines = []
            for idx, hint in enumerate(learning_hints[:LEARNING_PRIOR_MAX_HINTS], 1):
                corr = hint.get("correspondent", "?")
                count = int(hint.get("count", 0) or 0)
                top_type = hint.get("top_document_type", "") or "?"
                top_type_ratio = float(hint.get("document_type_ratio", 0.0) or 0.0)
                top_path = hint.get("top_storage_path", "") or "?"
                top_path_ratio = float(hint.get("storage_path_ratio", 0.0) or 0.0)
                top_tags = hint.get("top_tags") or []
                lines.append(
                    f"{idx}) {corr} (n={count}) -> Typ {top_type} ({top_type_ratio:.0%}), "
                    f"Pfad {top_path} ({top_path_ratio:.0%}), Tags {', '.join(top_tags) or 'keine'}"
                )
            learning_hint_text = "\n".join(lines)

        path_names = [p["name"] for p in existing_paths if "Duplikat" not in p["name"]]
        if compact_mode:
            path_names = path_names[:max(8, LLM_COMPACT_PROMPT_MAX_PATHS)]
            prompt = f"""Paperless-NGX: Dokument klassifizieren und zuordnen.
DOKUMENT #{document['id']}:
Titel: {document.get('title', '?')} | Datei: {document.get('original_file_name', '?')}
Aktuell: Tags={current_tags or 'keine'} | Korr={current_corr or 'keiner'} | Typ={current_type or 'keiner'} | Pfad={current_path or 'keiner'}
{f'WEB-KONTEXT: {web_hint}' if web_hint else ''}

INHALT (AUSZUG):
{content_preview}

ERLAUBTE KORRESPONDENTEN: {', '.join(corr_names)}
ERLAUBTE DOKUMENTTYPEN: {', '.join(ALLOWED_DOC_TYPES)}
ERLAUBTE SPEICHERPFADE: {', '.join(path_names)}
ERLAUBTE TAGS: {', '.join(tag_choices)}

FELDER (alle Pflichtfelder):
- title: Aussagekraeftiger deutscher Titel (z.B. "Rechnung Telekom Februar 2025")
- correspondent: Der KONKRETE Absender (Firma/Behoerde) aus dem Briefkopf, MUSS aus der Liste sein
- document_type: Dokumenttyp (z.B. Rechnung, Kontoauszug, Vertrag), MUSS aus der Liste sein
- storage_path: Kategorie-Pfad EXAKT aus der Liste (z.B. "Finanzen/Bank")
- tags: 1-{MAX_TAGS_PER_DOC} Tags NUR aus der Liste, keine neuen erfinden
- confidence: "high" (sicher), "medium" (wahrscheinlich), "low" (unsicher)
- reasoning: Kurze Begruendung

NUR JSON:
{{"title": "Titel", "tags": ["Tag1"], "correspondent": "Firma", "document_type": "Typ", "storage_path_name": "Kategorie", "storage_path": "Kategorie/Sub", "confidence": "high", "reasoning": "Kurz"}}"""
        else:
            prompt = f"""Paperless-NGX Dokument organisieren.
{profile_context}

DOKUMENT #{document['id']}:
Titel: {document.get('title', '?')} | Datei: {document.get('original_file_name', '?')} | Erstellt: {document.get('created', '?')}
Tags: {current_tags or 'keine'} | Korr: {current_corr or 'keiner'} | Typ: {current_type or 'keiner'} | Pfad: {current_path or 'keiner'}

INHALT:
{content_preview}

KORRESPONDENTEN (bevorzuge vorhandene): {', '.join(corr_names)}

DOKUMENTTYPEN (NUR diese): {', '.join(ALLOWED_DOC_TYPES)}

SPEICHERPFADE (NUR diese Kategorienamen verwenden, NICHT den vollen Pfad mit Jahr/Titel!):
{', '.join(path_names)}
WICHTIG: storage_path muss EXAKT einem der obigen Namen entsprechen, z.B. "Auto/Unfall" oder "Finanzen/Bank"

ERLAUBTE TAGS (NUR aus dieser Liste waehlen, keine neuen erfinden):
{', '.join(tag_choices)}
WICHTIG: maximal {MAX_TAGS_PER_DOC} Tags und nur aus der obigen Liste.

BRAND-HINWEIS: {brand_hint or 'kein spezieller Hinweis'}
WEB-HINWEIS: {web_hint or 'kein externer Treffer'}
KONTEXT (vorher gesammelt):
- erkannte Arbeitgeber im Bestand: {employers_info}
- erkannte externe Anbieter: {providers_info}
- haeufige Arbeitspfade: {work_paths_info}
- haeufige Nicht-Arbeitspfade: {private_paths_info}
- bestaetigte aehnliche Beispiele:
{examples_text}
- lernende Korrespondent-Hinweise (aus bestaetigten Faellen):
{learning_hint_text}

ZUORDNUNGSREGELN - GENAU BEACHTEN:
- Arbeit/msg: NUR Dokumente die DIREKT von msg systems ag stammen (Arbeitsvertrag, Zeugnis, Gehaltsabrechnung MIT msg im Absender/Inhalt)
- Arbeit/WBS: NUR Dokumente die DIREKT von WBS TRAINING AG stammen (Arbeitsvertrag, Gehaltsabrechnung MIT WBS im Absender/Inhalt)
- NIEMALS Arbeitgeber als Korrespondent setzen, wenn im Dokument klar ein externer Anbieter steht (z.B. Google Cloud, GitHub, JetBrains, OpenAI, ElevenLabs)
- NICHT zu Arbeit: Software-Abos (Claude, GitHub, JetBrains, etc.), Weiterbildungen die privat bezahlt werden, private Cloud-Dienste, Technik-Kaeufe
- Software-Abos, KI-Dienste, Cloud-Dienste, Hosting -> Freizeit/IT oder Finanzen je nach Kontext
- Korrespondent muss der KONKRETE Absender sein, nicht ein Oberbegriff.
- Verwandte Organisationen NICHT zusammenziehen (z.B. Tochterfirma, Abteilung, Dienst, Klinik, Kreisverband, Wasserwacht, Blutspendedienst sind getrennte Absender).
- Bei Mehrdeutigkeit immer offiziellen Absender aus Briefkopf/Fusszeile bevorzugen.
- Wenn unklar: bestehenden Korrespondenten beibehalten statt neuen erfinden.
- Versicherungen -> Versicherungen/[Typ]
- Bank/Finanzen/Depot -> Finanzen/Bank oder Finanzen/Depot
- Arzt/Gesundheit/AOK -> Gesundheit
- VW Polo / Polo / Golf Polo = PRIVAT einordnen (nicht Arbeit)
- Toyota = Firmenwagen-Kontext (arbeitsbezogen), nicht automatisch privat einordnen
- Frage dich IMMER: Ist das ein ARBEITSDOKUMENT (direkt vom Arbeitgeber) oder PRIVAT?
- Im Zweifel: PRIVAT einordnen, nicht Arbeit!

WEITERE REGELN: Tags kurz und sinnvoll. Titel deutsch, aussagekraeftig. Korrespondent=Absender/Firma.
confidence: Wie sicher bist du dir bei der Zuordnung? "high", "medium" oder "low".
Antwort moeglichst kurz und strukturiert.

NUR JSON, kein anderer Text:
{{"title": "Titel", "tags": ["Tag1", "Tag2"], "correspondent": "Firma", "document_type": "Typ", "storage_path_name": "Kategorie", "storage_path": "Kategorie/Sub", "confidence": "high", "reasoning": "Kurz"}}"""

        effective_read_timeout = (
            int(read_timeout_override)
            if read_timeout_override is not None
            else (LLM_COMPACT_TIMEOUT if compact_mode else LLM_TIMEOUT)
        )

        # LLM-Anfrage
        response = self._call_llm(
            prompt,
            read_timeout=effective_read_timeout,
            retries=1 if compact_mode else LLM_RETRY_COUNT,
            max_tokens=LLM_COMPACT_MAX_TOKENS if compact_mode else LLM_MAX_TOKENS,
        )

        # JSON parsen - bei Fehler Retry mit Reasoning-Prompt
        try:
            suggestion = self._parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            log.info("[yellow]JSON-Parse-Fehler, Retry mit Reasoning-Prompt...[/yellow]")
            retry_prompt = (
                "Analysiere das folgende Dokument Schritt fuer Schritt:\n"
                "1. Wer ist der Absender/Korrespondent?\n"
                "2. Welcher Dokumenttyp passt am besten?\n"
                "3. In welchen Speicherpfad gehoert es?\n"
                "4. Welche Tags sind relevant?\n"
                "5. Wie sicher bist du dir?\n\n"
                f"{prompt}\n\n"
                "WICHTIG: Antworte AUSSCHLIESSLICH mit einem einzigen JSON-Objekt. "
                "Kein Text davor oder danach."
            )
            response = self._call_llm(
                retry_prompt,
                read_timeout=effective_read_timeout,
                retries=1,
                max_tokens=LLM_COMPACT_MAX_TOKENS if compact_mode else LLM_MAX_TOKENS,
            )
            suggestion = self._parse_json_response(response)

        # Bei niedriger Konfidenz: Verifikation mit zweitem LLM-Aufruf
        confidence = suggestion.get("confidence", "high").lower()
        if LLM_VERIFY_ON_LOW_CONFIDENCE and not compact_mode and confidence in ("low", "medium"):
            log.info(f"  [yellow]Konfidenz: {confidence}[/yellow] -> Verifiziere Zuordnung...")
            suggestion = self._verify_suggestion(document, suggestion, content_preview)

        return suggestion

    def _verify_suggestion(self, document: dict, suggestion: dict, content_preview: str) -> dict:
        """Zweiter LLM-Aufruf zur Verifikation bei unsicherer Zuordnung."""
        verify_prompt = f"""Ich habe folgendes Dokument analysiert und bin mir bei der Zuordnung unsicher.
Pruefe meine Zuordnung und korrigiere sie wenn noetig.

DOKUMENT #{document['id']}:
Titel: {document.get('title', '?')} | Datei: {document.get('original_file_name', '?')}

INHALT (Auszug):
{content_preview[:1000]}

MEINE ZUORDNUNG:
- Titel: {suggestion.get('title', '?')}
- Korrespondent: {suggestion.get('correspondent', '?')}
- Dokumenttyp: {suggestion.get('document_type', '?')}
- Speicherpfad: {suggestion.get('storage_path', '?')}
- Tags: {suggestion.get('tags', [])}
- Begruendung: {suggestion.get('reasoning', '?')}

WICHTIGE REGELN:
- Arbeit/msg oder Arbeit/WBS NUR wenn das Dokument DIREKT vom Arbeitgeber stammt (Vertrag, Zeugnis, Gehaltsabrechnung)
- Software-Abos (Claude, GitHub, JetBrains), Cloud-Dienste, Hosting = PRIVAT, nicht Arbeit
- Im Zweifel PRIVAT einordnen
- Ist der Korrespondent korrekt? Das muss der tatsaechliche Absender/die Firma sein die das Dokument erstellt hat.

Antworte NUR mit korrigiertem JSON (oder identischem wenn alles stimmt):
{{"title": "Titel", "tags": ["Tag1", "Tag2"], "correspondent": "Firma", "document_type": "Typ", "storage_path_name": "Kategorie", "storage_path": "Kategorie/Sub", "confidence": "high", "reasoning": "Kurz"}}"""

        response = self._call_llm(
            verify_prompt,
            read_timeout=LLM_COMPACT_TIMEOUT,
            retries=1,
            max_tokens=LLM_COMPACT_MAX_TOKENS,
        )
        try:
            verified = self._parse_json_response(response)
            if verified.get("storage_path") != suggestion.get("storage_path"):
                log.info(f"  [green]Korrigiert:[/green] {suggestion.get('storage_path')} -> {verified.get('storage_path')}")
            return verified
        except (json.JSONDecodeError, ValueError):
            log.info("  [yellow]Verifikation fehlgeschlagen, nutze Original[/yellow]")
            return suggestion

    def _use_simple_chat_api(self) -> bool:
        return self.url.rstrip("/").endswith("/api/v1/chat")

    def _use_ollama_chat_api(self) -> bool:
        return self.url.rstrip("/").endswith("/api/chat")

    def _build_payload(self, prompt: str, max_tokens: int | None = None) -> dict:
        token_limit = max(64, int(max_tokens if max_tokens is not None else LLM_MAX_TOKENS))
        if self._use_ollama_chat_api():
            messages = []
            if LLM_SYSTEM_PROMPT:
                messages.append({"role": "system", "content": LLM_SYSTEM_PROMPT})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": token_limit,
                },
                "format": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "correspondent": {"type": "string"},
                        "document_type": {"type": "string"},
                        "storage_path_name": {"type": "string"},
                        "storage_path": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["title", "tags", "correspondent", "document_type", "storage_path", "confidence"],
                },
            }
            if self.model:
                payload["model"] = self.model
            if LLM_KEEP_ALIVE:
                payload["keep_alive"] = LLM_KEEP_ALIVE
            return payload
        if self._use_simple_chat_api():
            payload = {
                "input": prompt,
            }
            if self.model:
                payload["model"] = self.model
            if LLM_SYSTEM_PROMPT:
                payload["system_prompt"] = LLM_SYSTEM_PROMPT
            return payload
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": token_limit,
        }
        if self.model:
            payload["model"] = self.model
        return payload

    def _extract_response_text(self, data: dict) -> str:
        output = data.get("output")
        if isinstance(output, list) and output:
            for entry in output:
                if isinstance(entry, dict):
                    content = entry.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        if isinstance(output, str) and output.strip():
            return output.strip()
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
            text = first_choice.get("text")
            if isinstance(text, str):
                return text.strip()
        for key in ("output", "response", "text"):
            value = data.get(key)
            if isinstance(value, str):
                return value.strip()
        message = data.get("message")
        if isinstance(message, str):
            return message.strip()
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        keys = ", ".join(sorted(data.keys())[:8])
        raise RuntimeError(f"Unbekanntes LLM-Antwortformat (keys: {keys})")

    @staticmethod
    def _error_snippet(resp: requests.Response) -> str:
        text = (resp.text or "").strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:300] + "..."
        return text

    def _post_with_retry(
        self,
        headers: dict | None,
        payload: dict,
        read_timeout: int | None = None,
        retries: int | None = None,
    ) -> requests.Response:
        last_resp = None
        last_exc: Exception | None = None
        connect_timeout = max(3, LLM_CONNECT_TIMEOUT)
        effective_read_timeout = max(5, int(read_timeout if read_timeout is not None else LLM_TIMEOUT))
        max_attempts = max(1, int(retries if retries is not None else LLM_RETRY_COUNT))
        for attempt in range(max_attempts):
            try:
                resp = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=(connect_timeout, effective_read_timeout),
                )
                last_resp = resp
                if resp.ok:
                    return resp
                # Manche lokalen Server liefern kurzzeitig leere 400/5xx bei Last.
                transient_400 = resp.status_code == 400 and not (resp.text or "").strip()
                if resp.status_code in (429, 500, 502, 503, 504) or transient_400:
                    if attempt < max_attempts - 1:
                        time.sleep(0.8 + attempt * 0.8)
                        continue
                    continue
                return resp
            except requests.exceptions.Timeout:
                # Timeout sofort nach oben geben. Der Aufrufer steuert den Kompakt-Retry.
                raise
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    time.sleep(0.8 + attempt * 0.8)
                    continue
        if last_exc is not None:
            raise last_exc
        return last_resp

    def _call_llm(
        self,
        prompt: str,
        read_timeout: int | None = None,
        retries: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Sendet Prompt an LLM-Endpunkt und gibt Antwort zurueck."""
        headers = self._auth_headers()
        payload = self._build_payload(prompt, max_tokens=max_tokens)
        resp = self._post_with_retry(headers, payload, read_timeout=read_timeout, retries=retries)

        # Retry 1: if model handling might be the issue, use server-default once.
        if resp.status_code in (400, 422) and self.model:
            model_backup = self.model
            self.model = ""
            retry_payload = self._build_payload(prompt, max_tokens=max_tokens)
            retry_resp = self._post_with_retry(headers, retry_payload, read_timeout=read_timeout, retries=1)
            if retry_resp.ok:
                resp = retry_resp
            else:
                self.model = model_backup

        # Retry 2: discover model automatically if model is empty.
        if resp.status_code in (400, 422) and not self.model:
            detected_model = self._discover_model(headers)
            if detected_model:
                self.model = detected_model
                retry_payload = self._build_payload(prompt, max_tokens=max_tokens)
                retry_resp = self._post_with_retry(headers, retry_payload, read_timeout=read_timeout, retries=1)
                if retry_resp.ok:
                    resp = retry_resp

        if not resp.ok:
            detail = self._error_snippet(resp)
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for url: {self.url} | body: {detail}",
                response=resp,
            )
        return self._extract_response_text(resp.json())


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ID-Aufloesung & Anzeige (aus main.py)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def resolve_ids(paperless: PaperlessClient, suggestion: dict,
                tags: list, correspondents: list,
                doc_types: list, storage_paths: list,
                taxonomy: TagTaxonomy | None = None,
                run_db: LocalStateDB | None = None,
                run_id: int | None = None,
                doc_id: int | None = None) -> dict:
    """Wandelt Namen in IDs um, erstellt fehlende Eintraege."""

    # Tags
    tag_map = {t["name"].lower(): t["id"] for t in tags}
    tag_ids = []
    selected_tags, dropped_tags = _select_controlled_tags(
        suggestion.get("tags", []),
        tags,
        taxonomy=taxonomy,
        run_db=run_db,
        run_id=run_id,
        doc_id=doc_id,
    )

    for dropped_tag, reason in dropped_tags:
        log.info(f"  [yellow]Tag verworfen:[/yellow] {dropped_tag} ({reason})")

    for tag_name in selected_tags:
        key = tag_name.lower()
        if key in tag_map:
            tag_ids.append(tag_map[key])
        else:
            can_create = ALLOW_NEW_TAGS or (
                taxonomy is not None
                and AUTO_CREATE_TAXONOMY_TAGS
                and taxonomy.canonical_from_any(tag_name) is not None
            )
            if not can_create:
                log.info(f"  [yellow]Tag nicht erstellt (Policy):[/yellow] {tag_name}")
                if run_db:
                    run_db.record_tag_event(run_id, doc_id, "blocked", tag_name, "new tags disabled")
                continue
            if len(tags) >= MAX_TOTAL_TAGS:
                detail = f"global tag limit reached ({MAX_TOTAL_TAGS})"
                log.warning(f"  [yellow]Tag nicht erstellt (Limit):[/yellow] {tag_name} - {detail}")
                if run_db:
                    run_db.record_tag_event(run_id, doc_id, "blocked", tag_name, detail)
                continue
            try:
                console.print(f"  [yellow]+ Neuer Tag:[/yellow] {tag_name}")
                create_color = taxonomy.color_for_tag(tag_name) if taxonomy else None
                new = paperless.create_tag(tag_name, color=create_color)
                tag_ids.append(new["id"])
                tag_map[key] = new["id"]
                tags.append(new)
                if run_db:
                    detail = f"created color={create_color or 'auto'}"
                    if taxonomy:
                        detail += f" desc={taxonomy.description_for_tag(tag_name)}"
                    run_db.record_tag_event(run_id, doc_id, "created", tag_name, detail)
            except requests.exceptions.HTTPError:
                fresh_tags = paperless.get_tags()
                tag_map = {t["name"].lower(): t["id"] for t in fresh_tags}
                if key in tag_map:
                    tag_ids.append(tag_map[key])
                    tags.clear()
                    tags.extend(fresh_tags)
    tag_ids = list(dict.fromkeys(tag_ids))

    # Korrespondent
    corr_map = {c["name"].lower(): c["id"] for c in correspondents}
    corr_id = None
    corr_name_raw = suggestion.get("correspondent", "")
    resolved_corr_id, resolved_corr_name = _resolve_correspondent_from_name(correspondents, corr_name_raw)
    if resolved_corr_name:
        suggestion["correspondent"] = resolved_corr_name
    if resolved_corr_id is not None:
        corr_id = resolved_corr_id
    elif resolved_corr_name:
        key = resolved_corr_name.lower()
        if key in corr_map:
            corr_id = corr_map[key]
        else:
            if not ALLOW_NEW_CORRESPONDENTS:
                log.info(f"  [yellow]Korrespondent nicht erstellt (Policy):[/yellow] {resolved_corr_name}")
            else:
                try:
                    console.print(f"  [yellow]+ Neuer Korrespondent:[/yellow] {resolved_corr_name}")
                    new = paperless.create_correspondent(resolved_corr_name)
                    corr_id = new["id"]
                    correspondents.append(new)
                except requests.exceptions.HTTPError:
                    fresh = paperless.get_correspondents()
                    corr_map = {c["name"].lower(): c["id"] for c in fresh}
                    corr_id = corr_map.get(key)
                    correspondents.clear()
                    correspondents.extend(fresh)

    # Dokumenttyp (nur aus erlaubter Liste)
    type_map = {t["name"].lower(): t["id"] for t in doc_types}
    type_exact = {t["name"]: t["id"] for t in doc_types}
    type_id = None
    type_name = suggestion.get("document_type", "")
    if type_name:
        allowed_lower = {t.lower(): t for t in ALLOWED_DOC_TYPES}
        key = type_name.lower()
        if key not in allowed_lower:
            console.print(f"  [red]Dokumenttyp '{type_name}' nicht erlaubt, uebersprungen.[/red]")
        else:
            canonical_name = allowed_lower[key]
            canonical_id = type_exact.get(canonical_name)
            if canonical_id is not None:
                type_id = canonical_id
            else:
                # Falls nur eine nicht-canonical Variante existiert, canonical Typ erzeugen.
                try:
                    console.print(f"  [yellow]+ Neuer Dokumenttyp:[/yellow] {canonical_name}")
                    new = paperless.create_document_type(canonical_name)
                    type_id = new["id"]
                    doc_types.append(new)
                except requests.exceptions.HTTPError:
                    fresh = paperless.get_document_types()
                    type_map = {t["name"].lower(): t["id"] for t in fresh}
                    type_exact = {t["name"]: t["id"] for t in fresh}
                    type_id = type_exact.get(canonical_name) or type_map.get(key)
                    doc_types.clear()
                    doc_types.extend(fresh)

    # Speicherpfad
    path_name_map = {p["name"].lower(): p["id"] for p in storage_paths}
    path_path_map = {p["path"].lower(): p["id"] for p in storage_paths}
    path_id = None
    path_value = suggestion.get("storage_path", "")
    if path_value:
        key = path_value.lower()
        if key in path_name_map:
            path_id = path_name_map[key]
        elif key in path_path_map:
            path_id = path_path_map[key]
        else:
            for p in storage_paths:
                if p["name"].lower().startswith(key) or key.startswith(p["name"].lower()):
                    path_id = p["id"]
                    break
                if p["path"].lower().startswith(key) or key in p["path"].lower():
                    path_id = p["id"]
                    break
        if path_id is None:
            if not ALLOW_NEW_STORAGE_PATHS:
                log.info(f"  [yellow]Neuer Speicherpfad blockiert (Policy):[/yellow] {path_value}")
                path_value = ""
            else:
                template = f"{path_value}/{{{{ created_year }}}}/{{{{ title }}}}"
                try:
                    console.print(f"  [yellow]+ Neuer Speicherpfad:[/yellow] {path_value}")
                    new = paperless.create_storage_path(path_value, template)
                    path_id = new["id"]
                    storage_paths.append(new)
                except requests.exceptions.HTTPError as e:
                    console.print(f"  [red]Speicherpfad-Fehler: {e}[/red]")

    update_data = {"title": suggestion.get("title", "")}
    if tag_ids:
        update_data["tags"] = tag_ids
    if corr_id is not None:
        update_data["correspondent"] = corr_id
    if type_id is not None:
        update_data["document_type"] = type_id
    if path_id is not None:
        update_data["storage_path"] = path_id
    if USE_ARCHIVE_SERIAL_NUMBER:
        update_data["archive_serial_number"] = paperless.get_next_asn()

    return update_data


def show_suggestion(document: dict, suggestion: dict, asn: int | None,
                    tags: list, correspondents: list,
                    doc_types: list, storage_paths: list):
    """Zeigt Vorschlag als Rich-Table an."""

    current_tags = [t["name"] for t in tags if t["id"] in (document.get("tags") or [])]
    current_corr = next((c["name"] for c in correspondents if c["id"] == document.get("correspondent")), "Keiner")
    current_type = next((t["name"] for t in doc_types if t["id"] == document.get("document_type")), "Keiner")
    current_path = next((p["name"] for p in storage_paths if p["id"] == document.get("storage_path")), "Keiner")

    def _diff_style(old: str, new: str) -> tuple[str, str]:
        """Return styled strings: dim if unchanged, colored if changed."""
        if old.lower().strip() == new.lower().strip():
            return f"[dim]{old}[/dim]", f"[dim]{new}[/dim] [green]=[/green]"
        return f"[red]{old}[/red]", f"[bold green]{new}[/bold green]"

    table = Table(title=f"Dokument #{document['id']}", show_header=True, width=80)
    table.add_column("Feld", style="cyan", width=18)
    table.add_column("Aktuell", width=28)
    table.add_column("Vorschlag", width=30)

    old_title, new_title = _diff_style(document.get("title", ""), suggestion.get("title", ""))
    old_tags_str = ", ".join(current_tags) or "Keine"
    new_tags_str = ", ".join(suggestion.get("tags", []))
    old_tags, new_tags = _diff_style(old_tags_str, new_tags_str)
    old_corr, new_corr = _diff_style(current_corr, suggestion.get("correspondent", ""))
    old_type, new_type = _diff_style(current_type, suggestion.get("document_type", ""))
    old_path, new_path = _diff_style(current_path, suggestion.get("storage_path", ""))

    table.add_row("Titel", old_title, new_title)
    table.add_row("Tags", old_tags, new_tags)
    table.add_row("Korrespondent", old_corr, new_corr)
    table.add_row("Dokumenttyp", old_type, new_type)
    table.add_row("Speicherpfad", old_path, new_path)
    current_asn = document.get("archive_serial_number")
    if asn is not None:
        asn_suggestion = str(asn)
    else:
        asn_suggestion = str(current_asn) if current_asn else "unveraendert (keine)"
    table.add_row(
        "Archivnummer",
        str(current_asn or "Keine"),
        asn_suggestion,
    )

    # Confidence display
    conf = suggestion.get("confidence", "").lower()
    conf_style = {"high": "[bold green]HOCH[/bold green]", "medium": "[yellow]MITTEL[/yellow]",
                  "low": "[red]NIEDRIG[/red]"}.get(conf, conf)
    if conf:
        table.add_row("Konfidenz", "", conf_style)

    console.print(table)
    if suggestion.get("reasoning"):
        console.print(Panel(suggestion["reasoning"], title="Begruendung", border_style="blue"))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Dokument-Verarbeitung
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def process_document(doc_id: int, paperless: PaperlessClient,
                     analyzer: LocalLLMAnalyzer, tags: list,
                     correspondents: list, doc_types: list,
                     storage_paths: list, dry_run: bool = True,
                     batch_mode: bool = False,
                     prefer_compact: bool = False,
                     taxonomy: TagTaxonomy | None = None,
                     decision_context: DecisionContext | None = None,
                     learning_profile: LearningProfile | None = None,
                     learning_examples: LearningExamples | None = None,
                     run_db: LocalStateDB | None = None,
                     run_id: int | None = None) -> bool:
    """Einzelnes Dokument analysieren und organisieren."""
    t_total_start = time.perf_counter()

    console.print(f"\n[bold]{'=' * 60}[/bold]")
    log.info(f"[bold cyan]START[/bold cyan] Dokument #{doc_id} - Lade von API...")
    try:
        document = paperless.get_document(doc_id)
    except Exception as e:
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "error_load", {"title": "", "tags": []}, error=str(e))
        log.error(f"  Fehler beim Laden von Dokument #{doc_id}: {e}")
        return False

    doc_title = document.get('title', 'Unbekannt')
    content_len = len(document.get('content') or '')
    log.info(f"  Geladen: [white]{doc_title}[/white] ({content_len} Zeichen)")

    # OCR quality check
    ocr_quality, ocr_score = _assess_ocr_quality(document)
    if ocr_quality == "poor":
        log.warning(f"  [yellow]OCR-Qualitaet schlecht[/yellow] (Score: {ocr_score:.2f}) - Ergebnis koennte ungenau sein")
    elif ocr_quality == "medium":
        log.info(f"  OCR-Qualitaet: mittel (Score: {ocr_score:.2f})")

    # Language detection
    doc_lang = _detect_language(document.get("content") or "")
    if doc_lang == "en":
        log.info("  Sprache: Englisch erkannt")

    few_shot_examples = learning_examples.select(document, limit=LEARNING_EXAMPLE_LIMIT) if learning_examples else []
    learning_hints = (
        learning_examples.routing_hints_for_document(document, limit=LEARNING_PRIOR_MAX_HINTS)
        if learning_examples else []
    )

    # Rule-based fast path: skip LLM for well-known correspondent patterns
    rule_suggestion = _try_rule_based_suggestion(document, learning_hints, storage_paths)
    if rule_suggestion:
        log.info(f"  [green]Regelbasiert[/green] (LLM uebersprungen): {rule_suggestion.get('correspondent')}")
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "rule_based", document, suggestion=rule_suggestion)
        suggestion = rule_suggestion
        # Jump straight to guardrails/validation below
    else:
        log.info(f"  LLM-Analyse laeuft... (Modell: {analyzer.model})")

    initial_compact = bool(prefer_compact)
    t_start = time.perf_counter()
    if suggestion is None:
        try:
            suggestion = analyzer.analyze(
                document,
                tags,
                correspondents,
                doc_types,
                storage_paths,
                taxonomy=taxonomy,
                decision_context=decision_context,
                few_shot_examples=[] if initial_compact else few_shot_examples,
                learning_hints=learning_hints,
                compact_mode=initial_compact,
            )
        except json.JSONDecodeError as e:
            log.error(f"  JSON-Parse-Fehler: {e}")
            if run_db and run_id:
                run_db.record_document(run_id, doc_id, "error_json", document, error=str(e))
            suggestion = None
        except requests.exceptions.Timeout:
            if initial_compact:
                retry_timeout = max(LLM_COMPACT_TIMEOUT_RETRY, LLM_COMPACT_TIMEOUT + 10)
                log.warning(
                    f"  Timeout im Kompaktmodus ({LLM_COMPACT_TIMEOUT}s) - Retry mit erweitertem Kompakt-Timeout "
                    f"({retry_timeout}s)..."
                )
                try:
                    suggestion = analyzer.analyze(
                        document,
                        tags,
                        correspondents,
                        doc_types,
                        storage_paths,
                        taxonomy=taxonomy,
                        decision_context=decision_context,
                        few_shot_examples=[],
                        learning_hints=learning_hints,
                        compact_mode=True,
                        read_timeout_override=retry_timeout,
                    )
                except requests.exceptions.Timeout:
                    log.error(f"  Timeout im Kompaktmodus: LLM hat zu lange gebraucht (Server: {analyzer.url}).")
                    if run_db and run_id:
                        run_db.record_document(run_id, doc_id, "error_timeout", document, error="llm timeout compact")
                    suggestion = None
                except Exception as compact_retry_exc:
                    log.error(f"  Fehler im erweiterten Kompakt-Retry: {compact_retry_exc}")
                    if run_db and run_id:
                        run_db.record_document(run_id, doc_id, "error_generic", document, error=str(compact_retry_exc))
                    suggestion = None
            else:
                log.warning(
                    f"  Timeout beim ersten Versuch ({LLM_TIMEOUT}s) - Kompakt-Retry ohne Few-Shot ({LLM_COMPACT_TIMEOUT}s)..."
                )
                try:
                    suggestion = analyzer.analyze(
                        document,
                        tags,
                        correspondents,
                        doc_types,
                        storage_paths,
                        taxonomy=taxonomy,
                        decision_context=decision_context,
                        few_shot_examples=[],
                        learning_hints=learning_hints,
                        compact_mode=True,
                    )
                except requests.exceptions.Timeout:
                    if _is_fully_organized(document):
                        log.warning("  Timeout im Recheck - bestehende Zuordnung bleibt unveraendert.")
                        if run_db and run_id:
                            run_db.record_document(run_id, doc_id, "timeout_kept_existing", document, error="llm timeout recheck")
                        return False
                    log.error(f"  Timeout: LLM hat zu lange gebraucht (Server: {analyzer.url}).")
                    if run_db and run_id:
                        run_db.record_document(run_id, doc_id, "error_timeout", document, error="llm timeout")
                    suggestion = None
                except json.JSONDecodeError as e:
                    log.error(f"  JSON-Parse-Fehler nach Kompakt-Retry: {e}")
                    if run_db and run_id:
                        run_db.record_document(run_id, doc_id, "error_json", document, error=str(e))
                    suggestion = None
                except Exception as e:
                    log.error(f"  Fehler nach Kompakt-Retry: {e}")
                    if run_db and run_id:
                        run_db.record_document(run_id, doc_id, "error_generic", document, error=str(e))
                    suggestion = None
        except requests.exceptions.ConnectionError:
            log.warning("  Verbindungsfehler beim ersten Versuch - Kompakt-Retry ohne Few-Shot...")
            try:
                suggestion = analyzer.analyze(
                    document,
                    tags,
                    correspondents,
                    doc_types,
                    storage_paths,
                    taxonomy=taxonomy,
                    decision_context=decision_context,
                    few_shot_examples=[],
                    learning_hints=learning_hints,
                    compact_mode=True,
                )
            except requests.exceptions.RequestException:
                log.error(f"  LLM nicht erreichbar! Endpoint: {analyzer.url}")
                if run_db and run_id:
                    run_db.record_document(run_id, doc_id, "error_llm_connection", document, error="llm connection failed")
                suggestion = None
            except Exception as e:
                log.error(f"  Fehler nach Verbindungs-Retry: {e}")
                if run_db and run_id:
                    run_db.record_document(run_id, doc_id, "error_generic", document, error=str(e))
                suggestion = None
        except Exception as e:
            log.error(f"  Fehler: {e}")
            if run_db and run_id:
                run_db.record_document(run_id, doc_id, "error_generic", document, error=str(e))
            suggestion = None

    # Fallback-Kette: LLM fehlgeschlagen -> Websuche + LLM Retry -> Learning-Priors
    if suggestion is None and ENABLE_WEB_HINTS:
        web_context = _web_search_document_context(document)
        if web_context:
            log.info("  [cyan]Websuche-Fallback aktiv[/cyan] - erneuter LLM-Versuch mit Web-Kontext")
            try:
                suggestion = analyzer.analyze(
                    document, tags, correspondents, doc_types, storage_paths,
                    taxonomy=taxonomy,
                    decision_context=decision_context,
                    few_shot_examples=[],
                    learning_hints=learning_hints,
                    compact_mode=True,
                    web_hint_override=web_context,
                )
            except Exception:
                suggestion = None

    if suggestion is None and learning_hints:
        suggestion = _build_suggestion_from_priors(document, learning_hints, storage_paths)
        if suggestion:
            log.info("  [cyan]Learning-Prior Fallback aktiv[/cyan] (LLM nicht verfuegbar)")
            if run_db and run_id:
                run_db.record_document(run_id, doc_id, "prior_fallback", document, suggestion=suggestion)
    if suggestion is None:
        return False

    t_elapsed = time.perf_counter() - t_start
    log.info(f"  LLM-Antwort erhalten ({t_elapsed:.1f}s) -> Titel: [green]{suggestion.get('title', '?')}[/green]")

    # E5: Low-confidence + no correspondent -> web search verification
    confidence = (suggestion.get("confidence") or "high").lower()
    corr_now = _normalize_text(str(suggestion.get("correspondent", "")))
    if (
        ENABLE_WEB_HINTS
        and confidence == "low"
        and not corr_now
        and suggestion.get("confidence") != "prior_only"
    ):
        web_context = _web_search_document_context(document)
        if web_context:
            log.info("  [cyan]Websuche-Verifikation[/cyan] (low confidence, kein Korrespondent)")
            try:
                verified = analyzer.analyze(
                    document, tags, correspondents, doc_types, storage_paths,
                    taxonomy=taxonomy,
                    decision_context=decision_context,
                    few_shot_examples=[],
                    learning_hints=learning_hints,
                    compact_mode=True,
                    web_hint_override=web_context,
                )
                if verified and _normalize_text(str(verified.get("correspondent", ""))):
                    suggestion = verified
                    log.info(f"  [green]Websuche-Verifikation erfolgreich[/green]: Korr={verified.get('correspondent')}")
            except Exception:
                pass  # Keep original suggestion on error

    _sanitize_suggestion_spelling(suggestion)

    # Improve title
    orig_title = suggestion.get("title", "")
    improved_title = _improve_title(orig_title, document)
    if improved_title and improved_title != orig_title:
        suggestion["title"] = improved_title

    # Extract document date if not already set
    doc_date = _extract_document_date(document)
    if doc_date and not document.get("created"):
        suggestion["created_date"] = doc_date
        log.info(f"  [cyan]Datum extrahiert[/cyan]: {doc_date}")

    guardrail_fixes = _apply_vendor_guardrails(
        document,
        suggestion,
        correspondents,
        storage_paths,
        decision_context=decision_context,
    )
    for fix in guardrail_fixes:
        log.info(f"  [cyan]Guardrail[/cyan]: {fix}")

    vehicle_fixes = _apply_vehicle_guardrails(
        document,
        suggestion,
        storage_paths,
        decision_context=decision_context,
    )
    for fix in vehicle_fixes:
        log.info(f"  [cyan]Vehicle-Guardrail[/cyan]: {fix}")

    topic_fixes = _apply_topic_guardrails(
        document,
        suggestion,
        correspondents,
        storage_paths,
        decision_context=decision_context,
    )
    for fix in topic_fixes:
        log.info(f"  [cyan]Topic-Guardrail[/cyan]: {fix}")

    learning_fixes = _apply_learning_guardrails(
        suggestion,
        storage_paths,
        learning_hints,
    )
    for fix in learning_fixes:
        log.info(f"  [cyan]Learning-Guardrail[/cyan]: {fix}")

    selected_tags, dropped_tags = _select_controlled_tags(
        suggestion.get("tags", []),
        tags,
        taxonomy=taxonomy,
        run_db=run_db,
        run_id=run_id,
        doc_id=doc_id,
    )
    suggestion["tags"] = selected_tags

    for dropped_tag, reason in dropped_tags:
        log.info(f"  [yellow]Tag verworfen:[/yellow] {dropped_tag} ({reason})")

    hard_reasons = _collect_hard_review_reasons(
        document,
        suggestion,
        selected_tags,
        storage_paths,
        decision_context=decision_context,
    )
    if hard_reasons:
        reason_text = "; ".join(hard_reasons)
        log.warning(f"  [yellow]Review noetig[/yellow]: {reason_text}")
        if dry_run:
            if run_db and run_id:
                run_db.record_document(run_id, doc_id, "dry_run_review", document, suggestion=suggestion, error=reason_text)
            return False
        if run_db:
            run_db.enqueue_review(run_id, doc_id, reason_text, suggestion)
            run_db.record_document(run_id, doc_id, "queued_review", document, suggestion=suggestion, error=reason_text)
        _mark_document_for_review(paperless, document, tags, reason_text)
        return False

    asn = paperless.get_next_asn() if USE_ARCHIVE_SERIAL_NUMBER else None
    show_suggestion(document, suggestion, asn, tags, correspondents, doc_types, storage_paths)

    if dry_run:
        log.info(f"  [yellow]TESTMODUS[/yellow] - Keine Aenderungen fuer #{doc_id}")
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "dry_run", document, suggestion=suggestion)
        return False

    if not batch_mode and not Confirm.ask("Aenderungen anwenden?"):
        log.info(f"  Uebersprungen (Benutzer)")
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "skipped_user", document, suggestion=suggestion)
        return False

    log.info(f"  IDs aufloesen und Aenderungen anwenden...")
    try:
        with WRITE_LOCK:
            update_data = resolve_ids(
                paperless,
                suggestion,
                tags,
                correspondents,
                doc_types,
                storage_paths,
                taxonomy=taxonomy,
                run_db=run_db,
                run_id=run_id,
                doc_id=doc_id,
            )
            result, fallback_notes = _apply_update_with_fallbacks(paperless, doc_id, update_data)
            for note in fallback_notes:
                log.warning(f"  [yellow]API-Fallback[/yellow]: {note}")
        result_label = result.get("archived_file_name") or result.get("original_file_name") or result.get("title") or "?"
        t_total = time.perf_counter() - t_total_start
        log.info(f"[bold green]FERTIG[/bold green] Dokument #{doc_id} aktualisiert -> {result_label} ({t_total:.1f}s gesamt)")
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "updated", document, suggestion=suggestion)
        if learning_profile:
            try:
                learning_profile.learn_from_document(document, suggestion)
                learning_profile.save()
            except Exception as learn_exc:
                log.warning(f"  [yellow]Learning-Profil Update uebersprungen:[/yellow] {learn_exc}")
        if learning_examples:
            try:
                learning_examples.append(document, suggestion)
            except Exception as ex_exc:
                log.warning(f"  [yellow]Learning-Beispiele Update uebersprungen:[/yellow] {ex_exc}")
        return True
    except Exception as exc:
        reason_text = f"update-fehler: {exc}"
        log.error(f"  {reason_text}")
        if run_db:
            run_db.enqueue_review(run_id, doc_id, reason_text, suggestion)
            run_db.record_document(run_id, doc_id, "queued_review_update_error", document, suggestion=suggestion, error=reason_text)
        _mark_document_for_review(paperless, document, tags, reason_text)
        return False


def _is_fully_organized(doc: dict) -> bool:
    """Prueft ob ein Dokument vollstaendig sortiert ist."""
    basic = bool(
        doc.get("tags")
        and doc.get("correspondent")
        and doc.get("document_type")
        and doc.get("storage_path")
    )
    if USE_ARCHIVE_SERIAL_NUMBER:
        return bool(basic and doc.get("archive_serial_number"))
    return basic


def auto_organize_all(paperless: PaperlessClient, analyzer: LocalLLMAnalyzer,
                      tags: list, correspondents: list, doc_types: list,
                      storage_paths: list, dry_run: bool,
                      force_recheck_all: bool = False,
                      prefer_compact: bool = False,
                      taxonomy: TagTaxonomy | None = None,
                      decision_context: DecisionContext | None = None,
                      learning_profile: LearningProfile | None = None,
                      learning_examples: LearningExamples | None = None,
                      run_db: LocalStateDB | None = None,
                      run_id: int | None = None):
    """Scannt ALLE Dokumente, ueberspringt bereits sortierte, organisiert den Rest."""

    log.info("[bold]AUTO-SORTIERUNG[/bold] gestartet - scanne alle Dokumente...")

    documents = paperless.get_documents()
    total = len(documents)
    log.info(f"  {total} Dokumente geladen")

    # Duplikate rausfiltern
    duplikat_tag_id = next((t["id"] for t in tags if t["name"] == "Duplikat"), None)
    if duplikat_tag_id:
        before = len(documents)
        documents = [d for d in documents if duplikat_tag_id not in (d.get("tags") or [])]
        skipped = before - len(documents)
        if skipped:
            log.info(f"  {skipped} Duplikate uebersprungen")

    # Nur unsortierte oder optional alles neu pruefen
    if force_recheck_all:
        already_done = []
        todo = list(documents)
        log.info(f"  [yellow]{len(todo)} Dokumente werden komplett neu geprueft (RECHECK_ALL_DOCS_IN_AUTO aktiv)[/yellow]")
    else:
        already_done = [d for d in documents if _is_fully_organized(d)]
        todo = [d for d in documents if not _is_fully_organized(d)]
        log.info(f"  [green]{len(already_done)} bereits vollstaendig sortiert[/green] -> uebersprungen")
        log.info(f"  [yellow]{len(todo)} muessen noch sortiert werden[/yellow]")

    skipped_recent_total = 0
    if run_db and todo and SKIP_RECENT_LLM_ERRORS_THRESHOLD > 0 and SKIP_RECENT_LLM_ERRORS_MINUTES > 0:
        filtered_todo = []
        for d in todo:
            recent_errors = run_db.count_recent_document_statuses(
                d["id"],
                ["error_timeout", "error_llm_connection"],
                SKIP_RECENT_LLM_ERRORS_MINUTES,
            )
            if recent_errors >= SKIP_RECENT_LLM_ERRORS_THRESHOLD:
                skipped_recent_total += 1
                continue
            filtered_todo.append(d)
        if skipped_recent_total:
            log.warning(
                f"  {skipped_recent_total} Dokumente mit frischen LLM-Fehlern werden voruebergehend "
                f"({SKIP_RECENT_LLM_ERRORS_MINUTES} min) uebersprungen"
            )
        todo = filtered_todo

    if not todo:
        if skipped_recent_total:
            log.warning(
                f"[yellow]Nichts verarbeitet: {skipped_recent_total} Dokumente sind temporÃ¤r im LLM-Fehler-Backoff.[/yellow]"
            )
            return {"total": total, "todo": 0, "applied": 0, "errors": 0, "skipped_recent_errors": skipped_recent_total}
        log.info("[bold green]Alles sortiert! Nichts zu tun.[/bold green]")
        return {"total": total, "todo": 0, "applied": 0, "errors": 0}

    # Zeige was fehlt
    no_tags = sum(1 for d in todo if not d.get("tags"))
    no_corr = sum(1 for d in todo if not d.get("correspondent"))
    no_type = sum(1 for d in todo if not d.get("document_type"))
    no_path = sum(1 for d in todo if not d.get("storage_path"))
    if USE_ARCHIVE_SERIAL_NUMBER:
        no_asn = sum(1 for d in todo if not d.get("archive_serial_number"))
        log.info(f"  Davon: {no_tags} ohne Tags, {no_corr} ohne Korrespondent, "
                 f"{no_type} ohne Typ, {no_path} ohne Pfad, {no_asn} ohne ASN")
    else:
        log.info(f"  Davon: {no_tags} ohne Tags, {no_corr} ohne Korrespondent, "
                 f"{no_type} ohne Typ, {no_path} ohne Pfad")

    batch_start = time.perf_counter()
    applied = 0
    errors = 0

    if AGENT_WORKERS <= 1:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Sortiere...", total=len(todo))
            for i, doc in enumerate(todo, 1):
                progress.update(task, description=f"[{i}/{len(todo)}] #{doc['id']}")
                log.info(f"[bold]--- {i}/{len(todo)} --- Dokument #{doc['id']}[/bold]")
                try:
                    if process_document(doc["id"], paperless, analyzer, tags, correspondents,
                                        doc_types, storage_paths, dry_run, batch_mode=not dry_run,
                                        prefer_compact=prefer_compact,
                                        taxonomy=taxonomy,
                                        decision_context=decision_context,
                                        learning_profile=learning_profile,
                                        learning_examples=learning_examples,
                                        run_db=run_db, run_id=run_id):
                        applied += 1
                except Exception as e:
                    errors += 1
                    log.error(f"Fehler bei #{doc['id']}: {e}")
                progress.advance(task)
    else:
        log.info(f"[bold]PARALLEL[/bold] Starte {AGENT_WORKERS} Agent-Worker")
        aborted = False
        pool = ThreadPoolExecutor(max_workers=AGENT_WORKERS)
        try:
            futures = {
                pool.submit(
                    process_document,
                    doc["id"],
                    paperless,
                    analyzer,
                    tags,
                    correspondents,
                    doc_types,
                    storage_paths,
                    dry_run,
                    True,
                    prefer_compact,
                    taxonomy,
                    decision_context,
                    learning_profile,
                    learning_examples,
                    run_db,
                    run_id,
                ): doc["id"]
                for doc in todo
            }
            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    if future.result():
                        applied += 1
                except Exception as e:
                    errors += 1
                    log.error(f"Fehler bei #{doc_id}: {e}")
        except KeyboardInterrupt:
            aborted = True
            log.warning("Auto-Sortierung durch Benutzer abgebrochen")
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if not aborted:
                pool.shutdown(wait=True, cancel_futures=False)

    batch_elapsed = time.perf_counter() - batch_start
    avg_time = (batch_elapsed / len(todo)) if todo else 0
    log.info(f"[bold]AUTO-SORTIERUNG FERTIG[/bold] - {applied}/{len(todo)} aktualisiert, "
             f"{errors} Fehler, {batch_elapsed:.1f}s gesamt ({avg_time:.1f}s/Dokument)")
    if AUTO_CLEANUP_AFTER_ORGANIZE and not dry_run:
        log.info("[bold]AUTO-CLEANUP[/bold] nach Auto-Sortierung gestartet")
        cleanup_tags(paperless, dry_run=False)
        cleanup_correspondents(paperless, dry_run=False)
        cleanup_document_types(paperless, dry_run=False)
    return {"total": total, "todo": len(todo), "applied": applied, "errors": errors,
            "elapsed_sec": round(batch_elapsed, 1), "avg_sec_per_doc": round(avg_time, 1)}


def batch_process(paperless: PaperlessClient, analyzer: LocalLLMAnalyzer,
                  tags: list, correspondents: list, doc_types: list,
                  storage_paths: list, dry_run: bool, mode: str = "untagged",
                  limit: int = 0,
                  taxonomy: TagTaxonomy | None = None,
                  decision_context: DecisionContext | None = None,
                  learning_profile: LearningProfile | None = None,
                  learning_examples: LearningExamples | None = None,
                  run_db: LocalStateDB | None = None,
                  run_id: int | None = None):
    """Mehrere Dokumente mit Filter verarbeiten."""

    mode_labels = {"untagged": "Ohne Tags", "unorganized": "Unvollstaendig", "all": "Alle"}
    log.info(f"[bold]BATCH START[/bold] - Filter: {mode_labels.get(mode, mode)}, Limit: {limit or 'alle'}")

    log.info("Lade Dokumente...")
    documents = paperless.get_documents()
    log.info(f"  {len(documents)} Dokumente geladen")

    # Duplikat-Tag ID
    duplikat_tag_id = next((t["id"] for t in tags if t["name"] == "Duplikat"), None)
    if duplikat_tag_id:
        before = len(documents)
        documents = [d for d in documents if duplikat_tag_id not in (d.get("tags") or [])]
        skipped = before - len(documents)
        if skipped:
            log.info(f"  {skipped} Duplikate uebersprungen")

    # Filter
    if mode == "untagged":
        documents = [d for d in documents if not d.get("tags")]
        log.info(f"  {len(documents)} ohne Tags")
    elif mode == "unorganized":
        documents = [d for d in documents if not _is_fully_organized(d)]
        log.info(f"  {len(documents)} nicht vollstaendig organisiert")

    skipped_recent_total = 0
    if run_db and documents and SKIP_RECENT_LLM_ERRORS_THRESHOLD > 0 and SKIP_RECENT_LLM_ERRORS_MINUTES > 0:
        filtered_docs = []
        for d in documents:
            recent_errors = run_db.count_recent_document_statuses(
                d["id"],
                ["error_timeout", "error_llm_connection"],
                SKIP_RECENT_LLM_ERRORS_MINUTES,
            )
            if recent_errors >= SKIP_RECENT_LLM_ERRORS_THRESHOLD:
                skipped_recent_total += 1
                continue
            filtered_docs.append(d)
        if skipped_recent_total:
            log.warning(
                f"  {skipped_recent_total} Dokumente mit frischen LLM-Fehlern werden voruebergehend "
                f"({SKIP_RECENT_LLM_ERRORS_MINUTES} min) uebersprungen"
            )
        documents = filtered_docs

    # Limit (0 = alle)
    if limit > 0 and len(documents) > limit:
        documents = documents[:limit]

    log.info(f"  Verarbeite {len(documents)} Dokumente")

    if not documents:
        if skipped_recent_total:
            log.warning("[yellow]Keine verarbeitbaren Dokumente: alle Kandidaten sind im LLM-Fehler-Backoff.[/yellow]")
        else:
            log.info("[yellow]Keine Dokumente gefunden.[/yellow]")
        return {"total": 0, "applied": 0, "errors": 0}

    batch_start = time.perf_counter()
    applied = 0
    errors = 0

    if AGENT_WORKERS <= 1:
        for i, doc in enumerate(documents, 1):
            log.info(f"[bold]--- {i}/{len(documents)} --- Dokument #{doc['id']}[/bold]")
            try:
                if process_document(doc["id"], paperless, analyzer, tags, correspondents,
                                    doc_types, storage_paths, dry_run, batch_mode=not dry_run,
                                    taxonomy=taxonomy,
                                    decision_context=decision_context,
                                    learning_profile=learning_profile,
                                    learning_examples=learning_examples,
                                    run_db=run_db, run_id=run_id):
                    applied += 1
            except Exception as e:
                errors += 1
                log.error(f"Fehler bei #{doc['id']}: {e}")
    else:
        log.info(f"[bold]PARALLEL[/bold] Starte {AGENT_WORKERS} Agent-Worker")
        aborted = False
        pool = ThreadPoolExecutor(max_workers=AGENT_WORKERS)
        try:
            futures = {
                pool.submit(
                    process_document,
                    doc["id"],
                    paperless,
                    analyzer,
                    tags,
                    correspondents,
                    doc_types,
                    storage_paths,
                    dry_run,
                    True,
                    False,
                    taxonomy,
                    decision_context,
                    learning_profile,
                    learning_examples,
                    run_db,
                    run_id,
                ): doc["id"]
                for doc in documents
            }
            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    if future.result():
                        applied += 1
                except Exception as e:
                    errors += 1
                    log.error(f"Fehler bei #{doc_id}: {e}")
        except KeyboardInterrupt:
            aborted = True
            log.warning("Batch-Verarbeitung durch Benutzer abgebrochen")
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if not aborted:
                pool.shutdown(wait=True, cancel_futures=False)

    batch_elapsed = time.perf_counter() - batch_start
    log.info(f"[bold]BATCH FERTIG[/bold] - {applied}/{len(documents)} aktualisiert, "
             f"{errors} Fehler, {batch_elapsed:.1f}s gesamt")
    if AUTO_CLEANUP_AFTER_ORGANIZE and not dry_run:
        log.info("[bold]AUTO-CLEANUP[/bold] nach Batch gestartet")
        cleanup_tags(paperless, dry_run=False)
        cleanup_correspondents(paperless, dry_run=False)
        cleanup_document_types(paperless, dry_run=False)
    return {"total": len(documents), "applied": applied, "errors": errors}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cleanup-Funktionen
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _is_ascii_only(name: str) -> bool:
    """Prueft ob Name nur ASCII-Zeichen enthaelt (englisch)."""
    try:
        name.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def cleanup_tags(paperless: PaperlessClient, dry_run: bool = True):
    """Tags aufraeumen: Taxonomie + Whitelist + Schwellwerte."""
    console.print(Panel("[bold]Tag-Aufraeumung[/bold]", border_style="yellow"))
    log.info("[bold]CLEANUP TAGS[/bold] gestartet")

    tags = paperless.get_tags()
    log.info(f"  {len(tags)} Tags geladen")

    taxonomy = TagTaxonomy(TAXONOMY_FILE)
    taxonomy_set = {_normalize_tag_name(t) for t in taxonomy.canonical_tags}
    protected_tags = {_normalize_tag_name(t) for t in TAG_WHITELIST}
    protected_tags.add(_normalize_tag_name(REVIEW_TAG_NAME))
    protected_tags.add(_normalize_tag_name("Duplikat"))

    to_delete = []
    for t in tags:
        name = t["name"]
        doc_count = t.get("document_count", 0)
        normalized_name = _normalize_tag_name(name)
        in_taxonomy = normalized_name in taxonomy_set if taxonomy_set else True

        # Whitelist / geschuetzte Tags - nie loeschen
        if normalized_name in protected_tags:
            continue
        if KEEP_UNUSED_TAXONOMY_TAGS and in_taxonomy and doc_count == 0:
            # Verhindert Bootstrap/Delete-Pingpong bei aktivierter Taxonomie-Autocreate.
            continue

        reason = ""
        # Standard: nur ungenutzte Tags entfernen.
        if doc_count == 0:
            reason = "0 Dokumente" if in_taxonomy else "nicht in Taxonomie, 0 Dokumente"
        # Optional aggressiver Modus per env.
        elif DELETE_USED_TAGS:
            if not in_taxonomy and doc_count <= NON_TAXONOMY_DELETE_THRESHOLD:
                reason = f"nicht in Taxonomie, <= {NON_TAXONOMY_DELETE_THRESHOLD} Dokumente"
            elif TAG_DELETE_THRESHOLD > 0 and doc_count <= TAG_DELETE_THRESHOLD:
                reason = f"<= {TAG_DELETE_THRESHOLD} Dokumente"
            elif (
                TAG_ENGLISH_THRESHOLD > 0
                and not in_taxonomy
                and doc_count <= TAG_ENGLISH_THRESHOLD
                and _is_ascii_only(name)
            ):
                reason = f"ASCII/englisch, <= {TAG_ENGLISH_THRESHOLD} Dokumente"

        if reason:
            to_delete.append((t, reason))

    console.print(f"[yellow]Zum Loeschen markiert: {len(to_delete)}[/yellow]")

    if to_delete:
        # Anzeigen was geloescht wird
        table = Table(title="Tags zum Loeschen", show_header=True)
        table.add_column("Name", style="red")
        table.add_column("Dokumente", justify="right")
        table.add_column("Grund")

        for item, reason in sorted(to_delete, key=lambda x: x[0]["name"].lower()):
            doc_count = item.get("document_count", 0)
            table.add_row(item["name"], str(doc_count), reason)
        console.print(table)

        if not dry_run:
            deleted = paperless.batch_delete("tags", [item for item, _ in to_delete], "Tags")
            log.info(f"  [green]{deleted} Tags geloescht[/green]")

    remaining = paperless.get_tags()
    log.info(f"  CLEANUP TAGS fertig - {len(remaining)} Tags verbleibend")
    if dry_run:
        console.print("[yellow]TESTMODUS: Nichts geloescht[/yellow]")


def _strip_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _correspondent_core_name(value: str) -> str:
    raw = _strip_diacritics(_normalize_correspondent_name(value).lower())
    if not raw:
        return ""
    # E-Mail/URL nur exakt zusammenfassen, nicht gegen Firmennamen.
    if "@" in raw or raw.startswith("http://") or raw.startswith("https://"):
        return raw.strip()
    tokens = re.findall(r"[a-z0-9]+", raw)
    cleaned = []
    for token in tokens:
        if token in CORRESPONDENT_LEGAL_TOKENS:
            continue
        if token in CORRESPONDENT_STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        cleaned.append(token)
    return " ".join(cleaned).strip()


def _is_correspondent_duplicate_name(name_a: str, name_b: str) -> bool:
    core_a = _correspondent_core_name(name_a)
    core_b = _correspondent_core_name(name_b)
    if not core_a or not core_b or core_a == core_b:
        return bool(core_a and core_b and core_a == core_b)

    tokens_a = set(core_a.split())
    tokens_b = set(core_b.split())
    overlap = len(tokens_a & tokens_b)

    # Keine Teilstring-Merges: nur fast identische Namen zusammenlegen.
    ratio = difflib.SequenceMatcher(None, core_a, core_b).ratio()
    return ratio >= CORRESPONDENT_MERGE_MIN_NAME_SIMILARITY and overlap >= 2


def _find_correspondent_groups(correspondents: list) -> dict:
    """Findet echte Namensduplikate (generisch, konservativ)."""
    ordered = sorted(correspondents, key=lambda c: int(c.get("document_count", 0) or 0), reverse=True)
    by_id = {int(c["id"]): c for c in ordered if c.get("id") is not None}
    unassigned = set(by_id.keys())
    groups: dict[str, list] = {}
    group_no = 1

    for seed in ordered:
        seed_id = int(seed["id"])
        if seed_id not in unassigned:
            continue

        cluster_ids = {seed_id}
        queue = [seed_id]
        unassigned.remove(seed_id)

        while queue:
            current_id = queue.pop(0)
            current = by_id[current_id]
            for cand_id in list(unassigned):
                cand = by_id[cand_id]
                if _is_correspondent_duplicate_name(current.get("name", ""), cand.get("name", "")):
                    cluster_ids.add(cand_id)
                    queue.append(cand_id)
                    unassigned.remove(cand_id)

        cluster = [by_id[cid] for cid in cluster_ids]
        if len(cluster) > 1:
            cluster.sort(key=_correspondent_keep_sort_key, reverse=True)
            label = cluster[0].get("name") or f"Duplikatgruppe {group_no}"
            groups[label] = cluster
            group_no += 1

    return groups


def _correspondent_keep_sort_key(item: dict) -> tuple:
    """Waehlt den stabilsten Kanonischen Namen innerhalb einer Duplikatgruppe."""
    name = (item.get("name") or "").strip()
    name_lower = name.lower()
    core = _correspondent_core_name(name)
    doc_count = int(item.get("document_count", 0) or 0)

    quality = 0
    if " via " not in name_lower:
        quality += 2
    if "@" not in name_lower:
        quality += 2
    if "http://" not in name_lower and "https://" not in name_lower:
        quality += 2
    if len(core.split()) >= 2:
        quality += 1
    if len(name) <= 70:
        quality += 1

    return (doc_count, quality, -len(name))


def cleanup_correspondents(paperless: PaperlessClient, dry_run: bool = True):
    """Konservatives Korrespondenten-Cleanup: optional ungenutzte loeschen + echte Duplikate mergen."""
    console.print(Panel("[bold]Korrespondenten-Aufraeumung[/bold]", border_style="cyan"))
    log.info("[bold]CLEANUP KORRESPONDENTEN[/bold] gestartet")

    correspondents = paperless.get_correspondents()
    log.info(f"  {len(correspondents)} Korrespondenten geladen")

    # Leere loeschen
    unused = [c for c in correspondents if c.get("document_count", 0) == 0]
    console.print(f"[yellow]Ungenutzt (0 Dokumente): {len(unused)}[/yellow]")

    if unused and not dry_run:
        if DELETE_UNUSED_CORRESPONDENTS:
            deleted = paperless.batch_delete("correspondents", unused, "Korrespondenten")
            console.print(f"[green]{deleted} ungenutzte Korrespondenten geloescht[/green]")
        else:
            console.print("[cyan]Info:[/cyan] Ungenutzte Korrespondenten bleiben erhalten (Policy).")

    # Duplikat-Gruppen finden
    groups = _find_correspondent_groups(correspondents)
    duplicates = {}
    for name, items in groups.items():
        total_docs = sum(int(c.get("document_count", 0) or 0) for c in items)
        if len(items) > 1 and total_docs >= CORRESPONDENT_MERGE_MIN_GROUP_DOCS:
            duplicates[name] = items

    if duplicates:
        console.print(f"\n[bold]{len(duplicates)} Duplikat-Gruppen gefunden:[/bold]")
        for group_name, items in sorted(duplicates.items()):
            total = sum(c.get("document_count", 0) for c in items)
            items.sort(key=_correspondent_keep_sort_key, reverse=True)
            keep = items[0]
            merge = items[1:]

            table = Table(title=f"{group_name} ({total} Dokumente)", show_header=True)
            table.add_column("Aktion", width=10)
            table.add_column("ID", width=6)
            table.add_column("Name")
            table.add_column("Dokumente", width=10)

            table.add_row("[green]BEHALTEN[/green]", str(keep["id"]), keep["name"], str(keep["document_count"]))
            for m in merge:
                table.add_row("[red]MERGEN[/red]", str(m["id"]), m["name"], str(m["document_count"]))
            console.print(table)

            if not dry_run:
                all_docs = paperless.get_documents()
                for m in merge:
                    docs_to_move = [d for d in all_docs if d.get("correspondent") == m["id"]]
                    for doc in docs_to_move:
                        paperless.update_document(doc["id"], {"correspondent": keep["id"]})
                        console.print(f"    Dokument #{doc['id']}: {m['name']} -> {keep['name']}")
                    paperless.delete_correspondent(m["id"])
                    console.print(f"    [green]Geloescht:[/green] {m['name']} (gemergt in {keep['name']})")

    remaining = paperless.get_correspondents()
    log.info(f"  CLEANUP KORRESPONDENTEN fertig - {len(remaining)} verbleibend")
    if dry_run:
        console.print("[yellow]TESTMODUS: Nichts geaendert[/yellow]")


def cleanup_document_types(paperless: PaperlessClient, dry_run: bool = True):
    """Leere Dokumenttypen loeschen."""
    console.print(Panel("[bold]Dokumenttypen-Aufraeumung[/bold]", border_style="magenta"))
    log.info("[bold]CLEANUP DOKUMENTTYPEN[/bold] gestartet")

    types = paperless.get_document_types()
    log.info(f"  {len(types)} Dokumenttypen geladen")

    # 1) Canonical-Namen aus ALLOWED_DOC_TYPES erzwingen (z. B. "kontoauszug" -> "Kontoauszug")
    allowed_lower = {name.lower(): name for name in ALLOWED_DOC_TYPES}
    by_exact_name = {t["name"]: t for t in types}
    to_normalize: list[tuple[dict, dict]] = []

    for t in types:
        current_name = (t.get("name") or "").strip()
        key = current_name.lower()
        canonical_name = allowed_lower.get(key)
        if not canonical_name or current_name == canonical_name:
            continue

        target = by_exact_name.get(canonical_name)
        if target is None and not dry_run:
            try:
                target = paperless.create_document_type(canonical_name)
                types.append(target)
                by_exact_name[canonical_name] = target
                console.print(f"  [green]+ Canonical-Dokumenttyp erstellt:[/green] {canonical_name}")
            except requests.exceptions.HTTPError:
                fresh = paperless.get_document_types()
                types = fresh
                by_exact_name = {x["name"]: x for x in types}
                target = by_exact_name.get(canonical_name)

        if target and target["id"] != t["id"]:
            to_normalize.append((t, target))

    if to_normalize:
        table = Table(title="Dokumenttypen-Normalisierung", show_header=True)
        table.add_column("Von", style="yellow")
        table.add_column("Nach", style="green")
        table.add_column("Dokumente", justify="right")
        for src, dst in to_normalize:
            table.add_row(src["name"], dst["name"], str(src.get("document_count", 0)))
        console.print(table)

        if not dry_run:
            all_docs = paperless.get_documents()
            for src, dst in to_normalize:
                docs_to_move = [d for d in all_docs if d.get("document_type") == src["id"]]
                for doc in docs_to_move:
                    paperless.update_document(doc["id"], {"document_type": dst["id"]})
                    console.print(f"    Dokument #{doc['id']}: {src['name']} -> {dst['name']}")
                paperless.delete_document_type(src["id"])
                console.print(f"    [green]Geloescht:[/green] {src['name']} (normalisiert auf {dst['name']})")

            types = paperless.get_document_types()
            log.info(f"  [green]{len(to_normalize)} Dokumenttypen normalisiert[/green]")

    unused = [t for t in types if t.get("document_count", 0) == 0]
    console.print(f"[yellow]Ungenutzt (0 Dokumente): {len(unused)}[/yellow]")

    if unused:
        table = Table(title="Dokumenttypen zum Loeschen", show_header=True)
        table.add_column("ID", width=6)
        table.add_column("Name")
        for t in unused:
            table.add_row(str(t["id"]), t["name"])
        console.print(table)

        if not dry_run:
            deleted = paperless.batch_delete("document_types", unused, "Dokumenttypen")
            console.print(f"[green]{deleted} Dokumenttypen geloescht[/green]")

    remaining = paperless.get_document_types()
    console.print(f"\n[bold]Verbleibend: {len(remaining)} Dokumenttypen[/bold]")

    # Verbleibende anzeigen
    if remaining:
        table = Table(title="Verbleibende Dokumenttypen", show_header=True)
        table.add_column("ID", width=6)
        table.add_column("Name")
        table.add_column("Dokumente", justify="right")
        for t in sorted(remaining, key=lambda x: x.get("document_count", 0), reverse=True):
            table.add_row(str(t["id"]), t["name"], str(t.get("document_count", 0)))
        console.print(table)

    log.info(f"  CLEANUP DOKUMENTTYPEN fertig - {len(remaining)} verbleibend")
    if dry_run:
        console.print("[yellow]TESTMODUS: Nichts geloescht[/yellow]")


def cleanup_all(paperless: PaperlessClient, dry_run: bool = True):
    """Alle drei Cleanup-Funktionen nacheinander."""
    cleanup_tags(paperless, dry_run)
    console.print()
    cleanup_correspondents(paperless, dry_run)
    console.print()
    cleanup_document_types(paperless, dry_run)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Duplikate finden
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def find_duplicates(paperless: PaperlessClient):
    """Erkennung nach Dateiname und Titel - Ausgabe als Rich-Table."""
    console.print(Panel(
        "[bold]Duplikat-Erkennung[/bold]\n[red]Nur Meldung - es wird NICHTS geloescht![/red]",
        border_style="red"
    ))

    log.info("[bold]DUPLIKAT-SCAN[/bold] gestartet")
    log.info("Lade alle Dokumente...")
    docs = paperless.get_documents()
    log.info(f"  {len(docs)} Dokumente geladen")

    # Nach Dateiname gruppieren (exakt)
    by_filename = defaultdict(list)
    for d in docs:
        fname = d.get("original_file_name", "").strip()
        if fname:
            by_filename[fname].append(d)
    filename_dupes = {k: v for k, v in by_filename.items() if len(v) > 1}

    # Nach normalisiertem Dateinamen gruppieren (Datum, UUID, Scan-Prefix entfernen)
    by_norm_filename = defaultdict(list)
    for d in docs:
        fname = d.get("original_file_name", "").strip()
        if fname:
            norm = fname.lower()
            norm = re.sub(r"\.[a-z]{2,4}$", "", norm)  # Extension
            norm = re.sub(r"\d{4}[-_]\d{2}[-_]\d{2}", "", norm)  # Dates
            norm = re.sub(r"[0-9a-f]{8}[-_]?[0-9a-f]{4}[-_]?[0-9a-f]{4}[-_]?[0-9a-f]{4}[-_]?[0-9a-f]{12}", "", norm)  # UUIDs
            norm = re.sub(r"scan[-_]?\d+", "", norm, flags=re.IGNORECASE)  # Scan prefixes
            norm = re.sub(r"[-_\s]+", " ", norm).strip()
            if norm and len(norm) > 3:
                by_norm_filename[norm].append(d)
    norm_filename_dupes = {k: v for k, v in by_norm_filename.items() if len(v) > 1}
    # Remove groups that are already caught by exact filename match
    exact_ids = set()
    for items in filename_dupes.values():
        exact_ids.update(d["id"] for d in items)
    norm_filename_dupes = {
        k: v for k, v in norm_filename_dupes.items()
        if not all(d["id"] in exact_ids for d in v)
    }

    # Nach Titel gruppieren (case-insensitive)
    by_title = defaultdict(list)
    for d in docs:
        title = d.get("title", "").strip().lower()
        if title and title != "unbekannt":
            by_title[title].append(d)
    title_dupes = {k: v for k, v in by_title.items() if len(v) > 1}

    # Dateiname-Duplikate anzeigen
    if filename_dupes:
        console.print(f"\n[bold]Gleicher Dateiname: {len(filename_dupes)} Gruppen[/bold]")
        for fname, items in sorted(filename_dupes.items()):
            table = Table(title=f"Datei: {fname}", show_header=True)
            table.add_column("ID", width=6)
            table.add_column("Titel")
            table.add_column("Erstellt", width=12)
            table.add_column("Inhalt (Vorschau)", width=40)
            for d in sorted(items, key=lambda x: x.get("id", 0)):
                content_preview = (d.get("content") or "")[:80].replace("\n", " ")
                table.add_row(
                    str(d["id"]),
                    d.get("title", "?"),
                    str(d.get("created", "?"))[:10],
                    content_preview,
                )
            console.print(table)
    else:
        console.print("\n[green]Keine Dateiname-Duplikate gefunden.[/green]")

    # Normalisierte Dateiname-Duplikate anzeigen
    if norm_filename_dupes:
        console.print(f"\n[bold]Aehnlicher Dateiname (normalisiert): {len(norm_filename_dupes)} Gruppen[/bold]")
        shown = 0
        for norm_name, items in sorted(norm_filename_dupes.items(), key=lambda x: -len(x[1])):
            if shown >= 10:
                console.print(f"  ... und {len(norm_filename_dupes) - 10} weitere Gruppen")
                break
            table = Table(title=f"Muster: {norm_name}", show_header=True)
            table.add_column("ID", width=6)
            table.add_column("Dateiname")
            table.add_column("Titel")
            for d in sorted(items, key=lambda x: x.get("id", 0)):
                table.add_row(str(d["id"]), d.get("original_file_name", "?"), d.get("title", "?"))
            console.print(table)
            shown += 1

    # Titel-Duplikate anzeigen
    if title_dupes:
        console.print(f"\n[bold]Gleicher Titel: {len(title_dupes)} Gruppen[/bold]")
        # Nur Top 20 anzeigen
        shown = 0
        for title_key, items in sorted(title_dupes.items(), key=lambda x: -len(x[1])):
            if shown >= 20:
                console.print(f"  ... und {len(title_dupes) - 20} weitere Gruppen")
                break
            if len(items) <= 5:
                display_title = items[0].get("title", title_key)
                table = Table(title=f"Titel: {display_title}", show_header=True)
                table.add_column("ID", width=6)
                table.add_column("Datei")
                table.add_column("Erstellt", width=12)
                for d in sorted(items, key=lambda x: x.get("id", 0)):
                    table.add_row(
                        str(d["id"]),
                        d.get("original_file_name", "?"),
                        str(d.get("created", "?"))[:10],
                    )
                console.print(table)
                shown += 1
    else:
        console.print("\n[green]Keine Titel-Duplikate gefunden.[/green]")

    # Content-Fingerprint: near-duplicate detection
    console.print("\n[bold]Inhalts-Aehnlichkeit (Fingerprint-Analyse)...[/bold]")
    content_dupes = []
    fingerprints = []
    for d in docs:
        fp = _content_fingerprint(str(d.get("content") or "")[:3000])
        if fp:
            fingerprints.append((d, fp))
    # Compare all pairs (for small collections; skip if too many docs)
    if len(fingerprints) <= 2000:
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                sim = _content_similarity(fingerprints[i][1], fingerprints[j][1])
                if sim >= 0.75:
                    doc_a, doc_b = fingerprints[i][0], fingerprints[j][0]
                    content_dupes.append((sim, doc_a, doc_b))
        content_dupes.sort(key=lambda x: x[0], reverse=True)
        if content_dupes:
            console.print(f"\n[bold]Inhalts-Aehnlichkeit: {len(content_dupes)} Paare (>=75%)[/bold]")
            for sim, doc_a, doc_b in content_dupes[:20]:
                console.print(
                    f"  [yellow]{sim:.0%}[/yellow] aehnlich: "
                    f"#{doc_a['id']} '{doc_a.get('title', '?')[:40]}' <-> "
                    f"#{doc_b['id']} '{doc_b.get('title', '?')[:40]}'"
                )
            if len(content_dupes) > 20:
                console.print(f"  ... und {len(content_dupes) - 20} weitere Paare")
        else:
            console.print("[green]Keine inhaltsaehnlichen Dokumente gefunden.[/green]")
    else:
        console.print(f"  [yellow]Zu viele Dokumente ({len(fingerprints)}) fuer Fingerprint-Vergleich - uebersprungen[/yellow]")
        content_dupes = []

    # Zusammenfassung
    console.print(f"\n[bold]Zusammenfassung:[/bold]")
    console.print(f"  Dateiname-Duplikate: {len(filename_dupes)} Gruppen ({sum(len(v) for v in filename_dupes.values())} Dokumente)")
    console.print(f"  Titel-Duplikate: {len(title_dupes)} Gruppen ({sum(len(v) for v in title_dupes.values())} Dokumente)")
    console.print(f"  Inhalts-Aehnlichkeit: {len(content_dupes)} Paare")
    console.print("[red]Es wurden KEINE Dokumente geloescht![/red]")
    log.info(f"  DUPLIKAT-SCAN fertig - {len(filename_dupes)} Dateiname-Gruppen, {len(title_dupes)} Titel-Gruppen, {len(content_dupes)} Inhalts-Paare")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Statistiken
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def show_statistics(paperless: PaperlessClient):
    """Uebersicht ueber Paperless-NGX Instanz."""
    console.print(Panel("[bold]Statistiken / Uebersicht[/bold]", border_style="blue"))
    log.info("[bold]STATISTIKEN[/bold] laden...")

    with console.status("Lade Daten..."):
        tags = paperless.get_tags()
        correspondents = paperless.get_correspondents()
        doc_types = paperless.get_document_types()
        storage_paths = paperless.get_storage_paths()
        documents = paperless.get_documents()

    # Uebersicht
    table = Table(title="Paperless-NGX Uebersicht", show_header=True, width=50)
    table.add_column("Kategorie", style="cyan")
    table.add_column("Anzahl", justify="right", style="bold")

    table.add_row("Dokumente", str(len(documents)))
    table.add_row("Tags", str(len(tags)))
    table.add_row("Korrespondenten", str(len(correspondents)))
    table.add_row("Dokumenttypen", str(len(doc_types)))
    table.add_row("Speicherpfade", str(len(storage_paths)))
    console.print(table)

    # Unorganisierte zaehlen
    no_tags = sum(1 for d in documents if not d.get("tags"))
    no_type = sum(1 for d in documents if not d.get("document_type"))
    no_path = sum(1 for d in documents if not d.get("storage_path"))
    no_corr = sum(1 for d in documents if not d.get("correspondent"))
    incomplete = sum(1 for d in documents
                     if not (d.get("document_type") and d.get("storage_path") and d.get("tags")))

    table2 = Table(title="Unorganisierte Dokumente", show_header=True, width=50)
    table2.add_column("Status", style="yellow")
    table2.add_column("Anzahl", justify="right", style="bold")

    fully_organized = sum(1 for d in documents if _is_fully_organized(d))
    orphans = sum(1 for d in documents
                  if not d.get("tags") and not d.get("correspondent")
                  and not d.get("document_type") and not d.get("storage_path"))
    completeness = (fully_organized / len(documents) * 100) if documents else 0

    table2.add_row("Ohne Tags", str(no_tags))
    table2.add_row("Ohne Korrespondent", str(no_corr))
    table2.add_row("Ohne Dokumenttyp", str(no_type))
    table2.add_row("Ohne Speicherpfad", str(no_path))
    table2.add_row("Nicht vollstaendig", str(incomplete))
    table2.add_row("Vollstaendig sortiert", f"[green]{fully_organized}[/green]")
    table2.add_row("Waisen (komplett leer)", f"[red]{orphans}[/red]" if orphans else "0")
    table2.add_row("Organisationsgrad", f"{completeness:.1f}%")
    console.print(table2)

    # Top Tags
    if tags:
        top_tags = sorted(tags, key=lambda x: x.get("document_count", 0), reverse=True)[:10]
        table3 = Table(title="Top 10 Tags", show_header=True)
        table3.add_column("Tag", style="green")
        table3.add_column("Dokumente", justify="right")
        for t in top_tags:
            table3.add_row(t["name"], str(t.get("document_count", 0)))
        console.print(table3)

    # Top Korrespondenten
    if correspondents:
        top_corrs = sorted(correspondents, key=lambda x: x.get("document_count", 0), reverse=True)[:10]
        table4 = Table(title="Top 10 Korrespondenten", show_header=True)
        table4.add_column("Korrespondent", style="green")
        table4.add_column("Dokumente", justify="right")
        for c in top_corrs:
            table4.add_row(c["name"], str(c.get("document_count", 0)))
        console.print(table4)

    # Speicherpfad-Verteilung
    if storage_paths:
        path_counts = sorted(storage_paths, key=lambda x: x.get("document_count", 0), reverse=True)
        used_paths = [p for p in path_counts if p.get("document_count", 0) > 0]
        empty_paths = [p for p in path_counts if p.get("document_count", 0) == 0]
        table_paths = Table(title="Speicherpfad-Verteilung", show_header=True)
        table_paths.add_column("Pfad", style="green")
        table_paths.add_column("Dokumente", justify="right")
        table_paths.add_column("Anteil", justify="right")
        total_path_docs = sum(p.get("document_count", 0) for p in path_counts)
        for p in used_paths[:15]:
            count = p.get("document_count", 0)
            pct = (count / total_path_docs * 100) if total_path_docs else 0
            table_paths.add_row(p["name"], str(count), f"{pct:.1f}%")
        if empty_paths:
            table_paths.add_row(f"[dim]({len(empty_paths)} leere Pfade)[/dim]", "", "")
        console.print(table_paths)

    # Learning-System Statistiken
    try:
        learning_examples = LearningExamples(LEARNING_EXAMPLES_FILE)
        profiles = learning_examples._build_correspondent_profiles()
        total_examples = len(learning_examples._examples)
        total_profiles = len(profiles)
        rule_based_ready = sum(1 for p in profiles if p.get("count", 0) >= RULE_BASED_MIN_SAMPLES
                               and p.get("document_type_ratio", 0) >= RULE_BASED_MIN_RATIO
                               and p.get("storage_path_ratio", 0) >= RULE_BASED_MIN_RATIO)
        prior_ready = sum(1 for p in profiles if p.get("count", 0) >= LEARNING_PRIOR_MIN_SAMPLES)
        low_sample = sum(1 for p in profiles if p.get("count", 0) < LEARNING_PRIOR_MIN_SAMPLES)

        table5 = Table(title="Learning-System", show_header=True, width=60)
        table5.add_column("Metrik", style="cyan")
        table5.add_column("Wert", justify="right", style="bold")
        table5.add_row("Gespeicherte Beispiele", str(total_examples))
        table5.add_row("Korrespondenten-Profile", str(total_profiles))
        table5.add_row(f"Regelbasiert bereit (>={RULE_BASED_MIN_SAMPLES} Samples, >={RULE_BASED_MIN_RATIO:.0%} konsistent)", str(rule_based_ready))
        table5.add_row(f"Prior-bereit (>={LEARNING_PRIOR_MIN_SAMPLES} Samples)", str(prior_ready))
        table5.add_row(f"Unzureichend (<{LEARNING_PRIOR_MIN_SAMPLES} Samples)", str(low_sample))
        table5.add_row("LLM-Modell", LLM_MODEL or "(Server-Default)")
        table5.add_row("Web-Hinweise", "aktiv" if ENABLE_WEB_HINTS else "inaktiv")
        console.print(table5)
    except Exception:
        pass  # Learning file might not exist yet


MONTH_NAMES_DE = [
    "", "Januar", "Februar", "Maerz", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]


def show_monthly_report(run_db: LocalStateDB, paperless: PaperlessClient):
    """Monatlichen Report aus SQLite-Daten anzeigen."""
    now = datetime.now()
    year_str = Prompt.ask("Jahr", default=str(now.year))
    month_str = Prompt.ask("Monat (1-12)", default=str(now.month))
    try:
        year = int(year_str)
        month = int(month_str)
        if not 1 <= month <= 12:
            raise ValueError
    except ValueError:
        console.print("[red]Ungueltige Eingabe.[/red]")
        return

    with console.status("Generiere Report..."):
        report = run_db.generate_monthly_report(year, month)

    month_label = MONTH_NAMES_DE[month] if 1 <= month <= 12 else str(month)
    console.print(Panel(
        f"[bold]Monatlicher Report: {month_label} {year}[/bold]",
        border_style="blue",
    ))

    # Zusammenfassung
    t1 = Table(title="Zusammenfassung", show_header=True, width=50)
    t1.add_column("Metrik", style="cyan")
    t1.add_column("Wert", justify="right", style="bold")
    t1.add_row("Laeufe", str(report["total_runs"]))
    t1.add_row("Dokumente verarbeitet", str(report["total_docs"]))
    t1.add_row("Davon erfolgreich", str(report["success_docs"]))
    t1.add_row("Erfolgsquote", f"{report['success_rate']:.1f} %")
    for status, cnt in sorted(report["status_counts"].items()):
        t1.add_row(f"  Status: {status}", str(cnt))
    console.print(t1)

    # Neue Korrespondenten
    if report["new_correspondents"]:
        t2 = Table(title="Neue Korrespondenten", show_header=True)
        t2.add_column("Name", style="green")
        for name in report["new_correspondents"]:
            t2.add_row(name)
        console.print(t2)
    else:
        console.print("[dim]Keine neuen Korrespondenten in diesem Zeitraum.[/dim]")

    # Geloeschte Tags
    if report["deleted_tags"]:
        t3 = Table(title="Geloeschte Tags", show_header=True)
        t3.add_column("Tag", style="red")
        t3.add_column("Anzahl", justify="right")
        for tag in report["deleted_tags"]:
            t3.add_row(tag["name"], str(tag["count"]))
        console.print(t3)
    else:
        console.print("[dim]Keine Tags geloescht in diesem Zeitraum.[/dim]")

    # Offene Reviews
    if report["open_reviews"]:
        t4 = Table(title="Offene Reviews", show_header=True)
        t4.add_column("ID", width=6)
        t4.add_column("Dokument", width=10)
        t4.add_column("Grund")
        t4.add_column("Aktualisiert", width=20)
        for item in report["open_reviews"]:
            t4.add_row(str(item["id"]), str(item["doc_id"]), item["reason"], item["updated_at"])
        console.print(t4)
    else:
        console.print("[dim]Keine offenen Reviews.[/dim]")

    # Haeufigste Fehler
    if report["errors"]:
        t5 = Table(title="Haeufigste Fehler", show_header=True)
        t5.add_column("Fehler", style="red", max_width=60)
        t5.add_column("Anzahl", justify="right")
        for err in report["errors"]:
            t5.add_row(err["error"][:80], str(err["count"]))
        console.print(t5)
    else:
        console.print("[dim]Keine Fehler in diesem Zeitraum.[/dim]")




def _remove_review_tag_from_document(paperless: PaperlessClient, doc: dict):
    """Entfernt den Manuell-Pruefen-Tag von einem Dokument, falls vorhanden."""
    try:
        all_tags = paperless.get_tags()
    except Exception:
        return
    review_tag_id = None
    for tag in all_tags:
        if _normalize_tag_name(tag.get("name", "")) == _normalize_tag_name(REVIEW_TAG_NAME):
            review_tag_id = tag["id"]
            break
    if review_tag_id is None:
        return
    current_tags = list(doc.get("tags") or [])
    if review_tag_id not in current_tags:
        return
    current_tags.remove(review_tag_id)
    try:
        paperless.update_document(doc["id"], {"tags": current_tags})
        log.info(f"  Review-Tag '{REVIEW_TAG_NAME}' entfernt von Dokument #{doc['id']}")
    except Exception as exc:
        log.warning(f"  Review-Tag-Entfernung fehlgeschlagen fuer #{doc['id']}: {exc}")


def learn_from_review(review_id: int, run_db: LocalStateDB,
                      paperless: PaperlessClient,
                      learning_profile: LearningProfile | None,
                      learning_examples: LearningExamples | None):
    """Lernt aus einem manuell korrigierten Review-Eintrag und schliesst ihn."""
    review = run_db.get_review_with_suggestion(review_id)
    if not review:
        log.warning(f"  Review #{review_id} nicht gefunden")
        return False
    doc_id = review["doc_id"]
    try:
        document = paperless.get_document(doc_id)
    except Exception as exc:
        log.warning(f"  Dokument #{doc_id} konnte nicht geladen werden: {exc}")
        run_db.close_review(review_id)
        return False

    if not _is_fully_organized(document):
        log.info(f"  Dokument #{doc_id} ist noch nicht vollstaendig organisiert - Review bleibt offen")
        return False

    # Aktuellen Stand als Suggestion bauen (IDs -> Namen aufloesen)
    try:
        all_correspondents = paperless.get_correspondents()
        all_doc_types = paperless.get_document_types()
        all_storage_paths = paperless.get_storage_paths()
        all_tags = paperless.get_tags()
    except Exception as exc:
        log.warning(f"  Stammdaten konnten nicht geladen werden: {exc}")
        run_db.close_review(review_id)
        return False

    corr_name = ""
    if document.get("correspondent"):
        for c in all_correspondents:
            if c.get("id") == document["correspondent"]:
                corr_name = c.get("name", "")
                break
    doctype_name = ""
    if document.get("document_type"):
        for dt in all_doc_types:
            if dt.get("id") == document["document_type"]:
                doctype_name = dt.get("name", "")
                break
    path_name = ""
    if document.get("storage_path"):
        for sp in all_storage_paths:
            if sp.get("id") == document["storage_path"]:
                path_name = sp.get("name", "")
                break
    tag_names = []
    doc_tag_ids = set(document.get("tags") or [])
    for tag in all_tags:
        if tag.get("id") in doc_tag_ids:
            tag_names.append(tag.get("name", ""))

    current_suggestion = {
        "correspondent": corr_name,
        "document_type": doctype_name,
        "storage_path": path_name,
        "tags": tag_names,
        "title": document.get("title", ""),
    }

    if learning_examples:
        try:
            learning_examples.append(document, current_suggestion)
            log.info(f"  Learning-Beispiel gespeichert fuer Dokument #{doc_id}")
        except Exception as exc:
            log.warning(f"  Learning-Beispiel konnte nicht gespeichert werden: {exc}")

    if learning_profile:
        try:
            learning_profile.learn_from_document(document, current_suggestion)
            learning_profile.save()
            log.info(f"  Learning-Profil aktualisiert fuer Dokument #{doc_id}")
        except Exception as exc:
            log.warning(f"  Learning-Profil konnte nicht aktualisiert werden: {exc}")

    run_db.close_review(review_id)
    _remove_review_tag_from_document(paperless, document)
    log.info(f"  Review #{review_id} geschlossen + gelernt (Dokument #{doc_id})")
    return True


def auto_resolve_reviews(run_db: LocalStateDB, paperless: PaperlessClient,
                         learning_profile: LearningProfile | None,
                         learning_examples: LearningExamples | None,
                         limit: int = 20) -> tuple[int, int]:
    """Prueft offene Reviews und schliesst automatisch, wenn Dokumente inzwischen organisiert sind."""
    open_reviews = run_db.list_open_reviews(limit=limit)
    if not open_reviews:
        return 0, 0

    resolved = 0
    checked = 0
    for item in open_reviews:
        checked += 1
        review_id = item["id"]
        try:
            result = learn_from_review(review_id, run_db, paperless,
                                       learning_profile, learning_examples)
            if result:
                resolved += 1
        except Exception as exc:
            log.warning(f"  Auto-Resolve Fehler bei Review #{review_id}: {exc}")

    return resolved, checked


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Menuesystem
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class App:
    """Hauptanwendung mit Menuesystem."""

    def __init__(self):
        self.dry_run = os.getenv("DEFAULT_DRY_RUN", "1").strip().lower() in ("1", "true", "yes", "on")
        self.llm_url = LLM_URL
        self.llm_model = LLM_MODEL
        self.paperless = None
        self.analyzer = None
        self.run_db = LocalStateDB(STATE_DB_FILE)
        self.taxonomy = TagTaxonomy(TAXONOMY_FILE)
        self.learning_profile = LearningProfile(LEARNING_PROFILE_FILE)
        self.learning_examples = LearningExamples(LEARNING_EXAMPLES_FILE)
        self._active_run_id = None
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
            log.warning(f"Signal empfangen: {sig_name} - fahre sauber herunter...")
            if self._active_run_id:
                try:
                    self.run_db.finish_run(self._active_run_id, {"aborted": True, "signal": sig_name})
                    log.info(f"Run #{self._active_run_id} sauber abgeschlossen")
                except Exception:
                    pass
            raise KeyboardInterrupt
        try:
            signal.signal(signal.SIGTERM, _handle_signal)
        except (OSError, AttributeError):
            pass

    def _start_run(self, action: str) -> int:
        run_id = self.run_db.start_run(action, self.dry_run, self.llm_model, self.llm_url)
        self._active_run_id = run_id
        log.info(f"Run gestartet: #{run_id} ({action})")
        return run_id

    def _finish_run(self, run_id: int, summary: dict):
        self.run_db.finish_run(run_id, summary)
        self._active_run_id = None
        log.info(f"Run abgeschlossen: #{run_id}")

    def _reconnect_services(self, reason: str = "") -> bool:
        """Versucht Paperless- und LLM-Verbindungen neu aufzubauen."""
        if reason:
            log.warning(f"Reconnect gestartet ({reason})")
        self.paperless = None
        self.analyzer = None
        if not self._init_paperless():
            return False
        if not self._init_analyzer():
            return False
        return True

    def _init_paperless(self) -> bool:
        """Paperless-Client initialisieren."""
        if self.paperless:
            return True
        url = os.getenv("PAPERLESS_URL")
        token = os.getenv("PAPERLESS_TOKEN")
        if not url or not token:
            console.print("[red]Fehler: .env mit PAPERLESS_URL und PAPERLESS_TOKEN noetig![/red]")
            return False
        self.paperless = PaperlessClient(url, token)
        return True

    def _init_analyzer(self) -> bool:
        """LLM-Analyzer initialisieren und Verbindung pruefen."""
        self.analyzer = LocalLLMAnalyzer(self.llm_url, self.llm_model)
        ok = self.analyzer.verify_connection()
        if ok and self.analyzer.model:
            self.llm_model = self.analyzer.model
        return ok

    def _load_master_data(self):
        """Stammdaten laden."""
        with console.status("Lade Stammdaten..."):
            tags = self.paperless.get_tags()
            correspondents = self.paperless.get_correspondents()
            doc_types = self.paperless.get_document_types()
            storage_paths = self.paperless.get_storage_paths()
        console.print(f"  {len(tags)} Tags | {len(correspondents)} Korrespondenten | "
                       f"{len(doc_types)} Typen | {len(storage_paths)} Speicherpfade")
        return tags, correspondents, doc_types, storage_paths

    def _ensure_taxonomy_tags(self, tags: list):
        """Erstellt fehlende Taxonomie-Tags automatisch (mit Farben), falls aktiviert."""
        if not AUTO_CREATE_TAXONOMY_TAGS:
            return
        if not self.taxonomy.canonical_tags:
            return

        existing = {_normalize_tag_name(t.get("name", "")) for t in tags}
        missing = [name for name in self.taxonomy.canonical_tags if _normalize_tag_name(name) not in existing]
        if not missing:
            return

        capacity = max(0, MAX_TOTAL_TAGS - len(tags))
        if capacity <= 0:
            log.warning("Taxonomie-Bootstrap uebersprungen: globales Tag-Limit erreicht (%s)", MAX_TOTAL_TAGS)
            return

        created = 0
        skipped_limit = 0
        table = Table(title="Taxonomie-Bootstrap", show_header=True)
        table.add_column("Tag", style="green")
        table.add_column("Farbe", style="cyan")
        table.add_column("Beschreibung", style="white")

        for canonical in missing:
            if created >= capacity:
                skipped_limit += 1
                continue
            color = self.taxonomy.color_for_tag(canonical)
            desc = self.taxonomy.description_for_tag(canonical)
            try:
                new_tag = self.paperless.create_tag(canonical, color=color)
                tags.append(new_tag)
                created += 1
                table.add_row(canonical, color, desc[:70])
            except requests.exceptions.HTTPError as exc:
                log.warning(f"Taxonomie-Tag konnte nicht erstellt werden: {canonical} ({exc})")

        if created:
            console.print(table)
            log.info("Taxonomie-Bootstrap: %s Tags erstellt", created)
        if skipped_limit:
            log.warning("Taxonomie-Bootstrap: %s Tags wegen Limit (%s) nicht erstellt", skipped_limit, MAX_TOTAL_TAGS)

    def _collect_decision_context(self, correspondents: list, storage_paths: list) -> DecisionContext:
        """Phase 1: Daten sammeln, danach Entscheidungen treffen."""
        with console.status("Sammle Entscheidungsdaten..."):
            documents = self.paperless.get_documents()
        context = build_decision_context(
            documents,
            correspondents,
            storage_paths,
            learning_profile=self.learning_profile,
        )
        log.info(
            "KONTEXT gesammelt: %s Dokumente | %s Arbeitgeber-Hints | %s Anbieter-Hints | %s Jobs | %s/%s Fahrzeuge",
            len(documents),
            len(context.employer_names),
            len(context.provider_names),
            len(context.profile_employment_lines),
            len(context.profile_private_vehicles),
            len(context.profile_company_vehicles),
        )
        return context

    def _show_header(self):
        """App-Header anzeigen."""
        url = os.getenv("PAPERLESS_URL", "?")
        mode = "[red]LIVE[/red]" if not self.dry_run else "[green]TESTLAUF[/green]"
        llm_model_label = self.llm_model or "auto/server-default"
        tag_policy = "bestehende Tags" if not ALLOW_NEW_TAGS else "neue Tags erlaubt"
        taxonomy_info = f"{len(self.taxonomy.canonical_tags)} Tags"
        auto_tax = "JA" if AUTO_CREATE_TAXONOMY_TAGS else "NEIN"
        recheck_info = "JA" if RECHECK_ALL_DOCS_IN_AUTO else "NEIN"
        console.print(Panel(
            f"[bold]Paperless-NGX Organizer[/bold]\n"
            f"Server: {url}\n"
            f"LLM: {llm_model_label} ({self.llm_url})\n"
            f"Modus: {mode}\n"
            f"Tag-Policy: {tag_policy} (max {MAX_TAGS_PER_DOC})\n"
            f"Taxonomie: {taxonomy_info} | Auto-Create: {auto_tax} | Global-Limit: {MAX_TOTAL_TAGS}\n"
            f"Auto-Modus recheck alle Dokumente: {recheck_info}\n"
            f"Learning-Profil: {LEARNING_PROFILE_FILE}\n"
            f"Learning-Beispiele: {LEARNING_EXAMPLES_FILE}\n"
            f"State-DB: {STATE_DB_FILE}",
            border_style="blue",
        ))

    def _menu(self, title: str, options: list) -> str:
        """Menue anzeigen und Auswahl zurueckgeben."""
        console.print(f"\n[bold]{title}[/bold]")
        for key, label in options:
            console.print(f"  [cyan]{key}[/cyan]  {label}")
        try:
            return Prompt.ask("\nAuswahl", default="0")
        except EOFError:
            # Non-interactive stdin closed -> graceful exit path.
            log.warning("Eingabestrom geschlossen (EOF) - beende Menue.")
            return "0"
        except KeyboardInterrupt:
            log.info("Eingabe vom Benutzer abgebrochen (Ctrl+C).")
            return "0"

    def _run_action_safely(self, action_fn, action_label: str):
        try:
            action_fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.error(f"{action_label} fehlgeschlagen: {exc}")
            console.print(f"[red]{action_label} fehlgeschlagen:[/red] {exc}")

    # --- Menues ---

    def menu_main(self):
        """Hauptmenue."""
        while True:
            self._show_header()
            choice = self._menu("Hauptmenue", [
                ("1", "Alles sortieren (unsortierte Dokumente automatisch organisieren)"),
                ("2", "Dokumente organisieren (erweiterte Optionen)"),
                ("3", "Aufraeumen"),
                ("4", "Duplikate finden"),
                ("5", "Statistiken / Uebersicht"),
                ("6", "Review-Queue"),
                ("7", "Einstellungen"),
                ("8", "Live-Watch (neue Dokumente automatisch verarbeiten)"),
                ("9", "Vollautomatik (Sortieren + Live-Watch + Wartung)"),
                ("0", "Beenden"),
            ])

            if choice == "1":
                self._run_action_safely(self.action_auto_organize, "Auto-Sortierung")
            elif choice == "2":
                self._run_action_safely(self.menu_organize, "Organisieren")
            elif choice == "3":
                self._run_action_safely(self.menu_cleanup, "Cleanup")
            elif choice == "4":
                self._run_action_safely(self.action_find_duplicates, "Duplikat-Scan")
            elif choice == "5":
                self._run_action_safely(self.action_statistics, "Statistiken")
            elif choice == "6":
                self._run_action_safely(self.action_review_queue, "Review-Queue")
            elif choice == "7":
                self._run_action_safely(self.menu_settings, "Einstellungen")
            elif choice == "8":
                self._run_action_safely(self.action_live_watch, "Live-Watch")
            elif choice == "9":
                self._run_action_safely(self.action_autopilot, "Vollautomatik")
            elif choice == "0":
                console.print("[bold]Auf Wiedersehen![/bold]")
                break
            else:
                console.print("[red]Ungueltige Auswahl.[/red]")

    def menu_organize(self):
        """Untermenue: Dokumente organisieren."""
        if not self._init_paperless():
            return
        if not self._init_analyzer():
            return

        tags, correspondents, doc_types, storage_paths = self._load_master_data()
        self._ensure_taxonomy_tags(tags)
        decision_context = self._collect_decision_context(correspondents, storage_paths)

        choice = self._menu("Dokumente organisieren", [
            ("1", "Einzelnes Dokument (nach ID)"),
            ("2", "Batch: Ohne Tags"),
            ("3", "Batch: Nicht vollstaendig organisiert"),
            ("4", "Batch: Alle Dokumente"),
            ("0", "Zurueck"),
        ])

        if choice == "1":
            doc_id = Prompt.ask("Dokument-ID")
            try:
                doc_id = int(doc_id)
            except ValueError:
                console.print("[red]Ungueltige ID.[/red]")
                return
            run_id = self._start_run("organize_single")
            ok = process_document(
                doc_id,
                self.paperless,
                self.analyzer,
                tags,
                correspondents,
                doc_types,
                storage_paths,
                self.dry_run,
                taxonomy=self.taxonomy,
                decision_context=decision_context,
                learning_profile=self.learning_profile,
                learning_examples=self.learning_examples,
                run_db=self.run_db,
                run_id=run_id,
            )
            self._finish_run(run_id, {"mode": "single", "doc_id": doc_id, "updated": int(ok)})

        elif choice in ("2", "3", "4"):
            limit_str = Prompt.ask("Limit (0 = alle)", default="0")
            try:
                limit = int(limit_str)
            except ValueError:
                limit = 0
            mode_map = {"2": "untagged", "3": "unorganized", "4": "all"}
            run_id = self._start_run(f"batch_{mode_map[choice]}")
            summary = batch_process(
                self.paperless,
                self.analyzer,
                tags,
                correspondents,
                doc_types,
                storage_paths,
                self.dry_run,
                mode=mode_map[choice],
                limit=limit,
                taxonomy=self.taxonomy,
                decision_context=decision_context,
                learning_profile=self.learning_profile,
                learning_examples=self.learning_examples,
                run_db=self.run_db,
                run_id=run_id,
            )
            self._finish_run(run_id, summary)

    def action_auto_organize(self):
        """Alles sortieren - scannt alle Dokumente, ueberspringt bereits fertige."""
        if not self._init_paperless():
            return
        if not self._init_analyzer():
            return

        force_recheck_all = Confirm.ask(
            "Alle Dokumente fuer diesen Lauf neu pruefen?",
            default=RECHECK_ALL_DOCS_IN_AUTO,
        )

        tags, correspondents, doc_types, storage_paths = self._load_master_data()
        self._ensure_taxonomy_tags(tags)
        decision_context = self._collect_decision_context(correspondents, storage_paths)
        action_name = "auto_organize_recheck_all" if force_recheck_all else "auto_organize"
        run_id = self._start_run(action_name)
        summary = auto_organize_all(
            self.paperless,
            self.analyzer,
            tags,
            correspondents,
            doc_types,
            storage_paths,
            self.dry_run,
            force_recheck_all=force_recheck_all,
            taxonomy=self.taxonomy,
            decision_context=decision_context,
            learning_profile=self.learning_profile,
            learning_examples=self.learning_examples,
            run_db=self.run_db,
            run_id=run_id,
        )
        self._finish_run(run_id, summary)

    def action_live_watch(self):
        """Live-Modus: Pollt regelmaessig auf neue Dokumente und verarbeitet diese automatisch."""
        if not self._init_paperless():
            return
        if not self._init_analyzer():
            return

        if not self.dry_run and not Confirm.ask(
            "[red]LIVE-Modus aktiv! Neuen Dokumente automatisch verarbeiten?[/red]",
            default=True,
        ):
            console.print("[dim]Abgebrochen.[/dim]")
            return

        interval_default = max(10, LIVE_WATCH_INTERVAL_SEC)
        interval_str = Prompt.ask("Polling-Intervall in Sekunden", default=str(interval_default))
        try:
            interval_sec = max(5, int(interval_str))
        except ValueError:
            interval_sec = interval_default

        include_existing = Confirm.ask(
            "Beim Start auch bereits vorhandene unvollstaendige Dokumente mitnehmen?",
            default=False,
        )
        refresh_cycles = max(1, LIVE_WATCH_CONTEXT_REFRESH_CYCLES)

        with console.status("Initialisiere Live-Watch..."):
            initial_docs = self.paperless.get_documents()

        known_ids = set()
        if not include_existing:
            known_ids = {int(d["id"]) for d in initial_docs if d.get("id") is not None}

        run_id = self._start_run("live_watch")
        log.info(
            "LIVE-WATCH gestartet: Intervall=%ss | Initial-Docs=%s | include_existing=%s",
            interval_sec,
            len(initial_docs),
            "JA" if include_existing else "NEIN",
        )
        console.print("[cyan]Live-Watch laeuft. Stoppen mit Ctrl+C.[/cyan]")

        cycle = 0
        total_seen_new = 0
        total_candidates = 0
        total_updated = 0
        total_skipped_duplicates = 0
        total_skipped_fully_organized = 0
        total_poll_errors = 0
        total_doc_failures = 0
        reconnect_attempts = 0
        consecutive_poll_errors = 0
        tags = []
        correspondents = []
        doc_types = []
        storage_paths = []
        decision_context = None

        try:
            while True:
                cycle += 1
                try:
                    docs = self.paperless.get_documents()
                except Exception as exc:
                    total_poll_errors += 1
                    consecutive_poll_errors += 1
                    backoff = min(
                        WATCH_ERROR_BACKOFF_MAX_SEC,
                        interval_sec + (consecutive_poll_errors - 1) * max(1, WATCH_ERROR_BACKOFF_BASE_SEC),
                    )
                    log.error(f"LIVE-WATCH Poll-Fehler: {exc} (retry in {backoff}s)")
                    if consecutive_poll_errors >= max(1, WATCH_RECONNECT_ERROR_THRESHOLD):
                        reconnect_attempts += 1
                        if self._reconnect_services(f"live-watch poll errors={consecutive_poll_errors}"):
                            consecutive_poll_errors = 0
                    time.sleep(backoff)
                    continue
                consecutive_poll_errors = 0

                docs_by_id = {
                    int(d["id"]): d
                    for d in docs
                    if d.get("id") is not None
                }
                current_ids = set(docs_by_id.keys())
                new_ids = sorted(i for i in current_ids if i not in known_ids)
                known_ids.update(current_ids)

                if not new_ids:
                    if cycle == 1 or cycle % 5 == 0:
                        log.info(f"LIVE-WATCH: keine neuen Dokumente, warte {interval_sec}s...")
                    time.sleep(interval_sec)
                    continue

                total_seen_new += len(new_ids)
                log.info(f"LIVE-WATCH: {len(new_ids)} neue Dokumente erkannt")

                try:
                    tags, correspondents, doc_types, storage_paths = self._load_master_data()
                    self._ensure_taxonomy_tags(tags)
                    if decision_context is None or (cycle % refresh_cycles == 0):
                        decision_context = self._collect_decision_context(correspondents, storage_paths)
                except Exception as exc:
                    total_poll_errors += 1
                    log.error(f"LIVE-WATCH Stammdaten/Kontext-Fehler: {exc}")
                    time.sleep(min(interval_sec, 15))
                    continue

                duplikat_tag_id = next((t["id"] for t in tags if t["name"] == "Duplikat"), None)
                for doc_id in new_ids:
                    doc = docs_by_id.get(doc_id, {})
                    if duplikat_tag_id and duplikat_tag_id in (doc.get("tags") or []):
                        total_skipped_duplicates += 1
                        continue
                    if _is_fully_organized(doc):
                        total_skipped_fully_organized += 1
                        continue

                    total_candidates += 1
                    try:
                        if process_document(
                            doc_id,
                            self.paperless,
                            self.analyzer,
                            tags,
                            correspondents,
                            doc_types,
                            storage_paths,
                            self.dry_run,
                            batch_mode=not self.dry_run,
                            prefer_compact=LIVE_WATCH_COMPACT_FIRST,
                            taxonomy=self.taxonomy,
                            decision_context=decision_context,
                            learning_profile=self.learning_profile,
                            learning_examples=self.learning_examples,
                            run_db=self.run_db,
                            run_id=run_id,
                        ):
                            total_updated += 1
                        else:
                            total_doc_failures += 1
                    except Exception as exc:
                        total_doc_failures += 1
                        total_poll_errors += 1
                        log.error(f"LIVE-WATCH Fehler bei Dokument #{doc_id}: {exc}")

                time.sleep(interval_sec)
        except KeyboardInterrupt:
            log.info("LIVE-WATCH vom Benutzer gestoppt")
            console.print("[yellow]Live-Watch gestoppt.[/yellow]")
        finally:
            summary = {
                "mode": "live_watch",
                "cycles": cycle,
                "seen_new": total_seen_new,
                "candidates": total_candidates,
                "updated": total_updated,
                "failed_docs": total_doc_failures,
                "skipped_duplicates": total_skipped_duplicates,
                "skipped_fully_organized": total_skipped_fully_organized,
                "poll_errors": total_poll_errors,
                "reconnect_attempts": reconnect_attempts,
                "interval_sec": interval_sec,
                "include_existing": include_existing,
            }
            self._finish_run(run_id, summary)

    def action_autopilot(self):
        """Vollautomatik: initial sortieren, dann dauerhaft live beobachten + periodische Wartung."""
        if not self._init_paperless():
            return
        if not self._init_analyzer():
            return

        if not self.dry_run and not Confirm.ask(
            "[red]Vollautomatik im LIVE-Modus starten?[/red]",
            default=True,
        ):
            console.print("[dim]Abgebrochen.[/dim]")
            return

        interval_sec = max(5, AUTOPILOT_INTERVAL_SEC)
        refresh_cycles = max(1, AUTOPILOT_CONTEXT_REFRESH_CYCLES)
        cleanup_every = max(0, AUTOPILOT_CLEANUP_EVERY_CYCLES)
        dupscan_every = max(0, AUTOPILOT_DUPLICATE_SCAN_EVERY_CYCLES)
        review_resolve_every = max(0, AUTOPILOT_REVIEW_RESOLVE_EVERY_CYCLES)
        max_new_per_cycle = max(0, AUTOPILOT_MAX_NEW_DOCS_PER_CYCLE)

        run_id = self._start_run("autopilot")
        total_seen_new = 0
        total_candidates = 0
        total_updated = 0
        total_skipped_duplicates = 0
        total_skipped_fully_organized = 0
        total_poll_errors = 0
        total_doc_failures = 0
        total_maintenance_runs = 0
        total_duplicate_scans = 0
        total_reviews_resolved = 0
        reconnect_attempts = 0
        consecutive_poll_errors = 0
        cycle = 0
        tags = []
        correspondents = []
        doc_types = []
        storage_paths = []
        decision_context = None

        try:
            with console.status("Initialisiere Vollautomatik..."):
                docs = self.paperless.get_documents()
        except Exception as exc:
            log.error(f"AUTOPILOT Initialisierung fehlgeschlagen (get_documents): {exc}")
            self._finish_run(run_id, {
                "mode": "autopilot",
                "error": f"init_get_documents_failed: {exc}",
                "cycles": 0,
                "seen_new": 0,
                "candidates": 0,
                "updated": 0,
                "poll_errors": 1,
            })
            return
        known_ids = {int(d["id"]) for d in docs if d.get("id") is not None}

        # Quick LLM health check before starting long-running loop
        log.info("AUTOPILOT: LLM-Gesundheitscheck...")
        if not self.analyzer.verify_connection():
            log.error("AUTOPILOT: LLM nicht erreichbar - starte trotzdem (Learning-Priors als Fallback)")
        else:
            log.info("AUTOPILOT: LLM bereit")

        log.info(
            "AUTOPILOT gestartet: interval=%ss | start_auto_organize=%s | cleanup_every=%s | dupscan_every=%s",
            interval_sec,
            "JA" if AUTOPILOT_START_WITH_AUTO_ORGANIZE else "NEIN",
            cleanup_every,
            dupscan_every,
        )
        console.print("[cyan]Vollautomatik laeuft. Stoppen mit Ctrl+C.[/cyan]")

        try:
            if AUTOPILOT_START_WITH_AUTO_ORGANIZE:
                try:
                    tags, correspondents, doc_types, storage_paths = self._load_master_data()
                    self._ensure_taxonomy_tags(tags)
                    decision_context = self._collect_decision_context(correspondents, storage_paths)
                    auto_summary = auto_organize_all(
                        self.paperless,
                        self.analyzer,
                        tags,
                        correspondents,
                        doc_types,
                        storage_paths,
                        self.dry_run,
                        force_recheck_all=AUTOPILOT_RECHECK_ALL_ON_START,
                        prefer_compact=LIVE_WATCH_COMPACT_FIRST,
                        taxonomy=self.taxonomy,
                        decision_context=decision_context,
                        learning_profile=self.learning_profile,
                        learning_examples=self.learning_examples,
                        run_db=self.run_db,
                        run_id=run_id,
                    )
                    total_updated += int(auto_summary.get("applied", 0) or 0)
                    total_candidates += int(auto_summary.get("todo", 0) or 0)
                    try:
                        docs = self.paperless.get_documents()
                        known_ids = {int(d["id"]) for d in docs if d.get("id") is not None}
                    except Exception as exc:
                        total_poll_errors += 1
                        log.warning(f"AUTOPILOT: Re-Load nach Initialsortierung fehlgeschlagen: {exc}")
                except Exception as exc:
                    total_poll_errors += 1
                    log.error(f"AUTOPILOT: Initiale Auto-Sortierung fehlgeschlagen: {exc}")

            while True:
                cycle += 1
                try:
                    docs = self.paperless.get_documents()
                except Exception as exc:
                    total_poll_errors += 1
                    consecutive_poll_errors += 1
                    backoff = min(
                        WATCH_ERROR_BACKOFF_MAX_SEC,
                        interval_sec + (consecutive_poll_errors - 1) * max(1, WATCH_ERROR_BACKOFF_BASE_SEC),
                    )
                    log.error(f"AUTOPILOT Poll-Fehler: {exc} (retry in {backoff}s)")
                    if consecutive_poll_errors >= max(1, WATCH_RECONNECT_ERROR_THRESHOLD):
                        reconnect_attempts += 1
                        if self._reconnect_services(f"autopilot poll errors={consecutive_poll_errors}"):
                            consecutive_poll_errors = 0
                    time.sleep(backoff)
                    continue
                consecutive_poll_errors = 0

                docs_by_id = {int(d["id"]): d for d in docs if d.get("id") is not None}
                current_ids = set(docs_by_id.keys())
                new_ids = sorted(i for i in current_ids if i not in known_ids)
                known_ids.update(current_ids)

                if max_new_per_cycle > 0 and len(new_ids) > max_new_per_cycle:
                    log.warning(
                        "AUTOPILOT: %s neue Dokumente erkannt, begrenze auf %s in diesem Zyklus",
                        len(new_ids),
                        max_new_per_cycle,
                    )
                    new_ids = new_ids[:max_new_per_cycle]

                if new_ids:
                    total_seen_new += len(new_ids)
                    log.info(f"AUTOPILOT: {len(new_ids)} neue Dokumente erkannt")

                    try:
                        tags, correspondents, doc_types, storage_paths = self._load_master_data()
                        self._ensure_taxonomy_tags(tags)
                        if decision_context is None or (cycle % refresh_cycles == 0):
                            decision_context = self._collect_decision_context(correspondents, storage_paths)
                    except Exception as exc:
                        total_poll_errors += 1
                        log.error(f"AUTOPILOT Stammdaten/Kontext-Fehler: {exc}")
                        time.sleep(min(interval_sec, 15))
                        continue

                    duplikat_tag_id = next((t["id"] for t in tags if t["name"] == "Duplikat"), None)
                    for doc_id in new_ids:
                        doc = docs_by_id.get(doc_id, {})
                        if duplikat_tag_id and duplikat_tag_id in (doc.get("tags") or []):
                            total_skipped_duplicates += 1
                            continue
                        if _is_fully_organized(doc):
                            total_skipped_fully_organized += 1
                            continue

                        total_candidates += 1
                        try:
                            if process_document(
                                doc_id,
                                self.paperless,
                                self.analyzer,
                                tags,
                                correspondents,
                                doc_types,
                                storage_paths,
                                self.dry_run,
                                batch_mode=not self.dry_run,
                                prefer_compact=LIVE_WATCH_COMPACT_FIRST,
                                taxonomy=self.taxonomy,
                                decision_context=decision_context,
                                learning_profile=self.learning_profile,
                                learning_examples=self.learning_examples,
                                run_db=self.run_db,
                                run_id=run_id,
                            ):
                                total_updated += 1
                            else:
                                total_doc_failures += 1
                        except Exception as exc:
                            total_doc_failures += 1
                            total_poll_errors += 1
                            log.error(f"AUTOPILOT Fehler bei Dokument #{doc_id}: {exc}")
                elif cycle == 1 or cycle % 5 == 0:
                    log.info(f"AUTOPILOT: keine neuen Dokumente, warte {interval_sec}s...")

                if cleanup_every > 0 and cycle % cleanup_every == 0:
                    try:
                        log.info("AUTOPILOT: periodisches Cleanup gestartet")
                        cleanup_tags(self.paperless, self.dry_run)
                        cleanup_correspondents(self.paperless, self.dry_run)
                        cleanup_document_types(self.paperless, self.dry_run)
                        # Purge old DB entries
                        purge_result = self.run_db.purge_old_runs(keep_days=90)
                        if purge_result["runs"] > 0:
                            log.info(
                                "AUTOPILOT: DB-Cleanup: %s Runs, %s Dokumente, %s Tag-Events geloescht",
                                purge_result["runs"], purge_result["documents"], purge_result["tag_events"],
                            )
                        total_maintenance_runs += 1
                    except Exception as exc:
                        total_poll_errors += 1
                        log.error(f"AUTOPILOT Cleanup-Fehler: {exc}")

                if dupscan_every > 0 and cycle % dupscan_every == 0:
                    try:
                        log.info("AUTOPILOT: periodischer Duplikat-Scan gestartet")
                        find_duplicates(self.paperless)
                        total_duplicate_scans += 1
                    except Exception as exc:
                        total_poll_errors += 1
                        log.error(f"AUTOPILOT Duplikat-Scan-Fehler: {exc}")

                if review_resolve_every > 0 and cycle % review_resolve_every == 0:
                    try:
                        log.info("AUTOPILOT: Review-Queue pruefen...")
                        resolved, checked = auto_resolve_reviews(
                            self.run_db, self.paperless,
                            self.learning_profile, self.learning_examples)
                        if resolved:
                            log.info(f"AUTOPILOT: {resolved}/{checked} Reviews automatisch geschlossen")
                        total_reviews_resolved += resolved
                    except Exception as exc:
                        total_poll_errors += 1
                        log.error(f"AUTOPILOT Review-Resolve-Fehler: {exc}")

                # Periodic status summary
                if cycle % 10 == 0:
                    uptime_min = (cycle * interval_sec) / 60
                    status_table = Table(title=f"Autopilot Status (Zyklus {cycle})", show_header=True)
                    status_table.add_column("Metrik", style="cyan")
                    status_table.add_column("Wert", style="green", justify="right")
                    status_table.add_row("Laufzeit", f"{uptime_min:.0f} min")
                    status_table.add_row("Neue Dokumente", str(total_seen_new))
                    status_table.add_row("Verarbeitet", str(total_candidates))
                    status_table.add_row("Aktualisiert", str(total_updated))
                    status_table.add_row("Fehler", str(total_doc_failures))
                    status_table.add_row("Uebersprungen (Duplikate)", str(total_skipped_duplicates))
                    status_table.add_row("Uebersprungen (fertig)", str(total_skipped_fully_organized))
                    status_table.add_row("Wartungslaeufe", str(total_maintenance_runs))
                    status_table.add_row("Reviews geloest", str(total_reviews_resolved))
                    status_table.add_row("Poll-Fehler", str(total_poll_errors))
                    console.print(status_table)

                time.sleep(interval_sec)
        except KeyboardInterrupt:
            log.info("AUTOPILOT vom Benutzer gestoppt")
            console.print("[yellow]Vollautomatik gestoppt.[/yellow]")
        finally:
            summary = {
                "mode": "autopilot",
                "cycles": cycle,
                "seen_new": total_seen_new,
                "candidates": total_candidates,
                "updated": total_updated,
                "failed_docs": total_doc_failures,
                "skipped_duplicates": total_skipped_duplicates,
                "skipped_fully_organized": total_skipped_fully_organized,
                "maintenance_runs": total_maintenance_runs,
                "duplicate_scans": total_duplicate_scans,
                "reviews_resolved": total_reviews_resolved,
                "poll_errors": total_poll_errors,
                "reconnect_attempts": reconnect_attempts,
                "interval_sec": interval_sec,
                "start_auto_organize": AUTOPILOT_START_WITH_AUTO_ORGANIZE,
                "start_recheck_all": AUTOPILOT_RECHECK_ALL_ON_START,
            }
            self._finish_run(run_id, summary)

    def menu_cleanup(self):
        """Untermenue: Aufraeumen."""
        if not self._init_paperless():
            return

        choice = self._menu("Aufraeumen", [
            ("1", "Tags"),
            ("2", "Korrespondenten"),
            ("3", "Dokumenttypen"),
            ("4", "Alles"),
            ("0", "Zurueck"),
        ])

        if not self.dry_run:
            if not Confirm.ask("[red]LIVE-Modus aktiv! Wirklich aufraeumen?[/red]"):
                console.print("[dim]Abgebrochen.[/dim]")
                return

        if choice == "1":
            cleanup_tags(self.paperless, self.dry_run)
        elif choice == "2":
            cleanup_correspondents(self.paperless, self.dry_run)
        elif choice == "3":
            cleanup_document_types(self.paperless, self.dry_run)
        elif choice == "4":
            cleanup_all(self.paperless, self.dry_run)

    def menu_settings(self):
        """Untermenue: Einstellungen."""
        global ALLOW_NEW_TAGS, AUTO_CLEANUP_AFTER_ORGANIZE, ENABLE_WEB_HINTS
        global AUTO_CREATE_TAXONOMY_TAGS, RECHECK_ALL_DOCS_IN_AUTO
        llm_model_label = self.llm_model or "auto/server-default"
        choice = self._menu("Einstellungen", [
            ("1", f"Modus umschalten (aktuell: {'TESTLAUF' if self.dry_run else 'LIVE'})"),
            ("2", f"LLM-Modell aendern (aktuell: {llm_model_label})"),
            ("3", f"LLM-URL aendern (aktuell: {self.llm_url})"),
            ("4", "LLM-Verbindung testen"),
            ("5", f"Neue Tags erlauben (aktuell: {'JA' if ALLOW_NEW_TAGS else 'NEIN'})"),
            ("6", f"Auto-Cleanup nach Auto/Batches (aktuell: {'JA' if AUTO_CLEANUP_AFTER_ORGANIZE else 'NEIN'})"),
            ("7", f"Web-Hinweise aktiv (aktuell: {'JA' if ENABLE_WEB_HINTS else 'NEIN'})"),
            ("8", "Taxonomie neu laden"),
            ("9", f"Taxonomie-Tags auto-erstellen (aktuell: {'JA' if AUTO_CREATE_TAXONOMY_TAGS else 'NEIN'})"),
            ("10", f"Menue 1: alle Dokumente neu pruefen (aktuell: {'JA' if RECHECK_ALL_DOCS_IN_AUTO else 'NEIN'})"),
            ("11", "Learning-Daten sichern (Backup)"),
            ("12", "Datenbank bereinigen (alte Runs loeschen)"),
            ("0", "Zurueck"),
        ])

        if choice == "1":
            if self.dry_run:
                if Confirm.ask("[red]In LIVE-Modus wechseln? Aenderungen werden wirklich angewendet![/red]"):
                    self.dry_run = False
                    console.print("[red]LIVE-Modus aktiviert.[/red]")
            else:
                self.dry_run = True
                console.print("[green]TESTLAUF-Modus aktiviert.[/green]")

        elif choice == "2":
            new_model = Prompt.ask("Neues Modell", default=self.llm_model)
            self.llm_model = new_model
            self.analyzer = None  # Neuinitialisierung erzwingen
            console.print(f"[green]Modell geaendert: {self.llm_model}[/green]")

        elif choice == "3":
            new_url = Prompt.ask("Neue URL", default=self.llm_url)
            self.llm_url = new_url
            self.analyzer = None
            console.print(f"[green]URL geaendert: {self.llm_url}[/green]")

        elif choice == "4":
            self._init_analyzer()

        elif choice == "5":
            ALLOW_NEW_TAGS = not ALLOW_NEW_TAGS
            console.print(f"[green]Neue Tags {'aktiviert' if ALLOW_NEW_TAGS else 'deaktiviert'}.[/green]")

        elif choice == "6":
            AUTO_CLEANUP_AFTER_ORGANIZE = not AUTO_CLEANUP_AFTER_ORGANIZE
            console.print(f"[green]Auto-Cleanup {'aktiviert' if AUTO_CLEANUP_AFTER_ORGANIZE else 'deaktiviert'}.[/green]")

        elif choice == "7":
            ENABLE_WEB_HINTS = not ENABLE_WEB_HINTS
            console.print(f"[green]Web-Hinweise {'aktiviert' if ENABLE_WEB_HINTS else 'deaktiviert'}.[/green]")

        elif choice == "8":
            self.taxonomy.load()
            console.print(f"[green]Taxonomie neu geladen ({len(self.taxonomy.canonical_tags)} Tags).[/green]")

        elif choice == "9":
            AUTO_CREATE_TAXONOMY_TAGS = not AUTO_CREATE_TAXONOMY_TAGS
            console.print(f"[green]Taxonomie-Tag-Autocreate {'aktiviert' if AUTO_CREATE_TAXONOMY_TAGS else 'deaktiviert'}.[/green]")

        elif choice == "10":
            RECHECK_ALL_DOCS_IN_AUTO = not RECHECK_ALL_DOCS_IN_AUTO
            console.print(f"[green]Auto-Recheck {'aktiviert' if RECHECK_ALL_DOCS_IN_AUTO else 'deaktiviert'}.[/green]")

        elif choice == "11":
            import shutil
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(LOG_DIR, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            backed_up = []
            for src_file in [LEARNING_EXAMPLES_FILE, LEARNING_PROFILE_FILE]:
                if os.path.exists(src_file):
                    base = os.path.basename(src_file)
                    dst = os.path.join(backup_dir, f"{timestamp}_{base}")
                    shutil.copy2(src_file, dst)
                    backed_up.append(dst)
            if backed_up:
                for f in backed_up:
                    console.print(f"  [green]Gesichert:[/green] {f}")
                console.print(f"[green]{len(backed_up)} Dateien gesichert nach {backup_dir}[/green]")
            else:
                console.print("[yellow]Keine Learning-Dateien zum Sichern gefunden.[/yellow]")

        elif choice == "12":
            days_str = Prompt.ask("Eintraege aelter als N Tage loeschen", default="90")
            try:
                days = max(7, int(days_str))
            except ValueError:
                days = 90
            result = self.run_db.purge_old_runs(keep_days=days)
            if result["runs"] > 0:
                console.print(
                    f"[green]Geloescht: {result['runs']} Runs, {result['documents']} Dokumente, "
                    f"{result['tag_events']} Tag-Events, {result['reviews']} Reviews[/green]"
                )
            else:
                console.print("[dim]Keine alten Eintraege gefunden.[/dim]")

    def action_find_duplicates(self):
        """Duplikate finden."""
        if not self._init_paperless():
            return
        find_duplicates(self.paperless)

    def action_statistics(self):
        """Statistiken-Untermenue."""
        while True:
            choice = self._menu("Statistiken", [
                ("1", "Paperless-Uebersicht"),
                ("2", "Monatlicher Report"),
                ("3", "Review-Queue auto-resolve"),
                ("0", "Zurueck"),
            ])
            if choice == "1":
                if not self._init_paperless():
                    return
                show_statistics(self.paperless)
            elif choice == "2":
                show_monthly_report(self.run_db, self.paperless)
            elif choice == "3":
                if not self._init_paperless():
                    return
                with console.status("Pruefe offene Reviews..."):
                    resolved, checked = auto_resolve_reviews(
                        self.run_db, self.paperless,
                        self.learning_profile, self.learning_examples)
                if checked == 0:
                    console.print("[dim]Keine offenen Reviews vorhanden.[/dim]")
                elif resolved > 0:
                    console.print(f"[green]{resolved}/{checked} Reviews automatisch geschlossen + gelernt.[/green]")
                else:
                    console.print(f"[yellow]0/{checked} Reviews konnten automatisch geschlossen werden.[/yellow]")
            elif choice == "0":
                break

    def action_review_queue(self):
        """Offene menschliche Nachpruefungen anzeigen."""
        open_items = self.run_db.list_open_reviews(limit=100)

        # Learning stats summary
        n_examples = len(self.learning_examples.examples) if self.learning_examples else 0
        n_profiles = 0
        if self.learning_examples:
            profiles = self.learning_examples._build_correspondent_profiles()
            n_profiles = len(profiles)
        console.print(
            f"[dim]Learning: {n_examples} Beispiele, {n_profiles} Korrespondent-Profile[/dim]"
        )

        table = Table(title=f"Review-Queue ({len(open_items)} offen)", show_header=True)
        table.add_column("ID", width=6)
        table.add_column("Dokument", width=10)
        table.add_column("Grund")
        table.add_column("Aktualisiert", width=20)
        for item in open_items:
            table.add_row(str(item["id"]), str(item["doc_id"]), item["reason"], item["updated_at"])
        console.print(table)
        if not open_items:
            return
        if Confirm.ask("Einen Review-Eintrag als erledigt markieren?", default=False):
            review_id_str = Prompt.ask("Review-ID")
            try:
                review_id = int(review_id_str)
            except ValueError:
                console.print("[red]Ungueltige ID.[/red]")
                return
            if not self._init_paperless():
                if self.run_db.close_review(review_id):
                    console.print("[green]Review-Eintrag geschlossen (ohne Lernen).[/green]")
                else:
                    console.print("[yellow]Kein offener Eintrag mit dieser ID gefunden.[/yellow]")
                return
            result = learn_from_review(
                review_id, self.run_db, self.paperless,
                self.learning_profile, self.learning_examples,
            )
            if result:
                console.print("[green]Review-Eintrag geschlossen + gelernt.[/green]")
            elif self.run_db.close_review(review_id):
                console.print("[green]Review-Eintrag geschlossen (Dokument noch nicht fertig organisiert).[/green]")
            else:
                console.print("[yellow]Kein offener Eintrag mit dieser ID gefunden.[/yellow]")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Main
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main():
    log.info("=" * 40)
    log.info(f"[bold]Paperless-NGX Organizer v{__version__}[/bold]")
    log.info(f"  Python: {sys.version.split()[0]}")
    log.info(f"  LLM: {LLM_MODEL or '(auto)'} @ {LLM_URL}")
    log.info(f"  Log-Datei: {LOG_FILE}")
    log.info(f"  State-DB: {STATE_DB_FILE}")
    log.info("=" * 40)
    app = App()
    try:
        app.menu_main()
    except KeyboardInterrupt:
        console.print("\n[bold]Abgebrochen.[/bold]")
    finally:
        try:
            log.info("Paperless-NGX Organizer beendet")
        except KeyboardInterrupt:
            pass
        except Exception:
            pass


if __name__ == "__main__":
    main()
