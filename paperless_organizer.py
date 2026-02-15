"""
Paperless-NGX Organizer - Unified Terminal App
Vereint: main.py, cleanup.py, cleanup_correspondents.py, cleanup_doctypes.py,
         batch_delete_tags.py, find_duplicates.py, fast_cleanup.py

Nutzt lokales LLM via LM Studio statt Claude CLI.
Interaktive Rich Terminal-UI mit Menuesystem.
"""

from __future__ import annotations

import os
import sys
import json
import re
import time
import logging
import sqlite3
import threading
import difflib
import requests
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode, urlparse, urlunparse
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.logging import RichHandler

load_dotenv()
console = Console()

# ── Logging-Setup ────────────────────────────────────────────────────────────
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "organizer.log")
STATE_DB_FILE = os.path.join(LOG_DIR, "organizer_state.db")
TAXONOMY_FILE = os.path.join(LOG_DIR, "taxonomy_tags.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        RichHandler(console=console, show_path=False, markup=True, rich_tracebacks=True),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
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

    def close_review(self, review_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE review_queue SET status = 'resolved', updated_at = ? WHERE id = ? AND status = 'open'",
                (datetime.now().isoformat(timespec="seconds"), review_id),
            )
            return cur.rowcount > 0


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# --- LLM (LM Studio) ---
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:1234/v1/chat/completions").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", "").strip()
LLM_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "").strip()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))

# --- Paperless-NGX ---
OWNER_ID = 4  # Edgar

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
TAG_DELETE_THRESHOLD = 1      # Tags mit <= X Dokumenten loeschen (ausser Whitelist)
TAG_ENGLISH_THRESHOLD = 3     # Englische Tags mit <= X Dokumenten loeschen
NON_TAXONOMY_DELETE_THRESHOLD = int(os.getenv("NON_TAXONOMY_DELETE_THRESHOLD", "5"))

# --- Korrespondenten-Gruppen fuer Merge ---
CORRESPONDENT_GROUPS = {
    "DRK": ["drk", "deutsches rotes kreuz", "rotes kreuz", "wasserwacht", "blutspende"],
    "AOK PLUS": ["aok"],
    "IHK Dresden": ["ihk", "industrie- und handelskammer dresden", "industriekammer dresden", "thresien"],
    "WBS TRAINING AG": ["wbs training", "wbs training ag"],
    "msg systems ag": ["msg"],
    "Amazon": ["amazon"],
    "Baader Bank": ["baader"],
    "Apple": ["apple"],
    "Digistore24": ["digistore"],
    "CHECK24": ["check24"],
    "AXA": ["axa"],
    "Scalable Capital": ["scalable"],
    "DHL": ["dhl"],
    "Corporate Benefits": ["corporate benefits"],
    "1&1": ["1&1", "1und1", "1 & 1"],
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
RECHECK_ALL_DOCS_IN_AUTO = os.getenv("RECHECK_ALL_DOCS_IN_AUTO", "0").strip().lower() in ("1", "true", "yes", "on")
REVIEW_TAG_NAME = os.getenv("REVIEW_TAG_NAME", "Manuell-Pruefen")
AUTO_APPLY_REVIEW_TAG = os.getenv("AUTO_APPLY_REVIEW_TAG", "1").strip().lower() in ("1", "true", "yes", "on")
REVIEW_ON_MEDIUM_CONFIDENCE = os.getenv("REVIEW_ON_MEDIUM_CONFIDENCE", "0").strip().lower() in ("1", "true", "yes", "on")
USE_ARCHIVE_SERIAL_NUMBER = os.getenv("USE_ARCHIVE_SERIAL_NUMBER", "0").strip().lower() in ("1", "true", "yes", "on")
CORRESPONDENT_MATCH_THRESHOLD = float(os.getenv("CORRESPONDENT_MATCH_THRESHOLD", "0.86"))

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


@dataclass
class DecisionContext:
    """Collected runtime context for better document decisions."""
    employer_names: set[str] = field(default_factory=set)          # normalized names
    provider_names: set[str] = field(default_factory=set)          # normalized keys
    top_work_paths: list[str] = field(default_factory=list)
    top_private_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

# --- German-Character Detection ---
GERMAN_CHARS_RE = re.compile(r'[aeoeueAeOeUess]')


# ══════════════════════════════════════════════════════════════════════════════
# PaperlessClient
# ══════════════════════════════════════════════════════════════════════════════

class PaperlessClient:
    """Vereinter Client fuer die Paperless-NGX REST API."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token
        self._color_index = 0
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        })

    def _get_all(self, endpoint: str) -> list:
        """Alle Eintraege mit Paginierung laden."""
        results = []
        url = f"{self.url}/api/{endpoint}/?page_size=100"
        while url:
            url = url.replace("http://", "https://")
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))
            url = data.get("next")
        return results

    # --- Lesen ---
    def get_tags(self) -> list:
        return self._get_all("tags")

    def get_correspondents(self) -> list:
        return self._get_all("correspondents")

    def get_document_types(self) -> list:
        return self._get_all("document_types")

    def get_storage_paths(self) -> list:
        return self._get_all("storage_paths")

    def get_documents(self) -> list:
        return self._get_all("documents")

    def get_document(self, doc_id: int) -> dict:
        resp = self.session.get(f"{self.url}/api/documents/{doc_id}/", timeout=30)
        resp.raise_for_status()
        return resp.json()

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

    def update_document(self, doc_id: int, data: dict) -> dict:
        resp = self.session.patch(
            f"{self.url}/api/documents/{doc_id}/",
            json=data, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def create_tag(self, name: str, color: str | None = None, text_color: str | None = None) -> dict:
        color = (color or self._next_color()).strip().lower()
        text_color = (text_color or self._text_color_for_background(color)).strip().lower()
        data = {"name": name, "color": color, "text_color": text_color}
        resp = self.session.post(
            f"{self.url}/api/tags/",
            json=self._with_permissions(data), timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def create_correspondent(self, name: str) -> dict:
        resp = self.session.post(
            f"{self.url}/api/correspondents/",
            json=self._with_permissions({"name": name}), timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def create_document_type(self, name: str) -> dict:
        resp = self.session.post(
            f"{self.url}/api/document_types/",
            json=self._with_permissions({"name": name}), timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def create_storage_path(self, name: str, path: str) -> dict:
        resp = self.session.post(
            f"{self.url}/api/storage_paths/",
            json=self._with_permissions({"name": name, "path": path}), timeout=30,
        )
        resp.raise_for_status()
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
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.delete(
                    f"{self.url}/api/{endpoint}/{item_id}/", timeout=15,
                )
                return resp.status_code == 204
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                time.sleep(2 * (attempt + 1))
        return False

    def delete_tag(self, tag_id: int) -> bool:
        return self._delete_item("tags", tag_id)

    def delete_correspondent(self, corr_id: int) -> bool:
        return self._delete_item("correspondents", corr_id)

    def delete_document_type(self, type_id: int) -> bool:
        return self._delete_item("document_types", type_id)

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


# ══════════════════════════════════════════════════════════════════════════════
# LocalLLMAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

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


def _canonicalize_correspondent_name(value: str) -> str:
    normalized = _normalize_correspondent_name(value)
    if not normalized:
        return ""
    # OCR / typo resilience for known patterns
    normalized = normalized.replace("thresien", "dresden")
    normalized = normalized.replace("industriekammer", "industrie- und handelskammer")
    alias = CORRESPONDENT_ALIASES.get(normalized)
    return alias or _normalize_text(value)


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


def build_decision_context(documents: list, correspondents: list, storage_paths: list) -> DecisionContext:
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
    for norm_name, count in work_corr_counts.items():
        if count >= 2:
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
        hint = _fetch_entity_web_hint(candidate)
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

    has_private_vehicle = _contains_any_hint(text, PRIVATE_VEHICLE_HINTS)
    has_company_vehicle = _contains_any_hint(text, COMPANY_VEHICLE_HINTS)

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
    confidence = str(suggestion.get("confidence", "high")).strip().lower()
    if REVIEW_ON_MEDIUM_CONFIDENCE and confidence in ("low", "medium"):
        reasons.append(f"confidence={confidence}")

    if not _normalize_text(suggestion.get("title", "")) and not _normalize_text(document.get("title", "")):
        reasons.append("kein Titel")

    doc_type = _normalize_text(suggestion.get("document_type", ""))
    if not doc_type and not document.get("document_type"):
        reasons.append("kein Dokumenttyp")
    elif doc_type and doc_type.lower() not in {t.lower() for t in ALLOWED_DOC_TYPES} and not document.get("document_type"):
        reasons.append(f"dokumenttyp-unbekannt:{doc_type}")

    if not selected_tags and not document.get("tags"):
        reasons.append("keine gueltigen Tags")

    path_value = _normalize_text(suggestion.get("storage_path", ""))
    if not path_value and not document.get("storage_path"):
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
    """Analysiert Dokumente mit lokalem LLM via LM Studio."""

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
            "qwen2.5:7b",
            "qwen2.5-coder:7b",
            "google/gemma-3-4b",
            "gemma3:latest",
            "llama3.1:8b",
            "mistral:7b",
            "llama3.2:1b",
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
        """Prueft ob LM Studio erreichbar ist."""
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
        log.error(f"LM Studio nicht erreichbar! ({self.url})")
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
                decision_context: DecisionContext | None = None) -> dict:
        """Analysiert ein Dokument und gibt Organisationsvorschlag zurueck."""

        # Top-Korrespondenten nach Dokumentanzahl (Token sparen)
        top_corrs = sorted(existing_correspondents, key=lambda c: c.get("document_count", 0), reverse=True)
        corr_names = [c["name"] for c in top_corrs[:50]]
        if taxonomy and ENFORCE_TAG_TAXONOMY and taxonomy.canonical_tags:
            tag_choices = taxonomy.prompt_tags(MAX_PROMPT_TAG_CHOICES)
        else:
            top_tags = sorted(existing_tags, key=lambda t: t.get("document_count", 0), reverse=True)
            tag_choices = [t["name"] for t in top_tags[:MAX_PROMPT_TAG_CHOICES]]

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

        # Token-effizient: Kurze Dokumente komplett, lange nur Anfang
        content = document.get("content") or ""
        content_len = len(content)
        if content_len > 5000:
            content_preview = content[:1500] + f"\n[...{content_len} Zeichen insgesamt, Rest abgeschnitten...]"
        elif content_len > 2000:
            content_preview = content[:2000]
        else:
            content_preview = content

        brand_hint = _get_brand_hint(f"{document.get('title', '')} {document.get('original_file_name', '')} {content_preview[:1000]}")
        web_hint_primary = _fetch_web_hint(document.get("title", ""), content_preview) if ENABLE_WEB_HINTS else ""
        web_hint_entities = _collect_web_entity_hints(document, current_corr=current_corr)
        web_hint = " | ".join([h for h in [web_hint_primary, web_hint_entities] if h])

        employers_info = ", ".join(sorted(decision_context.employer_names)) if decision_context else "keine"
        providers_info = ", ".join(sorted(decision_context.provider_names)) if decision_context else "keine"
        work_paths_info = ", ".join(decision_context.top_work_paths) if decision_context else "keine"
        private_paths_info = ", ".join(decision_context.top_private_paths) if decision_context else "keine"

        prompt = f"""Paperless-NGX Dokument organisieren. Besitzer: Edgar Richter, Reichenbach/Sachsen.
Job: Systemadministrator bei WBS TRAINING AG (seit Aug 2025). Vorher: Azubi msg systems ag (2022-2025, Fachinformatiker). DRK-Mitglied. AOK PLUS.
Fahrzeugkontext: Privatwagen = VW Polo (auch "Golf Polo"/"Polo"), Firmenwagen = Toyota.

DOKUMENT #{document['id']}:
Titel: {document.get('title', '?')} | Datei: {document.get('original_file_name', '?')} | Erstellt: {document.get('created', '?')}
Tags: {current_tags or 'keine'} | Korr: {current_corr or 'keiner'} | Typ: {current_type or 'keiner'} | Pfad: {current_path or 'keiner'}

INHALT:
{content_preview}

KORRESPONDENTEN (bevorzuge vorhandene): {', '.join(corr_names)}

DOKUMENTTYPEN (NUR diese): {', '.join(ALLOWED_DOC_TYPES)}

SPEICHERPFADE (NUR diese Kategorienamen verwenden, NICHT den vollen Pfad mit Jahr/Titel!):
{', '.join(p['name'] for p in existing_paths if 'Duplikat' not in p['name'])}
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

ZUORDNUNGSREGELN - GENAU BEACHTEN:
- Arbeit/msg: NUR Dokumente die DIREKT von msg systems ag stammen (Arbeitsvertrag, Zeugnis, Gehaltsabrechnung MIT msg im Absender/Inhalt)
- Arbeit/WBS: NUR Dokumente die DIREKT von WBS TRAINING AG stammen (Arbeitsvertrag, Gehaltsabrechnung MIT WBS im Absender/Inhalt)
- NIEMALS Arbeitgeber als Korrespondent setzen, wenn im Dokument klar ein externer Anbieter steht (z.B. Google Cloud, GitHub, JetBrains, OpenAI, ElevenLabs)
- NICHT zu Arbeit: Software-Abos (Claude, GitHub, JetBrains, etc.), Weiterbildungen die privat bezahlt werden, private Cloud-Dienste, Technik-Kaeufe
- Software-Abos, KI-Dienste, Cloud-Dienste, Hosting -> Freizeit/IT oder Finanzen je nach Kontext
- DRK/Wasserwacht/Blutspende -> Persoenlich/DRK oder Ausbildung/DRK
- Versicherungen -> Versicherungen/[Typ]
- Bank/Finanzen/Depot -> Finanzen/Bank oder Finanzen/Depot
- Arzt/Gesundheit/AOK -> Gesundheit
- VW Polo / Polo / Golf Polo = PRIVAT einordnen (nicht Arbeit)
- Toyota = Firmenwagen-Kontext (arbeitsbezogen), nicht automatisch privat einordnen
- Frage dich IMMER: Ist das ein ARBEITSDOKUMENT (direkt vom Arbeitgeber) oder PRIVAT?
- Im Zweifel: PRIVAT einordnen, nicht Arbeit!

WEITERE REGELN: Tags kurz und sinnvoll. Titel deutsch, aussagekraeftig. Korrespondent=Absender/Firma.
confidence: Wie sicher bist du dir bei der Zuordnung? "high", "medium" oder "low".

NUR JSON, kein anderer Text:
{{"title": "Titel", "tags": ["Tag1", "Tag2"], "correspondent": "Firma", "document_type": "Typ", "storage_path_name": "Kategorie", "storage_path": "Kategorie/Sub", "confidence": "high", "reasoning": "Kurz"}}"""

        # LLM-Anfrage
        response = self._call_llm(prompt)

        # JSON parsen - bei Fehler Retry mit strengerem Prompt
        try:
            suggestion = self._parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            log.info("[yellow]JSON-Parse-Fehler, versuche erneut...[/yellow]")
            retry_prompt = (
                f"{prompt}\n\n"
                "WICHTIG: Antworte AUSSCHLIESSLICH mit einem einzigen JSON-Objekt. "
                "Kein Text davor oder danach. Keine Markdown-Formatierung."
            )
            response = self._call_llm(retry_prompt)
            suggestion = self._parse_json_response(response)

        # Bei niedriger Konfidenz: Verifikation mit zweitem LLM-Aufruf
        confidence = suggestion.get("confidence", "high").lower()
        if confidence in ("low", "medium"):
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

        response = self._call_llm(verify_prompt)
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

    def _build_payload(self, prompt: str) -> dict:
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
                    "num_predict": LLM_MAX_TOKENS,
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
            "max_tokens": LLM_MAX_TOKENS,
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

    def _post_with_retry(self, headers: dict | None, payload: dict) -> requests.Response:
        last_resp = None
        for attempt in range(3):
            resp = requests.post(self.url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            last_resp = resp
            if resp.ok:
                return resp
            # Manche lokalen Server liefern kurzzeitig leere 400/5xx bei Last.
            transient_400 = resp.status_code == 400 and not (resp.text or "").strip()
            if resp.status_code in (429, 500, 502, 503, 504) or transient_400:
                time.sleep(0.8 + attempt * 0.6)
                continue
            return resp
        return last_resp

    def _call_llm(self, prompt: str) -> str:
        """Sendet Prompt an LLM-Endpunkt und gibt Antwort zurueck."""
        headers = self._auth_headers()
        payload = self._build_payload(prompt)
        resp = self._post_with_retry(headers, payload)

        # Retry 1: if model handling might be the issue, use server-default once.
        if resp.status_code in (400, 422) and self.model:
            model_backup = self.model
            self.model = ""
            retry_payload = self._build_payload(prompt)
            retry_resp = self._post_with_retry(headers, retry_payload)
            if retry_resp.ok:
                resp = retry_resp
            else:
                self.model = model_backup

        # Retry 2: discover model automatically if model is empty.
        if resp.status_code in (400, 422) and not self.model:
            detected_model = self._discover_model(headers)
            if detected_model:
                self.model = detected_model
                retry_payload = self._build_payload(prompt)
                retry_resp = self._post_with_retry(headers, retry_payload)
                if retry_resp.ok:
                    resp = retry_resp

        if not resp.ok:
            detail = self._error_snippet(resp)
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for url: {self.url} | body: {detail}",
                response=resp,
            )
        return self._extract_response_text(resp.json())


# ══════════════════════════════════════════════════════════════════════════════
# ID-Aufloesung & Anzeige (aus main.py)
# ══════════════════════════════════════════════════════════════════════════════

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
    type_id = None
    type_name = suggestion.get("document_type", "")
    if type_name:
        allowed_lower = {t.lower(): t for t in ALLOWED_DOC_TYPES}
        key = type_name.lower()
        if key not in allowed_lower:
            console.print(f"  [red]Dokumenttyp '{type_name}' nicht erlaubt, uebersprungen.[/red]")
        elif key in type_map:
            type_id = type_map[key]
        else:
            canonical_name = allowed_lower[key]
            try:
                console.print(f"  [yellow]+ Neuer Dokumenttyp:[/yellow] {canonical_name}")
                new = paperless.create_document_type(canonical_name)
                type_id = new["id"]
                doc_types.append(new)
            except requests.exceptions.HTTPError:
                fresh = paperless.get_document_types()
                type_map = {t["name"].lower(): t["id"] for t in fresh}
                type_id = type_map.get(key)
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

    table = Table(title=f"Dokument #{document['id']}", show_header=True, width=80)
    table.add_column("Feld", style="cyan", width=18)
    table.add_column("Aktuell", style="red", width=28)
    table.add_column("Vorschlag", style="green", width=30)

    table.add_row("Titel", document.get("title", ""), suggestion.get("title", ""))
    table.add_row("Tags", ", ".join(current_tags) or "Keine", ", ".join(suggestion.get("tags", [])))
    table.add_row("Korrespondent", current_corr, suggestion.get("correspondent", ""))
    table.add_row("Dokumenttyp", current_type, suggestion.get("document_type", ""))
    table.add_row("Speicherpfad", current_path, suggestion.get("storage_path", ""))
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

    console.print(table)
    if suggestion.get("reasoning"):
        console.print(Panel(suggestion["reasoning"], title="Begruendung", border_style="blue"))


# ══════════════════════════════════════════════════════════════════════════════
# Dokument-Verarbeitung
# ══════════════════════════════════════════════════════════════════════════════

def process_document(doc_id: int, paperless: PaperlessClient,
                     analyzer: LocalLLMAnalyzer, tags: list,
                     correspondents: list, doc_types: list,
                     storage_paths: list, dry_run: bool = True,
                     batch_mode: bool = False,
                     taxonomy: TagTaxonomy | None = None,
                     decision_context: DecisionContext | None = None,
                     run_db: LocalStateDB | None = None,
                     run_id: int | None = None) -> bool:
    """Einzelnes Dokument analysieren und organisieren."""

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

    log.info(f"  LLM-Analyse laeuft... (Modell: {analyzer.model})")
    t_start = time.perf_counter()
    try:
        suggestion = analyzer.analyze(
            document,
            tags,
            correspondents,
            doc_types,
            storage_paths,
            taxonomy=taxonomy,
            decision_context=decision_context,
        )
    except json.JSONDecodeError as e:
        log.error(f"  JSON-Parse-Fehler: {e}")
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "error_json", document, error=str(e))
        return False
    except requests.exceptions.Timeout:
        log.error(f"  Timeout: LLM hat zu lange gebraucht.")
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "error_timeout", document, error="llm timeout")
        return False
    except requests.exceptions.ConnectionError:
        log.error(f"  LLM nicht erreichbar! Ist LM Studio gestartet?")
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "error_llm_connection", document, error="llm connection failed")
        return False
    except Exception as e:
        log.error(f"  Fehler: {e}")
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "error_generic", document, error=str(e))
        return False

    t_elapsed = time.perf_counter() - t_start
    log.info(f"  LLM-Antwort erhalten ({t_elapsed:.1f}s) -> Titel: [green]{suggestion.get('title', '?')}[/green]")
    _sanitize_suggestion_spelling(suggestion)

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
        log.info(f"[bold green]FERTIG[/bold green] Dokument #{doc_id} aktualisiert -> {result.get('archived_file_name', '?')}")
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "updated", document, suggestion=suggestion)
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
                      taxonomy: TagTaxonomy | None = None,
                      decision_context: DecisionContext | None = None,
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

    if not todo:
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
        for i, doc in enumerate(todo, 1):
            log.info(f"[bold]--- {i}/{len(todo)} --- Dokument #{doc['id']}[/bold]")
            try:
                if process_document(doc["id"], paperless, analyzer, tags, correspondents,
                                    doc_types, storage_paths, dry_run, batch_mode=not dry_run,
                                    taxonomy=taxonomy,
                                    decision_context=decision_context,
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
                    taxonomy,
                    decision_context,
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
    log.info(f"[bold]AUTO-SORTIERUNG FERTIG[/bold] - {applied}/{len(todo)} aktualisiert, "
             f"{errors} Fehler, {batch_elapsed:.1f}s gesamt")
    if AUTO_CLEANUP_AFTER_ORGANIZE and not dry_run:
        log.info("[bold]AUTO-CLEANUP[/bold] nach Auto-Sortierung gestartet")
        cleanup_tags(paperless, dry_run=False)
        cleanup_correspondents(paperless, dry_run=False)
        cleanup_document_types(paperless, dry_run=False)
    return {"total": total, "todo": len(todo), "applied": applied, "errors": errors}


def batch_process(paperless: PaperlessClient, analyzer: LocalLLMAnalyzer,
                  tags: list, correspondents: list, doc_types: list,
                  storage_paths: list, dry_run: bool, mode: str = "untagged",
                  limit: int = 0,
                  taxonomy: TagTaxonomy | None = None,
                  decision_context: DecisionContext | None = None,
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

    # Limit (0 = alle)
    if limit > 0 and len(documents) > limit:
        documents = documents[:limit]

    log.info(f"  Verarbeite {len(documents)} Dokumente")

    if not documents:
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
                    taxonomy,
                    decision_context,
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


# ══════════════════════════════════════════════════════════════════════════════
# Cleanup-Funktionen
# ══════════════════════════════════════════════════════════════════════════════

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

        reason = ""
        # Regel 1: Nicht-Taxonomie Tags aggressiver abbauen.
        if not in_taxonomy and doc_count <= NON_TAXONOMY_DELETE_THRESHOLD:
            reason = f"nicht in Taxonomie, <= {NON_TAXONOMY_DELETE_THRESHOLD} Dokumente"
        # Regel 2: allgemeiner Low-Count-Cleanup.
        elif doc_count <= TAG_DELETE_THRESHOLD:
            reason = f"<= {TAG_DELETE_THRESHOLD} Dokumente"
        # Regel 3: ASCII/Englisch nur ausserhalb Taxonomie.
        elif not in_taxonomy and doc_count <= TAG_ENGLISH_THRESHOLD and _is_ascii_only(name):
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


def _find_correspondent_groups(correspondents: list) -> dict:
    """Gruppiert Korrespondenten nach definierten Gruppen."""
    groups = defaultdict(list)
    for c in correspondents:
        name_lower = c["name"].lower()
        for group_name, keywords in CORRESPONDENT_GROUPS.items():
            for kw in keywords:
                if kw.lower() in name_lower:
                    groups[group_name].append(c)
                    break
    return groups


def cleanup_correspondents(paperless: PaperlessClient, dry_run: bool = True):
    """Leere loeschen + Gruppen-Merge."""
    console.print(Panel("[bold]Korrespondenten-Aufraeumung[/bold]", border_style="cyan"))
    log.info("[bold]CLEANUP KORRESPONDENTEN[/bold] gestartet")

    correspondents = paperless.get_correspondents()
    log.info(f"  {len(correspondents)} Korrespondenten geladen")

    # Leere loeschen
    unused = [c for c in correspondents if c.get("document_count", 0) == 0]
    console.print(f"[yellow]Ungenutzt (0 Dokumente): {len(unused)}[/yellow]")

    if unused and not dry_run:
        deleted = paperless.batch_delete("correspondents", unused, "Korrespondenten")
        console.print(f"[green]{deleted} ungenutzte Korrespondenten geloescht[/green]")

    # Duplikat-Gruppen finden
    groups = _find_correspondent_groups(correspondents)
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}

    if duplicates:
        console.print(f"\n[bold]{len(duplicates)} Duplikat-Gruppen gefunden:[/bold]")
        for group_name, items in sorted(duplicates.items()):
            total = sum(c.get("document_count", 0) for c in items)
            items.sort(key=lambda x: x.get("document_count", 0), reverse=True)
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


# ══════════════════════════════════════════════════════════════════════════════
# Duplikate finden
# ══════════════════════════════════════════════════════════════════════════════

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

    # Nach Dateiname gruppieren
    by_filename = defaultdict(list)
    for d in docs:
        fname = d.get("original_file_name", "").strip()
        if fname:
            by_filename[fname].append(d)
    filename_dupes = {k: v for k, v in by_filename.items() if len(v) > 1}

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

    # Zusammenfassung
    console.print(f"\n[bold]Zusammenfassung:[/bold]")
    console.print(f"  Dateiname-Duplikate: {len(filename_dupes)} Gruppen ({sum(len(v) for v in filename_dupes.values())} Dokumente)")
    console.print(f"  Titel-Duplikate: {len(title_dupes)} Gruppen ({sum(len(v) for v in title_dupes.values())} Dokumente)")
    console.print("[red]Es wurden KEINE Dokumente geloescht![/red]")
    log.info(f"  DUPLIKAT-SCAN fertig - {len(filename_dupes)} Dateiname-Gruppen, {len(title_dupes)} Titel-Gruppen")


# ══════════════════════════════════════════════════════════════════════════════
# Statistiken
# ══════════════════════════════════════════════════════════════════════════════

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

    table2.add_row("Ohne Tags", str(no_tags))
    table2.add_row("Ohne Korrespondent", str(no_corr))
    table2.add_row("Ohne Dokumenttyp", str(no_type))
    table2.add_row("Ohne Speicherpfad", str(no_path))
    table2.add_row("Nicht vollstaendig", str(incomplete))
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


# ══════════════════════════════════════════════════════════════════════════════
# Menuesystem
# ══════════════════════════════════════════════════════════════════════════════

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

    def _start_run(self, action: str) -> int:
        run_id = self.run_db.start_run(action, self.dry_run, self.llm_model, self.llm_url)
        log.info(f"Run gestartet: #{run_id} ({action})")
        return run_id

    def _finish_run(self, run_id: int, summary: dict):
        self.run_db.finish_run(run_id, summary)
        log.info(f"Run abgeschlossen: #{run_id}")

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
        context = build_decision_context(documents, correspondents, storage_paths)
        log.info(
            "KONTEXT gesammelt: %s Dokumente | %s Arbeitgeber-Hints | %s Anbieter-Hints",
            len(documents),
            len(context.employer_names),
            len(context.provider_names),
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
            f"State-DB: {STATE_DB_FILE}",
            border_style="blue",
        ))

    def _menu(self, title: str, options: list) -> str:
        """Menue anzeigen und Auswahl zurueckgeben."""
        console.print(f"\n[bold]{title}[/bold]")
        for key, label in options:
            console.print(f"  [cyan]{key}[/cyan]  {label}")
        return Prompt.ask("\nAuswahl", default="0")

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
                ("0", "Beenden"),
            ])

            if choice == "1":
                self.action_auto_organize()
            elif choice == "2":
                self.menu_organize()
            elif choice == "3":
                self.menu_cleanup()
            elif choice == "4":
                self.action_find_duplicates()
            elif choice == "5":
                self.action_statistics()
            elif choice == "6":
                self.action_review_queue()
            elif choice == "7":
                self.menu_settings()
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

        tags, correspondents, doc_types, storage_paths = self._load_master_data()
        self._ensure_taxonomy_tags(tags)
        decision_context = self._collect_decision_context(correspondents, storage_paths)
        run_id = self._start_run("auto_organize")
        summary = auto_organize_all(
            self.paperless,
            self.analyzer,
            tags,
            correspondents,
            doc_types,
            storage_paths,
            self.dry_run,
            force_recheck_all=RECHECK_ALL_DOCS_IN_AUTO,
            taxonomy=self.taxonomy,
            decision_context=decision_context,
            run_db=self.run_db,
            run_id=run_id,
        )
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

    def action_find_duplicates(self):
        """Duplikate finden."""
        if not self._init_paperless():
            return
        find_duplicates(self.paperless)

    def action_statistics(self):
        """Statistiken anzeigen."""
        if not self._init_paperless():
            return
        show_statistics(self.paperless)

    def action_review_queue(self):
        """Offene menschliche Nachpruefungen anzeigen."""
        open_items = self.run_db.list_open_reviews(limit=100)
        table = Table(title="Review-Queue (offen)", show_header=True)
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
            if self.run_db.close_review(review_id):
                console.print("[green]Review-Eintrag geschlossen.[/green]")
            else:
                console.print("[yellow]Kein offener Eintrag mit dieser ID gefunden.[/yellow]")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 40)
    log.info("[bold]Paperless-NGX Organizer gestartet[/bold]")
    log.info(f"  Log-Datei: {LOG_FILE}")
    log.info(f"  State-DB: {STATE_DB_FILE}")
    log.info("=" * 40)
    app = App()
    try:
        app.menu_main()
    except KeyboardInterrupt:
        console.print("\n[bold]Abgebrochen.[/bold]")
    finally:
        log.info("Paperless-NGX Organizer beendet")


if __name__ == "__main__":
    main()
