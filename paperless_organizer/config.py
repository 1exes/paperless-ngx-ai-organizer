"""Global configuration, logging setup, and shared state."""

from __future__ import annotations

import json as _json
import os
import logging
import logging.handlers
import re
import threading
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

load_dotenv()
console = Console()

# --- File Paths ---
LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(LOG_DIR, "organizer.log")
LOG_FILE_JSON = os.path.join(LOG_DIR, "organizer.jsonl")
STATE_DB_FILE = os.path.join(LOG_DIR, "organizer_state.db")
TAXONOMY_FILE = os.path.join(LOG_DIR, "taxonomy_tags.json")
LEARNING_PROFILE_FILE = os.path.join(LOG_DIR, "learning_profile.json")
LEARNING_EXAMPLES_FILE = os.path.join(LOG_DIR, "learning_examples.jsonl")

# --- Structured JSON Logging ---
STRUCTURED_LOG_ENABLED = os.getenv("STRUCTURED_LOG", "1").strip().lower() in ("1", "true", "yes", "on")

_RICH_MARKUP_RE = re.compile(r"\[/?[a-z_]+(?:\s[^\]]+)?\]")


class _JsonLineFormatter(logging.Formatter):
    """Formats log records as single-line JSON (JSONL) for monitoring tools."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        # Strip Rich markup tags like [bold], [cyan], [/cyan] etc.
        msg = _RICH_MARKUP_RE.sub("", msg)
        entry: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": msg,
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        # Attach extra structured fields if present (e.g. doc_id, action)
        for key in ("doc_id", "action", "run_id", "duration_ms", "confidence", "status"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return _json.dumps(entry, ensure_ascii=False, default=str)


# --- Logging ---
_LOG_LEVEL_MAP = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
_LOG_LEVEL = _LOG_LEVEL_MAP.get(os.getenv("LOG_LEVEL", "INFO").strip().upper(), logging.INFO)

# Plain-text file handler (always active)
_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

_handlers: list[logging.Handler] = [
    RichHandler(console=console, show_path=False, markup=True, rich_tracebacks=True),
    _file_handler,
]

# Structured JSON file handler (opt-in via STRUCTURED_LOG=1)
if STRUCTURED_LOG_ENABLED:
    _json_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE_JSON, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    _json_handler.setFormatter(_JsonLineFormatter())
    _handlers.append(_json_handler)

logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=_handlers,
)
log = logging.getLogger("organizer")

# --- Version ---
__version__ = "3.0.0"

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
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "").strip()
LLM_FALLBACK_AFTER_ERRORS = int(os.getenv("LLM_FALLBACK_AFTER_ERRORS", "3"))

# --- LLM Speedcheck ---
LLM_SPEEDCHECK_ENABLED = os.getenv("LLM_SPEEDCHECK_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
LLM_SPEEDCHECK_MAX_TIME = float(os.getenv("LLM_SPEEDCHECK_MAX_TIME", "10.0"))
LLM_SPEEDCHECK_TIMEOUT = int(os.getenv("LLM_SPEEDCHECK_TIMEOUT", "15"))
LLM_SPEEDCHECK_AUTO_SWITCH = os.getenv("LLM_SPEEDCHECK_AUTO_SWITCH", "1").strip().lower() in ("1", "true", "yes", "on")
LLM_SPEEDCHECK_INTERVAL_CYCLES = int(os.getenv("LLM_SPEEDCHECK_INTERVAL_CYCLES", "0"))

# --- Paperless-NGX ---
OWNER_ID = int(os.getenv("OWNER_ID", "4"))
ORGANIZER_OWNER_NAME = os.getenv("ORGANIZER_OWNER_NAME", "Document Owner").strip()

# --- Tag-Loeschregeln ---
TAG_DELETE_THRESHOLD = int(os.getenv("TAG_DELETE_THRESHOLD", "0"))
TAG_ENGLISH_THRESHOLD = int(os.getenv("TAG_ENGLISH_THRESHOLD", "0"))
NON_TAXONOMY_DELETE_THRESHOLD = int(os.getenv("NON_TAXONOMY_DELETE_THRESHOLD", "5"))
DELETE_USED_TAGS = os.getenv("DELETE_USED_TAGS", "0").strip().lower() in ("1", "true", "yes", "on")

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

# --- Quiet Hours ---
QUIET_HOURS_START = int(os.getenv("QUIET_HOURS_START", "0"))
QUIET_HOURS_END = int(os.getenv("QUIET_HOURS_END", "0"))

# --- Web Hints ---
ENABLE_WEB_HINTS = os.getenv("ENABLE_WEB_HINTS", "0").strip().lower() in ("1", "true", "yes", "on")
WEB_HINT_TIMEOUT = int(os.getenv("WEB_HINT_TIMEOUT", "6"))
WEB_HINT_MAX_ENTITIES = int(os.getenv("WEB_HINT_MAX_ENTITIES", "2"))
WEB_HINT_CACHE: dict[str, str] = {}
WEB_HINT_CACHE_LOCK = threading.Lock()

# --- Rule-based ---
RULE_BASED_MIN_SAMPLES = int(os.getenv("RULE_BASED_MIN_SAMPLES", "10"))
RULE_BASED_MIN_RATIO = float(os.getenv("RULE_BASED_MIN_RATIO", "0.80"))


def _is_quiet_hours() -> bool:
    """Check if current time is within configured quiet hours."""
    from datetime import datetime
    if QUIET_HOURS_START == 0 and QUIET_HOURS_END == 0:
        return False
    hour = datetime.now().hour
    if QUIET_HOURS_START < QUIET_HOURS_END:
        return QUIET_HOURS_START <= hour < QUIET_HOURS_END
    else:
        return hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END


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
