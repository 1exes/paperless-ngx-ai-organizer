"""Utility functions: normalization, extraction, fingerprinting, spelling, robustness."""

from __future__ import annotations

import difflib
import functools
import json
import logging
import random
import re
import time
import unicodedata
from collections import defaultdict
from datetime import datetime

from .config import (
    CORRESPONDENT_MERGE_MIN_NAME_SIMILARITY,
    MAX_PROMPT_TAG_CHOICES,
    USE_ARCHIVE_SERIAL_NUMBER,
)

_log = logging.getLogger("organizer")


# ---------------------------------------------------------------------------
# Robustness utilities
# ---------------------------------------------------------------------------

def safe_parse_json(text: str, fallback: dict | list | None = None) -> dict | list | None:
    """Parse JSON safely with fallback. Strips markdown code fences if present."""
    if not text or not text.strip():
        return fallback
    cleaned = text.strip()
    # Strip markdown code fences (```json ... ```)
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object/array in the text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = cleaned.find(start_char)
            end = cleaned.rfind(end_char)
            if start != -1 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    continue
        _log.debug("JSON-Parsing fehlgeschlagen: %s", cleaned[:120])
        return fallback


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, OSError),
):
    """Decorator: retries a function with exponential backoff on transient errors."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, 1))
                        _log.debug("Retry %d/%d fuer %s nach %.1fs: %s",
                                   attempt + 1, max_retries, func.__name__, delay, exc)
                        time.sleep(delay)
                    else:
                        raise
            if last_exc:
                raise last_exc
        return wrapper
    return decorator
from .constants import (
    CORRESPONDENT_ALIASES,
    CORRESPONDENT_LEGAL_TOKENS,
    CORRESPONDENT_STOPWORDS,
    GERMAN_MONTHS,
    KNOWN_BLZ,
    KNOWN_BRAND_HINTS,
    SPELLING_FIXES,
    TITLE_SPELLING_FIXES,
)


# ---------------------------------------------------------------------------
# Lookup helpers (avoid O(n) linear scans)
# ---------------------------------------------------------------------------

def _build_id_name_map(items: list) -> dict[int, str]:
    """Build {id: name} lookup dict from a list of Paperless-NGX objects."""
    return {int(item["id"]): str(item.get("name", "")) for item in items if item.get("id") is not None}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

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
    normalized = normalized.replace("thresien", "dresden")
    normalized = normalized.replace("industriekammer", "industrie- und handelskammer")
    alias = CORRESPONDENT_ALIASES.get(normalized)
    if alias:
        return alias
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


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

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


def _extract_document_date(document: dict) -> str:
    """Extract the most likely document date from content (German formats)."""
    content = str(document.get("content") or "")
    if not content:
        return ""
    text = content[:1500]

    dates_dot = re.findall(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", text)
    dates_named = re.findall(r"\b(\d{1,2})\.\s*([A-Za-z\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc]+)\s+(\d{4})\b", text)
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
        month_num = GERMAN_MONTHS.get(month_name.lower())
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
    best = max(candidates)
    return best.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> str:
    """Detect document language ('de' or 'en') based on word frequency."""
    from .constants import ENGLISH_MARKERS, GERMAN_MARKERS
    words = set(re.findall(r"\b[a-z]{3,}\b", text[:2000].lower()))
    de_count = len(words & GERMAN_MARKERS)
    en_count = len(words & ENGLISH_MARKERS)
    return "de" if de_count >= en_count else "en"


# ---------------------------------------------------------------------------
# OCR quality assessment
# ---------------------------------------------------------------------------

def _assess_ocr_quality(document: dict) -> tuple[str, float]:
    """Assess OCR quality. Returns (quality_level, score)."""
    content = str(document.get("content") or "")
    if not content:
        return ("poor", 0.0)

    total_chars = len(content)
    if total_chars < 50:
        return ("poor", 0.1)

    alpha_chars = sum(1 for c in content if c.isalpha())
    digit_chars = sum(1 for c in content if c.isdigit())
    space_chars = sum(1 for c in content if c.isspace())
    special_chars = total_chars - alpha_chars - digit_chars - space_chars

    words = content.split()
    total_words = len(words)
    if total_words < 5:
        return ("poor", 0.15)

    alpha_ratio = alpha_chars / max(1, total_chars)
    special_ratio = special_chars / max(1, total_chars)
    avg_word_len = sum(len(w) for w in words) / max(1, total_words)
    real_words = sum(1 for w in words if len(w) >= 3 and sum(1 for c in w if c.isalpha()) >= 2)
    real_word_ratio = real_words / max(1, total_words)

    score = 0.0
    score += min(0.3, alpha_ratio * 0.4)
    score += min(0.2, (1.0 - special_ratio) * 0.25)
    score += min(0.2, real_word_ratio * 0.25)
    score += 0.15 if 3.0 <= avg_word_len <= 12.0 else 0.0
    score += 0.15 if total_words >= 20 else (0.07 if total_words >= 10 else 0.0)

    if score >= 0.7:
        return ("good", score)
    elif score >= 0.4:
        return ("medium", score)
    else:
        return ("poor", score)


# ---------------------------------------------------------------------------
# Keyword / brand extraction
# ---------------------------------------------------------------------------

def _extract_keywords(text: str, limit: int = 6) -> list:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", text or "")
    skip = {"und", "oder", "mit", "fuer", "der", "die", "das", "invoice", "document"}
    counts: dict[str, int] = {}
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


# ---------------------------------------------------------------------------
# Invoice / amount extraction
# ---------------------------------------------------------------------------

def _extract_invoice_number(content: str) -> str:
    """Extract invoice/reference number from content."""
    patterns = [
        r"(?:Rechnungsnr|Rechnungsnummer|Rechnung\s*Nr|Invoice\s*No|Invoice\s*#|Belegnr|Beleg-Nr|Vorgangsnr)\.?\s*[:\s]?\s*([A-Z0-9][\w\-/]{3,20})",
        r"(?:Kundennr|Kunden-Nr|Vertragsnr|Vertrags-Nr|Aktenzeichen|Az)\.?\s*[:\s]?\s*([A-Z0-9][\w\-/]{3,20})",
    ]
    for pat in patterns:
        m = re.search(pat, content[:2000], re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _extract_amount(content: str) -> str:
    """Extract a monetary amount from content (German format)."""
    patterns = [
        r"(?:Gesamtbetrag|Rechnungsbetrag|Summe|Total|Betrag|Endbetrag)\s*[:\s]?\s*(\d{1,3}(?:\.\d{3})*,\d{2})\s*(?:EUR|\u20ac)?",
        r"(?:EUR|\u20ac)\s*(\d{1,3}(?:\.\d{3})*,\d{2})",
        r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*(?:EUR|\u20ac)",
    ]
    for pat in patterns:
        m = re.search(pat, content[:3000], re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Title improvement
# ---------------------------------------------------------------------------

def _improve_title(title: str, document: dict) -> str:
    """Clean up and improve LLM-generated titles."""
    if not title:
        return title
    title = re.sub(r"\.(pdf|png|jpg|jpeg|tiff?|docx?|xlsx?|csv)$", "", title, flags=re.IGNORECASE).strip()
    title = title.strip("'\"")
    title = re.sub(r"^(Dokument|Document|Datei|File|Titel|Betreff|Subject)\s*[:|-]\s*", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"[|/\\]{2,}", " ", title)
    title = re.sub(r"\s{2,}", " ", title).strip()
    words = title.split()
    deduped = []
    for w in words:
        if not deduped or w.lower() != deduped[-1].lower():
            deduped.append(w)
    title = " ".join(deduped)
    half = len(title) // 2
    if half > 5:
        first_half = title[:half].strip()
        second_half = title[half:].strip()
        if first_half.lower() == second_half.lower():
            title = first_half
    title = title.rstrip("-. ")
    if title and title[0].islower():
        title = title[0].upper() + title[1:]

    content = document.get("content") or ""
    title_lower = title.lower()
    if content and len(title) < 60:
        if any(kw in title_lower for kw in ("rechnung", "invoice", "beleg", "quittung")):
            inv_nr = _extract_invoice_number(content)
            if inv_nr and inv_nr.lower() not in title_lower:
                title = f"{title} Nr. {inv_nr}"
        if any(kw in title_lower for kw in ("rechnung", "mahnung", "gutschrift", "kontoauszug")):
            amount = _extract_amount(content)
            if amount and amount not in title:
                title = f"{title} ({amount} EUR)"

    if len(title) > 128:
        title = title[:125] + "..."
    return title


# ---------------------------------------------------------------------------
# Content fingerprinting (MinHash / LSH)
# ---------------------------------------------------------------------------

def _content_fingerprint(content: str) -> set[int]:
    """Create a set of trigram hashes for content similarity comparison."""
    if not content:
        return set()
    text = re.sub(r"[^a-z0-9\u00e4\u00f6\u00fc\u00df\s]", " ", content.lower())
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) < 5:
        return set()
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


def _minhash_signature(trigram_set: set[int], num_hashes: int = 128) -> tuple[int, ...]:
    """Compute a MinHash signature for LSH-based near-duplicate detection."""
    if not trigram_set:
        return ()
    seeds = tuple(range(1, num_hashes + 1))
    sig = []
    for seed in seeds:
        min_val = min((h ^ (seed * 0x9E3779B9)) & 0xFFFFFFFF for h in trigram_set)
        sig.append(min_val)
    return tuple(sig)


def _lsh_find_candidates(signatures: list[tuple[int, ...]], num_bands: int = 16) -> list[tuple[int, int]]:
    """Locality Sensitive Hashing: find candidate pairs with high similarity."""
    if not signatures:
        return []
    sig_len = len(signatures[0])
    rows_per_band = max(1, sig_len // num_bands)
    candidates: set[tuple[int, int]] = set()

    for band_idx in range(num_bands):
        start = band_idx * rows_per_band
        end = min(start + rows_per_band, sig_len)
        buckets: dict[int, list[int]] = defaultdict(list)
        for doc_idx, sig in enumerate(signatures):
            band_hash = hash(sig[start:end])
            buckets[band_hash].append(doc_idx)
        for bucket in buckets.values():
            if len(bucket) > 1 and len(bucket) <= 50:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        candidates.add((bucket[i], bucket[j]))
    return sorted(candidates)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def _extract_document_entities(document: dict) -> list[str]:
    """Extract potential entity names from document text (email domains, URLs, IBAN banks)."""
    content = str(document.get("content") or "")
    title = str(document.get("title") or "")
    filename = str(document.get("original_file_name") or "")
    text_parts = title + " " + filename + " " + content[:500] + " " + content[-500:] if len(content) > 500 else title + " " + filename + " " + content

    entities: list[str] = []
    seen_norm: set[str] = set()

    def _add_entity(name: str):
        norm = _normalize_tag_name(name)
        if norm and norm not in seen_norm and len(name) > 2:
            seen_norm.add(norm)
            entities.append(name)

    for match in re.finditer(r"@([a-z0-9.-]+\.[a-z]{2,})", text_parts, re.IGNORECASE):
        domain = match.group(1).lower()
        if domain.split(".")[-2] in ("gmail", "gmx", "yahoo", "outlook", "hotmail", "web", "t-online", "posteo", "protonmail"):
            continue
        company = domain.rsplit(".", 1)[0].replace(".", " ").replace("-", " ").strip()
        if company:
            _add_entity(company)

    for match in re.finditer(r"https?://(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})", text_parts, re.IGNORECASE):
        domain = match.group(1).lower()
        company = domain.rsplit(".", 1)[0].replace(".", " ").replace("-", " ").strip()
        if company and len(company) > 2:
            _add_entity(company)

    for match in re.finditer(r"\b([A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{0,4}\s?\d{0,2})\b", text_parts):
        iban = match.group(1).replace(" ", "")
        if len(iban) >= 16 and iban[:2] == "DE" and len(iban) >= 12:
            blz = iban[4:12]
            bank_name = KNOWN_BLZ.get(blz)
            if bank_name:
                _add_entity(bank_name)
            else:
                _add_entity(f"Bank BLZ {blz}")
            break

    return entities[:4]


# ---------------------------------------------------------------------------
# Correspondent deduplication helpers
# ---------------------------------------------------------------------------

def _strip_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _correspondent_core_name(value: str) -> str:
    raw = _strip_diacritics(_normalize_correspondent_name(value).lower())
    if not raw:
        return ""
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

    ratio = difflib.SequenceMatcher(None, core_a, core_b).ratio()
    return ratio >= CORRESPONDENT_MERGE_MIN_NAME_SIMILARITY and overlap >= 2


# ---------------------------------------------------------------------------
# Document organization check
# ---------------------------------------------------------------------------

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
