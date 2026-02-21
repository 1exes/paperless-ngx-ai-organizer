"""Web search hints via DuckDuckGo for unknown brands/entities."""

from __future__ import annotations

from urllib.parse import urlencode

import requests

from .config import (
    ENABLE_WEB_HINTS,
    WEB_HINT_CACHE,
    WEB_HINT_CACHE_LOCK,
    WEB_HINT_MAX_ENTITIES,
    WEB_HINT_TIMEOUT,
    log,
)
from .constants import KNOWN_BRAND_HINTS
from .utils import (
    _extract_document_entities,
    _extract_keywords,
    _normalize_tag_name,
    _normalize_text,
)


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
    except Exception as exc:
        log.debug("Web-Hint Fehler fuer '%s': %s", query_text[:60], exc)
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
    except Exception as exc:
        log.debug("Entity-Web-Hint Fehler fuer '%s': %s", entity_clean[:40], exc)
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
        result = _fetch_entity_web_hint(entity)
    except Exception as exc:
        log.debug("Websuche Fehler fuer '%s': %s", entity_clean[:40], exc)
        try:
            result = _fetch_entity_web_hint(entity)
        except Exception as exc2:
            log.debug("Entity-Fallback Fehler fuer '%s': %s", entity_clean[:40], exc2)
            result = ""
    with WEB_HINT_CACHE_LOCK:
        WEB_HINT_CACHE[cache_key] = result
    return result


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
