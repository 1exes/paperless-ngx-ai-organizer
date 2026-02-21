"""Guardrails: vendor, vehicle, topic, learning, tag selection, path resolution, review."""

from __future__ import annotations

import difflib
import re

import requests

from . import config as _cfg
from .config import (
    ALLOW_NEW_STORAGE_PATHS,
    AUTO_APPLY_REVIEW_TAG,
    CORRESPONDENT_MATCH_THRESHOLD,
    ENABLE_LEARNING_PRIORS,
    ENFORCE_TAG_TAXONOMY,
    LEARNING_PRIOR_ENABLE_TAG_SUGGESTION,
    LEARNING_PRIOR_MIN_RATIO,
    LEARNING_PRIOR_MIN_SAMPLES,
    MAX_TAGS_PER_DOC,
    REVIEW_ON_MEDIUM_CONFIDENCE,
    REVIEW_TAG_NAME,
    RULE_BASED_MIN_RATIO,
    RULE_BASED_MIN_SAMPLES,
    TAG_MATCH_THRESHOLD,
    WORK_CORR_EMPLOYER_MIN_DOCS,
    console,
    log,
)
from .constants import (
    ALLOWED_DOC_TYPES,
    COMPANY_VEHICLE_HINTS,
    EMPLOYER_HINTS,
    EVENT_TICKET_HINTS,
    HEALTH_HINTS,
    PRIVATE_VEHICLE_HINTS,
    SCHOOL_HINTS,
    TRANSPORT_TICKET_HINTS,
    VENDOR_GUARDRAILS,
)
from .models import DecisionContext
from .utils import (
    _assess_ocr_quality,
    _build_id_name_map,
    _canonicalize_correspondent_name,
    _normalize_correspondent_name,
    _normalize_tag_name,
    _normalize_text,
)
from .web_hints import _fetch_entity_web_hint


# ---------------------------------------------------------------------------
# Decision context
# ---------------------------------------------------------------------------

def build_decision_context(documents: list, correspondents: list, storage_paths: list,
                           learning_profile=None) -> DecisionContext:
    """Collect current system data before making document decisions."""
    from collections import defaultdict
    context = DecisionContext()
    corr_by_id = _build_id_name_map(correspondents)
    path_by_id = _build_id_name_map(storage_paths)

    work_corr_counts: dict[str, int] = defaultdict(int)
    provider_counts: dict[str, int] = defaultdict(int)
    work_path_counts: dict[str, int] = defaultdict(int)
    private_path_counts: dict[str, int] = defaultdict(int)

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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _find_vendor_key(text: str) -> str:
    haystack = (text or "").lower()
    for key in VENDOR_GUARDRAILS:
        if key in haystack:
            return key
    return ""


def _contains_any_hint(text: str, hints: list[str]) -> bool:
    haystack = (text or "").lower()
    return any(hint in haystack for hint in hints)


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


def _get_correspondent_name_by_id(correspondents: list, corr_id: int | None, _lookup: dict[int, str] | None = None) -> str:
    if not corr_id:
        return ""
    if _lookup is not None:
        return _lookup.get(int(corr_id), "")
    return _build_id_name_map(correspondents).get(int(corr_id), "")


def _resolve_correspondent_from_name(correspondents: list, corr_name: str) -> tuple[int | None, str]:
    """Resolve correspondent by canonical alias + fuzzy match."""
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


# ---------------------------------------------------------------------------
# Vendor guardrails
# ---------------------------------------------------------------------------

def _apply_vendor_guardrails(document: dict, suggestion: dict, correspondents: list, storage_paths: list,
                             decision_context: DecisionContext | None = None,
                             _corr_lookup: dict[int, str] | None = None) -> list[str]:
    """Auto-correct common provider/employer misclassifications."""
    corrections = []
    if _corr_lookup is None:
        _corr_lookup = _build_id_name_map(correspondents)
    current_corr_name = _get_correspondent_name_by_id(correspondents, document.get("correspondent"), _lookup=_corr_lookup)
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

    if suggested_corr in employer_hints and vendor_key not in employer_hints:
        suggestion["correspondent"] = guard["correspondent"]
        corrections.append(f"correspondent->{guard['correspondent']} (provider conflict)")
    elif not suggested_corr and current_corr in employer_hints:
        suggestion["correspondent"] = guard["correspondent"]
        corrections.append(f"correspondent->{guard['correspondent']} (empty+provider)")
    elif current_corr and vendor_key in current_corr and suggested_corr in employer_hints:
        suggestion["correspondent"] = current_corr_name
        corrections.append(f"correspondent->{current_corr_name} (keep current provider)")

    storage_path_value = _normalize_text(str(suggestion.get("storage_path", "")))
    if storage_path_value.lower().startswith("arbeit/"):
        safe_path = _pick_existing_storage_path(storage_paths, guard["path_preferences"])
        if safe_path:
            suggestion["storage_path"] = safe_path
            corrections.append(f"storage_path->{safe_path} (provider conflict)")

    web_provider = _fetch_entity_web_hint(guard["correspondent"])
    if web_provider:
        suggestion["reasoning"] = f"{suggestion.get('reasoning', '')} | Web: {web_provider}".strip(" |")
    return corrections


# ---------------------------------------------------------------------------
# Vehicle guardrails
# ---------------------------------------------------------------------------

def _apply_vehicle_guardrails(document: dict, suggestion: dict, storage_paths: list,
                              decision_context: DecisionContext | None = None) -> list[str]:
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


# ---------------------------------------------------------------------------
# Topic guardrails
# ---------------------------------------------------------------------------

def _apply_topic_guardrails(document: dict, suggestion: dict, correspondents: list, storage_paths: list,
                            decision_context: DecisionContext | None = None,
                            _corr_lookup: dict[int, str] | None = None) -> list[str]:
    """Generic content guardrails for health/school/ticket documents."""
    corrections = []
    if _corr_lookup is None:
        _corr_lookup = _build_id_name_map(correspondents)
    current_corr_name = _get_correspondent_name_by_id(correspondents, document.get("correspondent"), _lookup=_corr_lookup)
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


# ---------------------------------------------------------------------------
# Learning guardrails
# ---------------------------------------------------------------------------

def _apply_learning_guardrails(suggestion: dict, storage_paths: list, learning_hints: list[dict] | None) -> list[str]:
    """Apply conservative priors learned from confirmed examples."""
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


# ---------------------------------------------------------------------------
# Rule-based fast path
# ---------------------------------------------------------------------------

def _try_rule_based_suggestion(document: dict, learning_hints: list[dict],
                               storage_paths: list) -> dict | None:
    """Fast path: skip LLM entirely for well-known correspondent patterns."""
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
    """Build suggestion from learning priors without LLM call (fallback)."""
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


# ---------------------------------------------------------------------------
# Content hint detection
# ---------------------------------------------------------------------------

def _detect_content_hints(document: dict) -> list[str]:
    """Detect content patterns and suggest additional tags/document type hints."""
    hints = []
    content = (document.get("content") or "")[:3000].lower()
    if not content:
        return hints

    if re.search(r"[a-z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{0,4}", content):
        hints.append("iban_present")
    if re.search(r"(rechnungsnr|rechnungsnummer|invoice\s*no|faktura)", content):
        hints.append("invoice_detected")
    if re.search(r"(vertragsnummer|vertragslaufzeit|kuendigungsfrist|laufzeit)", content):
        hints.append("contract_detected")
    if re.search(r"(steuernummer|finanzamt|einkommensteuerbescheid|steuererkl)", content):
        hints.append("tax_detected")
    if re.search(r"(versicherungsnummer|policen?nummer|versicherungsschein|schadensnummer)", content):
        hints.append("insurance_detected")
    if re.search(r"(bruttobezug|nettobetrag|lohnsteuer|sozialversicherung|gehaltsabrechnung)", content):
        hints.append("salary_detected")
    if re.search(r"(diagnose|befund|therapie|patient|krankenkasse|heilbehandlung)", content):
        hints.append("medical_detected")

    return hints


# ---------------------------------------------------------------------------
# Tag selection
# ---------------------------------------------------------------------------

def _select_controlled_tags(suggested_tags: list, existing_tags: list, taxonomy=None,
                            run_db=None, run_id: int | None = None,
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

        if canonical and _cfg.AUTO_CREATE_TAXONOMY_TAGS:
            add_approved_once(canonical, raw_tag)
            continue

        if _cfg.ALLOW_NEW_TAGS and not taxonomy:
            add_approved_once(_normalize_text(raw_tag), raw_tag)
            continue

        dropped.append((raw_tag, "blocked by tag policy (existing-only)"))
        if run_db:
            run_db.record_tag_event(run_id, doc_id, "blocked", raw_tag, "blocked by tag policy (existing-only)")

    _TAG_CONFLICTS = [
        ("privat", "arbeit"),
        ("privat", "geschaeftlich"),
        ("eingang", "ausgang"),
    ]
    approved_lower = [t.lower() for t in approved]
    for tag_a, tag_b in _TAG_CONFLICTS:
        if tag_a in approved_lower and tag_b in approved_lower:
            idx_a = approved_lower.index(tag_a)
            idx_b = approved_lower.index(tag_b)
            remove_idx = max(idx_a, idx_b)
            removed_tag = approved[remove_idx]
            dropped.append((removed_tag, f"konflikt mit '{approved[min(idx_a, idx_b)]}'"))
            approved.pop(remove_idx)
            approved_lower.pop(remove_idx)

    approved = approved[:MAX_TAGS_PER_DOC]
    return approved, dropped


# ---------------------------------------------------------------------------
# Review reasons & marking
# ---------------------------------------------------------------------------

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

    ocr_quality, _ = _assess_ocr_quality(document)
    if ocr_quality == "poor" and not is_prior_only:
        reasons.append("ocr-qualitaet-schlecht")

    return reasons


def _mark_document_for_review(paperless, document: dict, tags: list, reason: str):
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


# ---------------------------------------------------------------------------
# HTTP error helpers & update with fallbacks
# ---------------------------------------------------------------------------

def _http_error_detail(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    status = getattr(response, "status_code", "?")
    text = (getattr(response, "text", "") or "").strip()
    if text:
        return f"HTTP {status}: {text[:1000]}"
    return f"HTTP {status}"


def _apply_update_with_fallbacks(paperless, doc_id: int, update_data: dict) -> tuple[dict, list[str]]:
    """Update with guarded retries for common 400 causes."""
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

            if "archive_serial" in lower and "archive_serial_number" in payload:
                payload.pop("archive_serial_number", None)
                notes.append("retry_without_archive_serial_number")
                changed = True

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
