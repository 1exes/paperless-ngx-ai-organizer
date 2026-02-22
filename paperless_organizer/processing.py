"""Document processing: resolve IDs, process single documents, auto-organize, batch."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm
from rich.table import Table
from rich.panel import Panel

from . import config as _cfg
from .config import (
    AGENT_WORKERS,
    ALLOW_NEW_CORRESPONDENTS,
    ALLOW_NEW_STORAGE_PATHS,
    LEARNING_EXAMPLE_LIMIT,
    LEARNING_PRIOR_MAX_HINTS,
    LEARNING_PRIOR_MIN_SAMPLES,
    LLM_COMPACT_TIMEOUT,
    LLM_COMPACT_TIMEOUT_RETRY,
    LLM_TIMEOUT,
    MAX_TOTAL_TAGS,
    SKIP_RECENT_LLM_ERRORS_MINUTES,
    SKIP_RECENT_LLM_ERRORS_THRESHOLD,
    USE_ARCHIVE_SERIAL_NUMBER,
    WRITE_LOCK,
    console,
    log,
)
from .constants import ALLOWED_DOC_TYPES
from .models import DecisionContext, ProcessingContext
from .db import LocalStateDB
from .client import PaperlessClient
from .taxonomy import TagTaxonomy
from .learning import LearningProfile, LearningExamples
from .llm import LocalLLMAnalyzer
from .utils import (
    _assess_ocr_quality,
    _build_id_name_map,
    _detect_language,
    _extract_document_date,
    _improve_title,
    _is_fully_organized,
    _normalize_text,
    _normalize_tag_name,
    _sanitize_suggestion_spelling,
)
from .guardrails import (
    _apply_learning_guardrails,
    _apply_topic_guardrails,
    _apply_update_with_fallbacks,
    _apply_vehicle_guardrails,
    _apply_vendor_guardrails,
    _build_suggestion_from_priors,
    _collect_hard_review_reasons,
    _detect_content_hints,
    _mark_document_for_review,
    _resolve_correspondent_from_name,
    _select_controlled_tags,
    _try_rule_based_suggestion,
)
from .web_hints import _web_search_document_context


# ---------------------------------------------------------------------------
# ID resolution & display
# ---------------------------------------------------------------------------

def resolve_ids(ctx: ProcessingContext, suggestion: dict,
                doc_id: int | None = None,
                quiet: bool = False) -> dict:
    """Wandelt Namen in IDs um, erstellt fehlende Eintraege."""
    paperless = ctx.paperless
    tags = ctx.tags
    correspondents = ctx.correspondents
    doc_types = ctx.doc_types
    storage_paths = ctx.storage_paths
    taxonomy = ctx.taxonomy
    run_db = ctx.run_db
    run_id = ctx.run_id

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
            can_create = _cfg.ALLOW_NEW_TAGS or (
                taxonomy is not None
                and _cfg.AUTO_CREATE_TAXONOMY_TAGS
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
                if not quiet:
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
                    if not quiet:
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
            if not quiet:
                console.print(f"  [red]Dokumenttyp '{type_name}' nicht erlaubt, uebersprungen.[/red]")
        else:
            canonical_name = allowed_lower[key]
            canonical_id = type_exact.get(canonical_name)
            if canonical_id is not None:
                type_id = canonical_id
            else:
                try:
                    if not quiet:
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
                    if not quiet:
                        console.print(f"  [yellow]+ Neuer Speicherpfad:[/yellow] {path_value}")
                    new = paperless.create_storage_path(path_value, template)
                    path_id = new["id"]
                    storage_paths.append(new)
                except requests.exceptions.HTTPError as e:
                    if not quiet:
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
                    doc_types: list, storage_paths: list,
                    quiet: bool = False):
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

    # Confidence display with color-coding and source
    conf = suggestion.get("confidence", "").lower()
    conf_style = {"high": "[bold green]HOCH[/bold green]", "medium": "[yellow]MITTEL[/yellow]",
                  "low": "[red]NIEDRIG[/red]", "prior_only": "[cyan]NUR PRIOR[/cyan]"}.get(conf, conf)
    source = suggestion.get("_source", "")
    if not source:
        if conf == "prior_only":
            source = "Learning-Prior"
        elif suggestion.get("_rule_based"):
            source = "Regelbasiert"
        else:
            source = "LLM"
    source_label = f" [dim]({source})[/dim]" if source else ""
    if conf:
        table.add_row("Konfidenz", "", f"{conf_style}{source_label}")

    if not quiet:
        console.print(table)
        if suggestion.get("reasoning"):
            console.print(Panel(suggestion["reasoning"], title="Begruendung", border_style="blue"))


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def process_document(doc_id: int, ctx: ProcessingContext,
                     dry_run: bool = True,
                     batch_mode: bool = False,
                     prefer_compact: bool = False,
                     status_callback: Callable[[str], None] | None = None,
                     quiet: bool = False) -> bool:
    """Einzelnes Dokument analysieren und organisieren."""
    paperless = ctx.paperless
    analyzer = ctx.analyzer
    tags = ctx.tags
    correspondents = ctx.correspondents
    doc_types = ctx.doc_types
    storage_paths = ctx.storage_paths
    taxonomy = ctx.taxonomy
    decision_context = ctx.decision_context
    learning_profile = ctx.learning_profile
    learning_examples = ctx.learning_examples
    run_db = ctx.run_db
    run_id = ctx.run_id
    t_total_start = time.perf_counter()
    _cb = status_callback or (lambda _msg: None)

    if not quiet:
        console.print(f"\n[bold]{'=' * 60}[/bold]")
    _cb("Lade Dokument...")
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
    _cb(f"Geladen: {doc_title} ({content_len} Zeichen)")
    log.info(f"  Geladen: [white]{doc_title}[/white] ({content_len} Zeichen)")

    # OCR quality check
    ocr_quality, ocr_score = _assess_ocr_quality(document)
    _cb(f"OCR-Check: {ocr_quality}")
    if ocr_quality == "poor":
        log.warning(f"  [yellow]OCR-Qualitaet schlecht[/yellow] (Score: {ocr_score:.2f}) - Ergebnis koennte ungenau sein")
    elif ocr_quality == "medium":
        log.info(f"  OCR-Qualitaet: mittel (Score: {ocr_score:.2f})")

    # Language detection
    doc_lang = _detect_language(document.get("content") or "")
    _cb(f"Sprache: {doc_lang or 'de'}")
    if doc_lang == "en":
        log.info("  Sprache: Englisch erkannt")

    # Content pattern hints
    content_hints = _detect_content_hints(document)
    if content_hints:
        _cb(f"Muster: {', '.join(content_hints)}")
        log.info(f"  Inhaltsmuster erkannt: {', '.join(content_hints)}")

    few_shot_examples = learning_examples.select(document, limit=LEARNING_EXAMPLE_LIMIT) if learning_examples else []
    learning_hints = (
        learning_examples.routing_hints_for_document(document, limit=LEARNING_PRIOR_MAX_HINTS)
        if learning_examples else []
    )

    # Rule-based fast path: skip LLM for well-known correspondent patterns
    rule_suggestion = _try_rule_based_suggestion(document, learning_hints, storage_paths)
    suggestion = None
    if rule_suggestion:
        rule_suggestion["_rule_based"] = True
        _cb(f"Regelbasiert erkannt -> {rule_suggestion.get('correspondent', '?')}")
        log.info(f"  [green]Regelbasiert[/green] (LLM uebersprungen): {rule_suggestion.get('correspondent')}")
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "rule_based", document, suggestion=rule_suggestion)
        suggestion = rule_suggestion
    else:
        _cb("LLM-Analyse laeuft...")
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
                doc_language=doc_lang,
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
                        doc_language=doc_lang,
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
                        doc_language=doc_lang,
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
                    doc_language=doc_lang,
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
    if suggestion is None and _cfg.ENABLE_WEB_HINTS:
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
                    doc_language=doc_lang,
                )
            except Exception as exc:
                log.debug("Websuche-Fallback LLM-Retry fehlgeschlagen: %s", exc)
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
    _cb(f"LLM-Antwort: {suggestion.get('title', '?')} ({t_elapsed:.1f}s)")
    log.info(f"  LLM-Antwort erhalten ({t_elapsed:.1f}s) -> Titel: [green]{suggestion.get('title', '?')}[/green]")

    # E5: Low-confidence + no correspondent -> web search verification
    confidence = (suggestion.get("confidence") or "high").lower()
    corr_now = _normalize_text(str(suggestion.get("correspondent", "")))
    if (
        _cfg.ENABLE_WEB_HINTS
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
                    doc_language=doc_lang,
                )
                if verified and _normalize_text(str(verified.get("correspondent", ""))):
                    suggestion = verified
                    log.info(f"  [green]Websuche-Verifikation erfolgreich[/green]: Korr={verified.get('correspondent')}")
            except Exception:
                pass  # Keep original suggestion on error

    _sanitize_suggestion_spelling(suggestion)

    # Fill missing correspondent from learning hints if LLM didn't identify one
    suggested_corr = _normalize_text(str(suggestion.get("correspondent", "")))
    if not suggested_corr and learning_hints:
        best_hint = learning_hints[0]
        hint_corr = _normalize_text(str(best_hint.get("correspondent", "")))
        hint_count = int(best_hint.get("count", 0) or 0)
        if hint_corr and hint_count >= LEARNING_PRIOR_MIN_SAMPLES:
            suggestion["correspondent"] = hint_corr
            log.info(f"  [cyan]Learning-Korrespondent[/cyan]: {hint_corr} (n={hint_count})")

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

    _cb("Guardrails pruefen...")
    corr_lookup = _build_id_name_map(correspondents)

    guardrail_fixes = _apply_vendor_guardrails(
        document,
        suggestion,
        correspondents,
        storage_paths,
        decision_context=decision_context,
        _corr_lookup=corr_lookup,
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
        _corr_lookup=corr_lookup,
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

    _cb("Tags auswaehlen...")
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
        _cb(f"Review noetig: {reason_text}")
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
    show_suggestion(document, suggestion, asn, tags, correspondents, doc_types, storage_paths, quiet=quiet)

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

    _cb("Aenderungen anwenden...")
    log.info(f"  IDs aufloesen und Aenderungen anwenden...")
    try:
        with WRITE_LOCK:
            update_data = resolve_ids(
                ctx,
                suggestion,
                doc_id=doc_id,
                quiet=quiet,
            )
            result, fallback_notes = _apply_update_with_fallbacks(paperless, doc_id, update_data)
            for note in fallback_notes:
                log.warning(f"  [yellow]API-Fallback[/yellow]: {note}")
        result_label = result.get("archived_file_name") or result.get("original_file_name") or result.get("title") or "?"
        t_total = time.perf_counter() - t_total_start
        _cb(f"Fertig: {doc_title} ({t_total:.1f}s)")
        log.info(f"[bold green]FERTIG[/bold green] Dokument #{doc_id} aktualisiert -> {result_label} ({t_total:.1f}s gesamt)")
        if run_db and run_id:
            run_db.record_document(run_id, doc_id, "updated", document, suggestion=suggestion)
            # Track confidence for calibration
            conf = (suggestion.get("confidence") or "high").lower()
            source = "rule_based" if rule_suggestion else "llm"
            run_db.record_confidence(doc_id, conf, source=source)
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


# ---------------------------------------------------------------------------
# Auto-organize & batch
# ---------------------------------------------------------------------------

def auto_organize_all(ctx: ProcessingContext, dry_run: bool,
                      force_recheck_all: bool = False,
                      prefer_compact: bool = False):
    """Scannt ALLE Dokumente, ueberspringt bereits sortierte, organisiert den Rest."""

    log.info("[bold]AUTO-SORTIERUNG[/bold] gestartet - scanne alle Dokumente...")

    paperless = ctx.paperless
    tags = ctx.tags
    run_db = ctx.run_db

    documents = paperless.get_documents()
    total = len(documents)
    log.info(f"  {total} Dokumente geladen")

    if total == 0:
        log.warning("[yellow]Keine Dokumente gefunden - API-Verbindung pruefen![/yellow]")
        return {"total": 0, "todo": 0, "applied": 0, "errors": 0}

    # Duplikate rausfiltern
    duplikat_tag_id = next((t["id"] for t in tags if t["name"] == "Duplikat"), None)
    if duplikat_tag_id:
        before = len(documents)
        documents = [d for d in documents if duplikat_tag_id not in (d.get("tags") or [])]
        skipped = before - len(documents)
        if skipped:
            log.info(f"  {skipped} Duplikate uebersprungen")

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

    # Sort: newest documents first
    todo.sort(key=lambda d: d.get("added", d.get("created", "")), reverse=True)

    if not todo:
        if skipped_recent_total:
            log.warning(
                f"[yellow]Nichts verarbeitet: {skipped_recent_total} Dokumente sind temporaer im LLM-Fehler-Backoff.[/yellow]"
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
    error_categories: dict[str, int] = defaultdict(int)
    doc_times: list[float] = []

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
                if doc_times:
                    avg_t = sum(doc_times) / len(doc_times)
                    eta_sec = avg_t * (len(todo) - i + 1)
                    eta_min = eta_sec / 60
                    eta_str = f" | ETA {eta_min:.0f}m" if eta_min >= 1 else f" | ETA {eta_sec:.0f}s"
                else:
                    eta_str = ""
                progress.update(task, description=f"[{i}/{len(todo)}] #{doc['id']}{eta_str}")
                log.info(f"[bold]--- {i}/{len(todo)} --- Dokument #{doc['id']}[/bold]")
                doc_t0 = time.perf_counter()
                try:
                    if process_document(doc["id"], ctx, dry_run,
                                        batch_mode=not dry_run,
                                        prefer_compact=prefer_compact):
                        applied += 1
                except Exception as e:
                    errors += 1
                    err_str = str(e).lower()
                    if "timeout" in err_str:
                        error_categories["timeout"] += 1
                    elif "json" in err_str:
                        error_categories["json_parse"] += 1
                    elif "connection" in err_str or "connect" in err_str:
                        error_categories["connection"] += 1
                    elif "status" in err_str or "http" in err_str:
                        error_categories["api_error"] += 1
                    else:
                        error_categories["other"] += 1
                    log.error(f"Fehler bei #{doc['id']}: {e}")
                doc_times.append(time.perf_counter() - doc_t0)
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
                    ctx.copy_master_data(),
                    dry_run,
                    True,
                    prefer_compact,
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
    throughput = (len(todo) / batch_elapsed * 60) if batch_elapsed > 0 else 0
    log.info(f"[bold]AUTO-SORTIERUNG FERTIG[/bold] - {applied}/{len(todo)} aktualisiert, "
             f"{errors} Fehler, {batch_elapsed:.1f}s gesamt ({avg_time:.1f}s/Dokument, "
             f"{throughput:.1f} Dok/min)")
    if error_categories:
        cats = ", ".join(f"{k}={v}" for k, v in sorted(error_categories.items(), key=lambda x: x[1], reverse=True))
        log.info(f"  Fehler-Aufschluesselung: {cats}")
    if _cfg.AUTO_CLEANUP_AFTER_ORGANIZE and not dry_run:
        from .cleanup import cleanup_tags, cleanup_correspondents, cleanup_document_types
        log.info("[bold]AUTO-CLEANUP[/bold] nach Auto-Sortierung gestartet")
        cleanup_tags(paperless, dry_run=False)
        cleanup_correspondents(paperless, dry_run=False)
        cleanup_document_types(paperless, dry_run=False)
    return {"total": total, "todo": len(todo), "applied": applied, "errors": errors,
            "elapsed_sec": round(batch_elapsed, 1), "avg_sec_per_doc": round(avg_time, 1)}


def batch_process(ctx: ProcessingContext, dry_run: bool,
                  mode: str = "untagged", limit: int = 0):
    """Mehrere Dokumente mit Filter verarbeiten."""

    paperless = ctx.paperless
    tags = ctx.tags
    run_db = ctx.run_db

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
                if process_document(doc["id"], ctx, dry_run,
                                    batch_mode=not dry_run):
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
                    ctx.copy_master_data(),
                    dry_run,
                    True,
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
    if _cfg.AUTO_CLEANUP_AFTER_ORGANIZE and not dry_run:
        from .cleanup import cleanup_tags, cleanup_correspondents, cleanup_document_types
        log.info("[bold]AUTO-CLEANUP[/bold] nach Batch gestartet")
        cleanup_tags(paperless, dry_run=False)
        cleanup_correspondents(paperless, dry_run=False)
        cleanup_document_types(paperless, dry_run=False)
    return {"total": len(documents), "applied": applied, "errors": errors}
