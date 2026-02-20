"""Review queue: learn from corrections, auto-resolve reviews."""

from __future__ import annotations

import json

from .config import REVIEW_TAG_NAME, console, log
from .db import LocalStateDB
from .client import PaperlessClient
from .learning import LearningProfile, LearningExamples
from .utils import _build_id_name_map, _is_fully_organized, _normalize_tag_name, _normalize_text


def _remove_review_tag_from_document(paperless: PaperlessClient, doc: dict):
    """Entfernt den Manuell-Pruefen-Tag von einem Dokument, falls vorhanden."""
    try:
        all_tags = paperless.get_tags()
    except Exception as exc:
        log.debug("Tags konnten nicht geladen werden fuer Review-Tag-Entfernung: %s", exc)
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

    corr_map = _build_id_name_map(all_correspondents)
    type_map = _build_id_name_map(all_doc_types)
    path_map = _build_id_name_map(all_storage_paths)
    tag_map = _build_id_name_map(all_tags)

    corr_name = corr_map.get(int(document["correspondent"]), "") if document.get("correspondent") else ""
    doctype_name = type_map.get(int(document["document_type"]), "") if document.get("document_type") else ""
    path_name = path_map.get(int(document["storage_path"]), "") if document.get("storage_path") else ""
    tag_names = [tag_map[tid] for tid in (document.get("tags") or []) if tid in tag_map]

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

        # Negative learning: if original suggestion differed, record it as rejected
        try:
            orig_suggestion = review.get("suggestion")
            if isinstance(orig_suggestion, str):
                try:
                    orig_suggestion = json.loads(orig_suggestion)
                except (json.JSONDecodeError, TypeError):
                    orig_suggestion = None
            if orig_suggestion and isinstance(orig_suggestion, dict):
                orig_corr = _normalize_text(str(orig_suggestion.get("correspondent", "")))
                curr_corr = _normalize_text(corr_name)
                orig_type = _normalize_text(str(orig_suggestion.get("document_type", "")))
                curr_type = _normalize_text(doctype_name)
                orig_path = _normalize_text(str(orig_suggestion.get("storage_path", "")))
                curr_path = _normalize_text(path_name)
                if (orig_corr and orig_corr != curr_corr) or \
                   (orig_type and orig_type != curr_type) or \
                   (orig_path and orig_path != curr_path):
                    learning_examples.append(document, orig_suggestion, rejected=True)
                    log.info(f"  Negative Learning: Originaler Vorschlag als Anti-Pattern gespeichert")
        except Exception as exc:
            log.debug("Negative Learning fehlgeschlagen: %s", exc)

    if learning_profile:
        try:
            learning_profile.learn_from_document(document, current_suggestion)
            learning_profile.save()
            log.info(f"  Learning-Profil aktualisiert fuer Dokument #{doc_id}")
        except Exception as exc:
            log.warning(f"  Learning-Profil konnte nicht aktualisiert werden: {exc}")

    # Mark confidence calibration as corrected
    try:
        run_db.mark_confidence_corrected(doc_id)
    except Exception as exc:
        log.debug("Konfidenz-Kalibrierung fehlgeschlagen fuer #%s: %s", doc_id, exc)

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
