"""Main application: TUI menu system, live-watch, autopilot."""

from __future__ import annotations

import json
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime

import requests
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from . import config as _cfg
from .config import (
    AUTOPILOT_CLEANUP_EVERY_CYCLES,
    AUTOPILOT_CONTEXT_REFRESH_CYCLES,
    AUTOPILOT_DUPLICATE_SCAN_EVERY_CYCLES,
    AUTOPILOT_INTERVAL_SEC,
    AUTOPILOT_REVIEW_RESOLVE_EVERY_CYCLES,
    AUTOPILOT_MAX_NEW_DOCS_PER_CYCLE,
    AUTOPILOT_RECHECK_ALL_ON_START,
    AUTOPILOT_START_WITH_AUTO_ORGANIZE,
    KNOWLEDGE_DB_URL,
    LEARNING_EXAMPLES_FILE,
    LEARNING_PROFILE_FILE,
    LIVE_WATCH_COMPACT_FIRST,
    LIVE_WATCH_CONTEXT_REFRESH_CYCLES,
    LIVE_WATCH_INTERVAL_SEC,
    LLM_KEEP_ALIVE,
    LLM_MODEL,
    LLM_SPEEDCHECK_AUTO_SWITCH,
    LLM_SPEEDCHECK_ENABLED,
    LLM_SPEEDCHECK_INTERVAL_CYCLES,
    LLM_SPEEDCHECK_MAX_TIME,
    LLM_SPEEDCHECK_TIMEOUT,
    LLM_URL,
    LOG_DIR,
    LOG_FILE,
    MAX_TAGS_PER_DOC,
    MAX_TOTAL_TAGS,
    QUIET_HOURS_END,
    QUIET_HOURS_START,
    STATE_DB_FILE,
    TAXONOMY_FILE,
    WATCH_ERROR_BACKOFF_BASE_SEC,
    WATCH_ERROR_BACKOFF_MAX_SEC,
    WATCH_RECONNECT_ERROR_THRESHOLD,
    __version__,
    _is_quiet_hours,
    console,
    log,
)
from .cleanup import (
    cleanup_all,
    cleanup_correspondents,
    cleanup_document_types,
    cleanup_tags,
)
from .client import PaperlessClient
from .db import LocalStateDB
from .duplicates import find_duplicates
from .guardrails import (
    _detect_content_hints,
    build_decision_context,
)
from .knowledge import KnowledgeDB
from .learning import LearningExamples, LearningProfile
from .llm import LocalLLMAnalyzer
from .models import DecisionContext, ProcessingContext
from .processing import (
    auto_organize_all,
    batch_process,
    process_document,
)
from .review import auto_resolve_reviews, learn_from_review
from .statistics import show_monthly_report, show_statistics
from .taxonomy import TagTaxonomy
from .utils import _is_fully_organized, _normalize_tag_name


# =========================================================================
# App class
# =========================================================================

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
        self.knowledge_db: KnowledgeDB | None = None
        self._init_knowledge_db()
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

    def _init_knowledge_db(self) -> bool:
        """Initialize Knowledge DB if configured."""
        if not KNOWLEDGE_DB_URL:
            log.debug("KnowledgeDB: KNOWLEDGE_DB_URL nicht gesetzt, uebersprungen")
            return False
        try:
            self.knowledge_db = KnowledgeDB(KNOWLEDGE_DB_URL)
            log.info("KnowledgeDB: Verbunden (%s)", KNOWLEDGE_DB_URL.split("@")[-1] if "@" in KNOWLEDGE_DB_URL else KNOWLEDGE_DB_URL)
            return True
        except Exception as exc:
            log.warning(f"KnowledgeDB: Verbindung fehlgeschlagen: {exc}")
            self.knowledge_db = None
            return False

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

    def _run_speedcheck(self, auto_switch: bool = True) -> dict | None:
        """Fuehrt LLM-Speedcheck durch und wechselt ggf. auf schnellstes Modell.

        Returns benchmark result dict or None on error.
        """
        if not self.analyzer:
            return None

        log.info("LLM-Speedcheck: Teste aktuelles Modell...")
        check = self.analyzer.speedcheck(timeout=LLM_SPEEDCHECK_TIMEOUT)

        if check["ok"] and check["response_time"] <= LLM_SPEEDCHECK_MAX_TIME:
            log.info(
                "[green]LLM-Speedcheck OK[/green]: %s in %.1fs (Limit: %.1fs)",
                check["model"], check["response_time"], LLM_SPEEDCHECK_MAX_TIME,
            )
            return {"results": [check], "best_model": check["model"],
                    "best_time": check["response_time"],
                    "current_model": check["model"], "current_ok": True,
                    "current_time": check["response_time"],
                    "max_acceptable_time": LLM_SPEEDCHECK_MAX_TIME,
                    "switched": False}

        # Current model too slow or failed -> benchmark all
        if check["ok"]:
            log.warning(
                "[yellow]LLM-Speedcheck[/yellow]: %s zu langsam (%.1fs > %.1fs) - benchmarke alle Modelle...",
                check["model"], check["response_time"], LLM_SPEEDCHECK_MAX_TIME,
            )
        else:
            log.warning(
                "[yellow]LLM-Speedcheck[/yellow]: %s fehlgeschlagen (%s) - benchmarke alle Modelle...",
                check["model"], check["error"],
            )

        benchmark = self.analyzer.benchmark_available_models(
            max_acceptable_time=LLM_SPEEDCHECK_MAX_TIME,
            timeout_per_model=LLM_SPEEDCHECK_TIMEOUT,
        )
        benchmark["switched"] = False

        best = benchmark["best_model"]
        best_time = benchmark["best_time"]
        ok_results = [r for r in benchmark["results"] if r["ok"]]

        if not ok_results:
            log.warning("[yellow]LLM-Speedcheck[/yellow]: Kein Modell erreichbar - fahre trotzdem fort")
            return benchmark

        if best != self.analyzer.model and auto_switch and LLM_SPEEDCHECK_AUTO_SWITCH:
            log.warning(
                "[yellow]LLM-Speedcheck: Wechsle Modell[/yellow] %s (%.1fs) -> %s (%.1fs)",
                self.analyzer.model or "(default)", benchmark.get("current_time", 0),
                best, best_time,
            )
            self.analyzer.model = best
            self.analyzer._original_model = best
            self.llm_model = best
            benchmark["switched"] = True
        elif best_time > LLM_SPEEDCHECK_MAX_TIME:
            log.warning(
                "[yellow]LLM-Speedcheck[/yellow]: Bestes Modell %s immer noch langsam (%.1fs) - fahre trotzdem fort",
                best, best_time,
            )

        return benchmark

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
        if not _cfg.AUTO_CREATE_TAXONOMY_TAGS:
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

    def _build_ctx(self, tags: list, correspondents: list, doc_types: list,
                   storage_paths: list, decision_context: DecisionContext | None = None,
                   run_id: int | None = None) -> ProcessingContext:
        """Build a ProcessingContext from current App state + master data."""
        return ProcessingContext(
            paperless=self.paperless,
            analyzer=self.analyzer,
            tags=tags,
            correspondents=correspondents,
            doc_types=doc_types,
            storage_paths=storage_paths,
            taxonomy=self.taxonomy,
            decision_context=decision_context,
            learning_profile=self.learning_profile,
            learning_examples=self.learning_examples,
            knowledge_db=self.knowledge_db,
            run_db=self.run_db,
            run_id=run_id,
        )

    def _collect_decision_context(self, correspondents: list, storage_paths: list) -> DecisionContext:
        """Phase 1: Daten sammeln, danach Entscheidungen treffen."""
        with console.status("Sammle Entscheidungsdaten..."):
            documents = self.paperless.get_documents()
        context = build_decision_context(
            documents,
            correspondents,
            storage_paths,
            learning_profile=self.learning_profile,
            knowledge_db=self.knowledge_db,
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
        tag_policy = "bestehende Tags" if not _cfg.ALLOW_NEW_TAGS else "neue Tags erlaubt"
        taxonomy_info = f"{len(self.taxonomy.canonical_tags)} Tags"
        auto_tax = "JA" if _cfg.AUTO_CREATE_TAXONOMY_TAGS else "NEIN"
        recheck_info = "JA" if _cfg.RECHECK_ALL_DOCS_IN_AUTO else "NEIN"
        # Quick organization status from cached paperless data
        org_line = ""
        if self.paperless:
            try:
                docs = self.paperless.get_documents()
                if docs:
                    organized = sum(1 for d in docs if _is_fully_organized(d))
                    pct = organized / len(docs) * 100
                    remaining = len(docs) - organized
                    color = "green" if pct >= 90 else "yellow" if pct >= 50 else "red"
                    org_line = f"\nOrganisiert: [{color}]{organized}/{len(docs)} ({pct:.0f}%)[/{color}] | {remaining} offen"
            except Exception:
                pass
        console.print(Panel(
            f"[bold]Paperless-NGX Organizer[/bold]\n"
            f"Server: {url}\n"
            f"LLM: {llm_model_label} ({self.llm_url})\n"
            f"Modus: {mode}\n"
            f"Tag-Policy: {tag_policy} (max {MAX_TAGS_PER_DOC})\n"
            f"Taxonomie: {taxonomy_info} | Auto-Create: {auto_tax} | Global-Limit: {MAX_TOTAL_TAGS}\n"
            f"Auto-Modus recheck alle Dokumente: {recheck_info}"
            f"{org_line}\n"
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
            kb_label = ""
            if self.knowledge_db:
                try:
                    ks = self.knowledge_db.get_statistics()
                    kb_label = f" [{ks['entities']} Entities, {ks['active_facts']} Fakten]"
                except Exception:
                    kb_label = " [verbunden]"
            else:
                kb_label = " [nicht konfiguriert]"
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
                ("10", f"Wissensdatenbank{kb_label}"),
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
            elif choice == "10":
                self._run_action_safely(self.menu_knowledge, "Wissensdatenbank")
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
            ctx = self._build_ctx(tags, correspondents, doc_types, storage_paths,
                                  decision_context=decision_context, run_id=run_id)
            ok = process_document(doc_id, ctx, self.dry_run)
            self._finish_run(run_id, {"mode": "single", "doc_id": doc_id, "updated": int(ok)})

        elif choice in ("2", "3", "4"):
            limit_str = Prompt.ask("Limit (0 = alle)", default="0")
            try:
                limit = int(limit_str)
            except ValueError:
                limit = 0
            mode_map = {"2": "untagged", "3": "unorganized", "4": "all"}
            run_id = self._start_run(f"batch_{mode_map[choice]}")
            ctx = self._build_ctx(tags, correspondents, doc_types, storage_paths,
                                  decision_context=decision_context, run_id=run_id)
            summary = batch_process(ctx, self.dry_run,
                                    mode=mode_map[choice], limit=limit)
            self._finish_run(run_id, summary)

    def action_auto_organize(self):
        """Alles sortieren - scannt alle Dokumente, ueberspringt bereits fertige."""
        if not self._init_paperless():
            return
        if not self._init_analyzer():
            return

        force_recheck_all = Confirm.ask(
            "Alle Dokumente fuer diesen Lauf neu pruefen?",
            default=_cfg.RECHECK_ALL_DOCS_IN_AUTO,
        )

        tags, correspondents, doc_types, storage_paths = self._load_master_data()
        self._ensure_taxonomy_tags(tags)
        decision_context = self._collect_decision_context(correspondents, storage_paths)
        action_name = "auto_organize_recheck_all" if force_recheck_all else "auto_organize"
        run_id = self._start_run(action_name)
        ctx = self._build_ctx(tags, correspondents, doc_types, storage_paths,
                              decision_context=decision_context, run_id=run_id)
        summary = auto_organize_all(ctx, self.dry_run,
                                    force_recheck_all=force_recheck_all)
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

                ctx = self._build_ctx(tags, correspondents, doc_types, storage_paths,
                                      decision_context=decision_context, run_id=run_id)
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
                            doc_id, ctx, self.dry_run,
                            batch_mode=not self.dry_run,
                            prefer_compact=LIVE_WATCH_COMPACT_FIRST,
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

    # ------------------------------------------------------------------
    # Autopilot: Live-Dashboard helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dashboard(
        *,
        start_time: float,
        cycle: int,
        interval_sec: int,
        total_docs: int,
        total_candidates: int,
        total_updated: int,
        total_doc_failures: int,
        total_reviews_resolved: int,
        llm_model: str,
        llm_ok: bool,
        llm_last_time: float,
        llm_recovery_msg: str,
        current_doc_id: int | None,
        current_doc_title: str,
        current_step: str,
        current_step_start: float,
        step_progress: float,
        recent_results: list[tuple[str, int, str, float]],
        phase: str = "",
        init_steps: list[tuple[str, str]] | None = None,
        spin_idx: int = 0,
    ) -> Panel:
        """Build a Rich Panel showing autopilot live status.

        phase: current high-level phase (e.g. "Initialisierung", "Sortierung", "Watch")
        init_steps: list of (label, status) tuples for startup checklist.
            status is one of: "done", "active", "pending", "error"
        """
        elapsed = time.perf_counter() - start_time
        elapsed_m = int(elapsed // 60)
        elapsed_s = int(elapsed % 60)
        success_rate = (total_updated / total_candidates * 100) if total_candidates else 0

        # -- LLM status --
        if llm_recovery_msg:
            llm_label = f"[yellow]{llm_model} - {llm_recovery_msg}[/yellow]"
        elif llm_ok:
            llm_label = f"[green]{llm_model} ✓[/green]"
        else:
            llm_label = f"[red]{llm_model} ✗[/red]"
        llm_time_str = f" | Letzte Antwort: {llm_last_time:.1f}s" if llm_last_time > 0 else ""

        lines: list[str] = []

        # -- Phase header --
        phase_label = f" | Phase: {phase}" if phase else ""
        lines.append(
            f"Laufzeit: {elapsed_m}m{elapsed_s:02d}s{phase_label}"
        )

        # -- Init checklist (shown during startup) --
        if init_steps:
            lines.append("")
            _SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            _spin = _SPINNER_CHARS[spin_idx % len(_SPINNER_CHARS)]
            _ICONS = {"done": "[green]✓[/green]", "active": f"[cyan]{_spin}[/cyan]", "pending": "[dim]○[/dim]", "error": "[red]✗[/red]"}
            for label, status in init_steps:
                icon = _ICONS.get(status, "○")
                style_open, style_close = ("", "")
                if status == "active":
                    style_open, style_close = "[bold]", "[/bold]"
                elif status == "done":
                    style_open, style_close = "[dim]", "[/dim]"
                lines.append(f"  {icon} {style_open}{label}{style_close}")
            lines.append("")

        # -- Stats (shown once we have data) --
        if total_candidates > 0 or cycle > 0:
            if cycle > 0:
                lines.append(f"Zyklus: {cycle} | Intervall: {interval_sec}s")
            lines.append(
                f"Dokumente: {total_updated}/{total_docs} verarbeitet | "
                f"{total_candidates} Kandidaten | {total_doc_failures} Fehler"
            )
            lines.append(
                f"Erfolgsrate: {success_rate:.0f}% | Reviews geloest: {total_reviews_resolved}"
            )

        lines.append(f"LLM: {llm_label}{llm_time_str}")

        # -- Current document section --
        lines.append("")
        if current_doc_id is not None:
            step_elapsed = time.perf_counter() - current_step_start if current_step_start > 0 else 0
            lines.append(f"Aktuell: #{current_doc_id} - {current_doc_title}")
            lines.append(f"Schritt: {current_step} ({step_elapsed:.0f}s)")
            # Progress bar
            bar_width = 30
            filled = int(bar_width * step_progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            lines.append(f"[{bar}] {step_progress * 100:.0f}%")
        elif current_step and not init_steps:
            lines.append(f"[dim]{current_step}[/dim]")

        # -- Recent results section --
        if recent_results:
            lines.append("")
            lines.append("Letzte Ergebnisse:")
            for status_icon, doc_id, title, duration in recent_results[-5:]:
                title_short = (title[:45] + "...") if len(title) > 48 else title
                lines.append(f"  {status_icon} #{doc_id} {title_short} ({duration:.1f}s)")

        return Panel(
            "\n".join(lines),
            title="[bold]Autopilot[/bold]",
            border_style="cyan",
            width=min(console.width, 70),
        )

    def _llm_preload(self) -> bool:
        """Try to preload the LLM model via Ollama keep_alive endpoint."""
        base_url = self.analyzer.url
        # Determine Ollama generate endpoint
        if "/api/chat" in base_url:
            generate_url = base_url.replace("/api/chat", "/api/generate")
        elif "/v1/chat/completions" in base_url:
            # Not Ollama, try generic health check
            return self.analyzer.verify_connection()
        else:
            generate_url = base_url.rstrip("/") + "/api/generate"

        keep_alive = LLM_KEEP_ALIVE if LLM_KEEP_ALIVE else "10m"
        try:
            resp = requests.post(
                generate_url,
                json={"model": self.llm_model, "keep_alive": keep_alive},
                timeout=30,
            )
            if resp.status_code < 400:
                log.info(f"AUTOPILOT: LLM-Modell vorgeladen (keep_alive={keep_alive})")
                return True
        except requests.exceptions.RequestException as exc:
            log.warning(f"AUTOPILOT: LLM-Preload fehlgeschlagen: {exc}")
        return False

    def action_autopilot(self):
        """Vollautomatik: initial sortieren, dann dauerhaft live beobachten + periodische Wartung."""

        interval_sec = max(5, AUTOPILOT_INTERVAL_SEC)
        refresh_cycles = max(1, AUTOPILOT_CONTEXT_REFRESH_CYCLES)
        cleanup_every = max(0, AUTOPILOT_CLEANUP_EVERY_CYCLES)
        dupscan_every = max(0, AUTOPILOT_DUPLICATE_SCAN_EVERY_CYCLES)
        review_resolve_every = max(0, AUTOPILOT_REVIEW_RESOLVE_EVERY_CYCLES)
        max_new_per_cycle = max(0, AUTOPILOT_MAX_NEW_DOCS_PER_CYCLE)

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
        total_docs = 0

        # Smart recovery state
        consecutive_timeouts = 0
        llm_ok = False
        llm_last_response_time = 0.0
        llm_recovery_msg = ""

        # Dashboard state
        ap_start = time.perf_counter()
        current_doc_id: int | None = None
        current_doc_title = ""
        current_step = ""
        current_step_start = 0.0
        step_progress = 0.0
        recent_results: list[tuple[str, int, str, float]] = []
        spin_counter = 0

        _STEP_PROGRESS = {
            "Lade Dokument...": 0.05,
            "OCR-Check": 0.15,
            "Sprache": 0.20,
            "Muster": 0.25,
            "Regelbasiert erkannt": 0.35,
            "LLM-Analyse laeuft...": 0.40,
            "LLM-Antwort": 0.65,
            "Guardrails pruefen...": 0.75,
            "Tags auswaehlen...": 0.80,
            "Review noetig": 0.85,
            "Aenderungen anwenden...": 0.90,
            "Fertig": 1.0,
        }

        def _status_callback(msg: str):
            nonlocal current_step, current_step_start, step_progress
            current_step = msg
            current_step_start = time.perf_counter()
            for key, pct in _STEP_PROGRESS.items():
                if msg.startswith(key):
                    step_progress = pct
                    break

        # Init checklist
        init_steps: list[tuple[str, str]] = [
            ("Paperless-Verbindung", "pending"),
            ("Dokumente laden", "pending"),
            ("LLM-Verbindung", "pending"),
            ("LLM-Speedcheck", "pending"),
            ("Stammdaten laden", "pending"),
            ("Entscheidungskontext", "pending"),
            ("Initiale Sortierung", "pending"),
        ]
        phase = "Initialisierung"
        run_id: int | None = None
        known_ids: set[int] = set()

        # Suppress console RichHandler during Live display
        _rich_handler = None
        for h in log.parent.handlers if log.parent else []:
            if isinstance(h, RichHandler):
                _rich_handler = h
                break

        try:
            with Live(
                self._build_dashboard(
                    start_time=ap_start, cycle=0, interval_sec=interval_sec,
                    total_docs=0, total_candidates=0,
                    total_updated=0, total_doc_failures=0,
                    total_reviews_resolved=0,
                    llm_model=self.llm_model or "auto", llm_ok=False,
                    llm_last_time=0, llm_recovery_msg="",
                    current_doc_id=None, current_doc_title="", current_step="",
                    current_step_start=0, step_progress=0, recent_results=[],
                    phase=phase, init_steps=init_steps,
                ),
                console=console,
                refresh_per_second=2,
                transient=True,
            ) as live:
                # Mute console handler so log lines don't break the Live display
                if _rich_handler:
                    _rich_handler.setLevel(logging.CRITICAL)

                def _refresh():
                    nonlocal spin_counter
                    spin_counter += 1
                    live.update(self._build_dashboard(
                        start_time=ap_start, cycle=cycle, interval_sec=interval_sec,
                        total_docs=total_docs, total_candidates=total_candidates,
                        total_updated=total_updated, total_doc_failures=total_doc_failures,
                        total_reviews_resolved=total_reviews_resolved,
                        llm_model=self.llm_model or "auto", llm_ok=llm_ok,
                        llm_last_time=llm_last_response_time, llm_recovery_msg=llm_recovery_msg,
                        current_doc_id=current_doc_id, current_doc_title=current_doc_title,
                        current_step=current_step, current_step_start=current_step_start,
                        step_progress=step_progress, recent_results=recent_results,
                        phase=phase, init_steps=init_steps if phase == "Initialisierung" else None,
                        spin_idx=spin_counter,
                    ))

                def _status_cb_with_refresh(msg: str):
                    _status_callback(msg)
                    _refresh()

                def _set_init(idx: int, status: str):
                    init_steps[idx] = (init_steps[idx][0], status)

                def _finish_init_step(idx: int, min_duration: float = 0.5):
                    """Wait with spinner animation until at least min_duration has passed, then mark done."""
                    start = time.perf_counter()
                    while time.perf_counter() - start < min_duration:
                        _refresh()
                        time.sleep(0.08)
                    init_steps[idx] = (init_steps[idx][0], "done")
                    _refresh()

                # -- Step 0: Paperless connection --
                _set_init(0, "active")
                _refresh()
                if not self._init_paperless():
                    _set_init(0, "error")
                    _refresh()
                    return
                _finish_init_step(0)

                # -- Step 1: Load documents --
                _set_init(1, "active")
                _refresh()
                log.info("AUTOPILOT: Lade Dokumente...")
                try:
                    docs = self.paperless.get_documents()
                except Exception as exc:
                    _set_init(1, "error")
                    _refresh()
                    log.error(f"AUTOPILOT Initialisierung fehlgeschlagen: {exc}")
                    return
                known_ids = {int(d["id"]) for d in docs if d.get("id") is not None}
                total_docs = len(docs)
                init_steps[1] = (f"Dokumente laden ({total_docs})", "active")
                _finish_init_step(1)

                # -- Step 2: LLM connection --
                _set_init(2, "active")
                _refresh()
                if not self._init_analyzer():
                    _set_init(2, "error")
                    llm_ok = False
                else:
                    llm_ok = True
                _refresh()

                # LLM health check + recovery (only if init succeeded)
                if llm_ok:
                    log.info("AUTOPILOT: LLM-Gesundheitscheck...")
                    if not self.analyzer.verify_connection():
                        llm_ok = False
                if not llm_ok and self.analyzer:
                    llm_recovery_msg = "Nicht erreichbar - Recovery..."
                    _refresh()
                    log.warning("AUTOPILOT: LLM nicht erreichbar - versuche Recovery...")
                    self._llm_preload()
                    for attempt in range(1, 4):
                        wait_sec = attempt * 30
                        init_steps[2] = (f"LLM-Recovery {attempt}/3 ({wait_sec}s)", "active")
                        for remaining in range(wait_sec, 0, -5):
                            llm_recovery_msg = f"Recovery {attempt}/3 ({remaining}s...)"
                            _refresh()
                            time.sleep(5)
                        llm_recovery_msg = f"Health-Check {attempt}/3..."
                        _refresh()
                        if self.analyzer.verify_connection():
                            llm_ok = True
                            llm_recovery_msg = ""
                            log.info("AUTOPILOT: LLM-Recovery erfolgreich!")
                            break
                        self._llm_preload()
                    if not llm_ok:
                        llm_recovery_msg = "Nicht erreichbar (Fallback aktiv)"
                        init_steps[2] = ("LLM-Verbindung (Fallback)", "error")
                        _refresh()
                        log.warning("AUTOPILOT: LLM nicht erreichbar - Learning-Priors als Fallback")
                        consecutive_timeouts = 3
                    else:
                        init_steps[2] = (f"LLM-Verbindung ({self.llm_model})", "active")
                        _finish_init_step(2)
                elif llm_ok:
                    init_steps[2] = (f"LLM-Verbindung ({self.llm_model})", "active")
                    log.info("AUTOPILOT: LLM bereit")
                    _finish_init_step(2)

                # -- Step 3: LLM-Speedcheck --
                if llm_ok and LLM_SPEEDCHECK_ENABLED:
                    _set_init(3, "active")
                    init_steps[3] = ("LLM-Speedcheck laeuft...", "active")
                    _refresh()
                    sc_result = self._run_speedcheck(auto_switch=True)
                    if sc_result and sc_result.get("switched"):
                        init_steps[3] = (f"LLM-Speedcheck (gewechselt: {self.llm_model})", "active")
                    elif sc_result:
                        best_t = sc_result.get("best_time", 0)
                        init_steps[3] = (f"LLM-Speedcheck ({best_t:.1f}s)", "active")
                    _finish_init_step(3)
                else:
                    reason = "deaktiviert" if not LLM_SPEEDCHECK_ENABLED else "LLM nicht verfuegbar"
                    init_steps[3] = (f"LLM-Speedcheck ({reason})", "active")
                    _finish_init_step(3, min_duration=0.3)

                # Start run tracking
                run_id = self._start_run("autopilot")

                log.info(
                    "AUTOPILOT gestartet: interval=%ss | start_auto_organize=%s | cleanup_every=%s | dupscan_every=%s",
                    interval_sec,
                    "JA" if AUTOPILOT_START_WITH_AUTO_ORGANIZE else "NEIN",
                    cleanup_every,
                    dupscan_every,
                )

                # -- Initial auto-organize (inline, with dashboard) --
                if AUTOPILOT_START_WITH_AUTO_ORGANIZE:
                    # Step 4: Master data
                    _set_init(4, "active")
                    _refresh()
                    try:
                        tags, correspondents, doc_types, storage_paths = self._load_master_data()
                        self._ensure_taxonomy_tags(tags)
                        init_steps[4] = (
                            f"Stammdaten ({len(tags)} Tags, {len(correspondents)} Korr.)", "active"
                        )
                        _finish_init_step(4)

                        # Step 5: Decision context
                        _set_init(5, "active")
                        _refresh()
                        decision_context = self._collect_decision_context(correspondents, storage_paths)
                        ctx = self._build_ctx(tags, correspondents, doc_types, storage_paths,
                                              decision_context=decision_context, run_id=run_id)
                        _finish_init_step(5)

                        # Step 6: Scan & sort
                        _set_init(6, "active")
                        _refresh()
                        all_docs = self.paperless.get_documents()
                        total_docs = len(all_docs)

                        # Filter
                        duplikat_tag_id = next((t["id"] for t in tags if t["name"] == "Duplikat"), None)
                        if duplikat_tag_id:
                            all_docs = [d for d in all_docs if duplikat_tag_id not in (d.get("tags") or [])]
                        if AUTOPILOT_RECHECK_ALL_ON_START:
                            todo = list(all_docs)
                        else:
                            todo = [d for d in all_docs if not _is_fully_organized(d)]
                        todo.sort(key=lambda d: d.get("added", d.get("created", "")), reverse=True)

                        if todo:
                            init_steps[6] = (f"Initiale Sortierung ({len(todo)} Dok.)", "active")
                            phase = "Sortierung"
                            log.info(f"AUTOPILOT: Initiale Sortierung - {len(todo)} Dokumente")
                            current_step = f"Initiale Sortierung: 0/{len(todo)}"
                            _refresh()

                            for i, doc in enumerate(todo, 1):
                                current_doc_id = doc["id"]
                                current_doc_title = doc.get("title", "Unbekannt")[:50]
                                current_step = f"Initiale Sortierung: {i}/{len(todo)}"
                                step_progress = 0.0
                                current_step_start = time.perf_counter()
                                _refresh()

                                total_candidates += 1
                                doc_t0 = time.perf_counter()
                                is_timeout = False
                                try:
                                    if process_document(
                                        doc["id"], ctx, self.dry_run,
                                        batch_mode=not self.dry_run,
                                        prefer_compact=LIVE_WATCH_COMPACT_FIRST,
                                        status_callback=_status_cb_with_refresh,
                                        quiet=True,
                                    ):
                                        total_updated += 1
                                        doc_dur = time.perf_counter() - doc_t0
                                        llm_last_response_time = doc_dur
                                        llm_ok = True
                                        llm_recovery_msg = ""
                                        consecutive_timeouts = 0
                                        recent_results.append(("[green]✓[/green]", doc["id"], current_doc_title, doc_dur))
                                    else:
                                        total_doc_failures += 1
                                        doc_dur = time.perf_counter() - doc_t0
                                        if "timeout" in current_step.lower() or "nicht erreichbar" in current_step.lower():
                                            is_timeout = True
                                        recent_results.append(("[yellow]–[/yellow]", doc["id"], current_doc_title, doc_dur))
                                except requests.exceptions.Timeout:
                                    is_timeout = True
                                    total_doc_failures += 1
                                    doc_dur = time.perf_counter() - doc_t0
                                    recent_results.append(("[red]✗[/red]", doc["id"], "Timeout", doc_dur))
                                    log.error(f"AUTOPILOT Timeout bei Dokument #{doc['id']}")
                                except requests.exceptions.ConnectionError:
                                    is_timeout = True
                                    total_doc_failures += 1
                                    doc_dur = time.perf_counter() - doc_t0
                                    recent_results.append(("[red]✗[/red]", doc["id"], "Verbindungsfehler", doc_dur))
                                    log.error(f"AUTOPILOT Verbindungsfehler bei #{doc['id']}")
                                except Exception as exc:
                                    total_doc_failures += 1
                                    doc_dur = time.perf_counter() - doc_t0
                                    err_str = str(exc).lower()
                                    if "timeout" in err_str or "timed out" in err_str:
                                        is_timeout = True
                                    recent_results.append(("[red]✗[/red]", doc["id"], str(exc)[:40], doc_dur))
                                    log.error(f"AUTOPILOT Fehler bei #{doc['id']}: {exc}")

                                if is_timeout:
                                    consecutive_timeouts += 1
                                    llm_ok = False
                                else:
                                    consecutive_timeouts = 0
                                    llm_ok = True
                                    llm_recovery_msg = ""
                                if len(recent_results) > 20:
                                    recent_results = recent_results[-20:]
                                _refresh()

                            init_steps[6] = (f"Initiale Sortierung ({total_updated} aktualisiert)", "done")
                            log.info(f"AUTOPILOT: Initiale Sortierung fertig - {total_updated} aktualisiert")
                        else:
                            init_steps[6] = ("Initiale Sortierung (alles sortiert)", "done")
                            log.info("AUTOPILOT: Alle Dokumente bereits sortiert")

                        # Refresh known IDs after initial organize
                        try:
                            docs = self.paperless.get_documents()
                            known_ids = {int(d["id"]) for d in docs if d.get("id") is not None}
                            total_docs = len(docs)
                        except Exception as exc:
                            total_poll_errors += 1
                            log.warning(f"AUTOPILOT: Re-Load nach Initialsortierung fehlgeschlagen: {exc}")
                    except Exception as exc:
                        total_poll_errors += 1
                        _set_init(4, "error")
                        log.error(f"AUTOPILOT: Initiale Sortierung fehlgeschlagen: {exc}")
                else:
                    # Skip init steps 4-6 when auto-organize is off
                    for idx in (4, 5, 6):
                        init_steps[idx] = (init_steps[idx][0] + " (uebersprungen)", "active")
                        _finish_init_step(idx, min_duration=0.3)

                # -- Switch to watch phase --
                phase = "Watch"
                current_doc_id = None
                current_step = ""
                _refresh()

                # -- Main polling loop --
                while True:
                    cycle += 1
                    current_doc_id = None
                    current_doc_title = ""
                    current_step = "Polling..."
                    step_progress = 0.0
                    _refresh()

                    # Quiet hours: pause processing to reduce server load
                    if _is_quiet_hours():
                        current_step = f"Ruhestunden ({QUIET_HOURS_START}:00-{QUIET_HOURS_END}:00)"
                        _refresh()
                        if cycle == 1 or cycle % 20 == 0:
                            log.info(f"AUTOPILOT: Ruhestunden aktiv ({QUIET_HOURS_START}:00-{QUIET_HOURS_END}:00), pausiert...")
                        time.sleep(60)
                        continue

                    # --- Smart LLM Recovery ---
                    if consecutive_timeouts >= 5:
                        llm_recovery_msg = "Pause (5min Recovery...)"
                        llm_ok = False
                        _refresh()
                        log.warning(f"AUTOPILOT: {consecutive_timeouts} Timeouts - lange Pause (5min)")
                        for remaining in range(300, 0, -5):
                            llm_recovery_msg = f"Pause (Recovery in {remaining}s...)"
                            _refresh()
                            time.sleep(5)
                        llm_recovery_msg = "Modell vorladen..."
                        _refresh()
                        self._llm_preload()
                        if self.analyzer.verify_connection():
                            llm_ok = True
                            consecutive_timeouts = 0
                            llm_recovery_msg = ""
                            log.info("AUTOPILOT: LLM-Recovery nach 5min Pause erfolgreich")
                        else:
                            llm_recovery_msg = "LLM nicht erreichbar"
                            log.error("AUTOPILOT: LLM-Recovery nach 5min fehlgeschlagen")
                        _refresh()
                    elif consecutive_timeouts >= 3:
                        llm_recovery_msg = "Pause (60s Recovery...)"
                        llm_ok = False
                        _refresh()
                        log.warning(f"AUTOPILOT: {consecutive_timeouts} Timeouts - kurze Pause (60s)")
                        for remaining in range(60, 0, -5):
                            llm_recovery_msg = f"Pause (Recovery in {remaining}s...)"
                            _refresh()
                            time.sleep(5)
                        llm_recovery_msg = "Health-Check..."
                        _refresh()
                        if self.analyzer.verify_connection():
                            llm_ok = True
                            consecutive_timeouts = 0
                            llm_recovery_msg = ""
                            log.info("AUTOPILOT: LLM-Recovery nach 60s Pause erfolgreich")
                        else:
                            llm_recovery_msg = "Modell vorladen..."
                            _refresh()
                            self._llm_preload()
                            log.info("AUTOPILOT: LLM-Preload versucht, warte 120s...")
                            for remaining in range(120, 0, -5):
                                llm_recovery_msg = f"Modell laden ({remaining}s...)"
                                _refresh()
                                time.sleep(5)
                            if self.analyzer.verify_connection():
                                llm_ok = True
                                consecutive_timeouts = 0
                                llm_recovery_msg = ""
                                log.info("AUTOPILOT: LLM-Recovery nach Preload erfolgreich")
                            else:
                                llm_recovery_msg = "LLM nicht erreichbar"
                                log.error("AUTOPILOT: LLM-Recovery nach Preload fehlgeschlagen")
                        _refresh()

                    try:
                        docs = self.paperless.get_documents()
                    except Exception as exc:
                        total_poll_errors += 1
                        consecutive_poll_errors += 1
                        backoff = min(
                            WATCH_ERROR_BACKOFF_MAX_SEC,
                            interval_sec + (consecutive_poll_errors - 1) * max(1, WATCH_ERROR_BACKOFF_BASE_SEC),
                        )
                        current_step = f"Poll-Fehler (retry in {backoff}s)"
                        _refresh()
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
                    total_docs = len(docs)

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

                        # Smart scheduling: prioritize time-sensitive documents
                        def _doc_priority(did: int) -> int:
                            d = docs_by_id.get(did, {})
                            hints = _detect_content_hints(d)
                            if "invoice_detected" in hints or "iban_present" in hints:
                                return 0  # Invoices first (time-sensitive)
                            if "contract_detected" in hints or "insurance_detected" in hints:
                                return 1  # Contracts/insurance second
                            return 2  # Everything else
                        new_ids = sorted(new_ids, key=_doc_priority)

                        try:
                            tags, correspondents, doc_types, storage_paths = self._load_master_data()
                            self._ensure_taxonomy_tags(tags)
                            if decision_context is None or (cycle % refresh_cycles == 0):
                                decision_context = self._collect_decision_context(correspondents, storage_paths)
                            ctx = self._build_ctx(tags, correspondents, doc_types, storage_paths,
                                                  decision_context=decision_context, run_id=run_id)
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
                            current_doc_id = doc_id
                            current_doc_title = doc.get("title", "Unbekannt")[:50]
                            current_step = "Starte..."
                            step_progress = 0.0
                            current_step_start = time.perf_counter()
                            _refresh()

                            doc_t0 = time.perf_counter()
                            is_timeout = False
                            try:
                                if process_document(
                                    doc_id, ctx, self.dry_run,
                                    batch_mode=not self.dry_run,
                                    prefer_compact=LIVE_WATCH_COMPACT_FIRST,
                                    status_callback=_status_cb_with_refresh,
                                    quiet=True,
                                ):
                                    total_updated += 1
                                    doc_dur = time.perf_counter() - doc_t0
                                    llm_last_response_time = doc_dur
                                    llm_ok = True
                                    llm_recovery_msg = ""
                                    consecutive_timeouts = 0
                                    recent_results.append(("[green]✓[/green]", doc_id, current_doc_title, doc_dur))
                                else:
                                    total_doc_failures += 1
                                    doc_dur = time.perf_counter() - doc_t0
                                    # Check if the step was a timeout indicator
                                    if "timeout" in current_step.lower() or "nicht erreichbar" in current_step.lower():
                                        is_timeout = True
                                    recent_results.append(("[yellow]–[/yellow]", doc_id, current_doc_title, doc_dur))
                            except requests.exceptions.Timeout:
                                is_timeout = True
                                total_doc_failures += 1
                                doc_dur = time.perf_counter() - doc_t0
                                recent_results.append(("[red]✗[/red]", doc_id, "Timeout", doc_dur))
                                log.error(f"AUTOPILOT Timeout bei Dokument #{doc_id}")
                            except requests.exceptions.ConnectionError:
                                is_timeout = True
                                total_doc_failures += 1
                                doc_dur = time.perf_counter() - doc_t0
                                recent_results.append(("[red]✗[/red]", doc_id, "Verbindungsfehler", doc_dur))
                                log.error(f"AUTOPILOT Verbindungsfehler bei Dokument #{doc_id}")
                            except Exception as exc:
                                total_doc_failures += 1
                                total_poll_errors += 1
                                doc_dur = time.perf_counter() - doc_t0
                                err_str = str(exc).lower()
                                if "timeout" in err_str or "timed out" in err_str:
                                    is_timeout = True
                                recent_results.append(("[red]✗[/red]", doc_id, str(exc)[:40], doc_dur))
                                log.error(f"AUTOPILOT Fehler bei Dokument #{doc_id}: {exc}")

                            if is_timeout:
                                consecutive_timeouts += 1
                                llm_ok = False
                            else:
                                consecutive_timeouts = 0
                                llm_ok = True
                                llm_recovery_msg = ""

                            # Keep only last 20 results
                            if len(recent_results) > 20:
                                recent_results = recent_results[-20:]
                            _refresh()

                    elif cycle == 1 or cycle % 5 == 0:
                        current_step = f"Keine neuen Dokumente, warte {interval_sec}s..."
                        _refresh()
                        log.info(f"AUTOPILOT: keine neuen Dokumente, warte {interval_sec}s...")

                    # -- Periodic speedcheck --
                    speedcheck_every = LLM_SPEEDCHECK_INTERVAL_CYCLES
                    if speedcheck_every > 0 and cycle % speedcheck_every == 0 and llm_ok:
                        try:
                            current_step = "Periodischer LLM-Speedcheck..."
                            _refresh()
                            sc_result = self._run_speedcheck(auto_switch=True)
                            if sc_result and sc_result.get("switched"):
                                log.info(f"AUTOPILOT: Speedcheck - Modell gewechselt auf {self.llm_model}")
                        except Exception as exc:
                            log.warning(f"AUTOPILOT: Periodischer Speedcheck fehlgeschlagen: {exc}")

                    # -- Periodic maintenance --
                    if cleanup_every > 0 and cycle % cleanup_every == 0:
                        try:
                            current_step = "Cleanup laeuft..."
                            _refresh()
                            log.info("AUTOPILOT: periodisches Cleanup gestartet")
                            cleanup_tags(self.paperless, self.dry_run)
                            cleanup_correspondents(self.paperless, self.dry_run)
                            cleanup_document_types(self.paperless, self.dry_run)
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
                            current_step = "Duplikat-Scan..."
                            _refresh()
                            log.info("AUTOPILOT: periodischer Duplikat-Scan gestartet")
                            find_duplicates(self.paperless)
                            total_duplicate_scans += 1
                        except Exception as exc:
                            total_poll_errors += 1
                            log.error(f"AUTOPILOT Duplikat-Scan-Fehler: {exc}")

                    if review_resolve_every > 0 and cycle % review_resolve_every == 0:
                        try:
                            current_step = "Review-Queue pruefen..."
                            _refresh()
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

                    # Periodic log summary (dashboard replaces console table)
                    if cycle % 10 == 0:
                        uptime_min = (time.perf_counter() - ap_start) / 60
                        success_rate = (total_updated / total_candidates * 100) if total_candidates else 0
                        log.info(
                            f"AUTOPILOT Status Zyklus {cycle}: {uptime_min:.0f}min Laufzeit, "
                            f"{total_updated}/{total_candidates} aktualisiert ({success_rate:.0f}%), "
                            f"{total_doc_failures} Fehler, {total_poll_errors} Poll-Fehler"
                        )

                    current_doc_id = None
                    current_step = f"Warte {interval_sec}s..."
                    step_progress = 0.0
                    _refresh()
                    time.sleep(interval_sec)

        except KeyboardInterrupt:
            log.info("AUTOPILOT vom Benutzer gestoppt")
        finally:
            # Restore console log handler
            if _rich_handler:
                _rich_handler.setLevel(logging.NOTSET)

            # Print final summary to console
            elapsed_total = time.perf_counter() - ap_start
            elapsed_min = elapsed_total / 60
            success_rate = (total_updated / total_candidates * 100) if total_candidates else 0
            summary_table = Table(title="Autopilot Zusammenfassung", show_header=True)
            summary_table.add_column("Metrik", style="cyan")
            summary_table.add_column("Wert", style="green", justify="right")
            summary_table.add_row("Laufzeit", f"{elapsed_min:.1f} min")
            summary_table.add_row("Zyklen", str(cycle))
            summary_table.add_row("Neue Dokumente", str(total_seen_new))
            summary_table.add_row("Kandidaten", str(total_candidates))
            summary_table.add_row("Aktualisiert", str(total_updated))
            summary_table.add_row("Erfolgsrate", f"{success_rate:.0f}%")
            summary_table.add_row("Fehler", str(total_doc_failures))
            summary_table.add_row("Wartungslaeufe", str(total_maintenance_runs))
            summary_table.add_row("Reviews geloest", str(total_reviews_resolved))
            summary_table.add_row("Poll-Fehler", str(total_poll_errors))
            console.print(summary_table)

            if run_id is not None:
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
        llm_model_label = self.llm_model or "auto/server-default"
        choice = self._menu("Einstellungen", [
            ("1", f"Modus umschalten (aktuell: {'TESTLAUF' if self.dry_run else 'LIVE'})"),
            ("2", f"LLM-Modell aendern (aktuell: {llm_model_label})"),
            ("3", f"LLM-URL aendern (aktuell: {self.llm_url})"),
            ("4", "LLM-Verbindung testen"),
            ("5", f"Neue Tags erlauben (aktuell: {'JA' if _cfg.ALLOW_NEW_TAGS else 'NEIN'})"),
            ("6", f"Auto-Cleanup nach Auto/Batches (aktuell: {'JA' if _cfg.AUTO_CLEANUP_AFTER_ORGANIZE else 'NEIN'})"),
            ("7", f"Web-Hinweise aktiv (aktuell: {'JA' if _cfg.ENABLE_WEB_HINTS else 'NEIN'})"),
            ("8", "Taxonomie neu laden"),
            ("9", f"Taxonomie-Tags auto-erstellen (aktuell: {'JA' if _cfg.AUTO_CREATE_TAXONOMY_TAGS else 'NEIN'})"),
            ("10", f"Menue 1: alle Dokumente neu pruefen (aktuell: {'JA' if _cfg.RECHECK_ALL_DOCS_IN_AUTO else 'NEIN'})"),
            ("11", "Learning-Daten sichern (Backup)"),
            ("12", "Datenbank bereinigen (alte Runs loeschen)"),
            ("13", "Verarbeitungshistorie exportieren (CSV)"),
            ("14", "Learning-Daten Integritaetspruefung"),
            ("15", "LLM-Speedcheck (alle Modelle benchmarken)"),
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
            if self._init_analyzer():
                check = self.analyzer.speedcheck()
                if check["ok"]:
                    console.print(f"  [green]Speedcheck:[/green] {check['response_time']:.1f}s ({check['model']})")
                else:
                    console.print(f"  [yellow]Speedcheck fehlgeschlagen:[/yellow] {check['error']}")

        elif choice == "5":
            _cfg.ALLOW_NEW_TAGS = not _cfg.ALLOW_NEW_TAGS
            console.print(f"[green]Neue Tags {'aktiviert' if _cfg.ALLOW_NEW_TAGS else 'deaktiviert'}.[/green]")

        elif choice == "6":
            _cfg.AUTO_CLEANUP_AFTER_ORGANIZE = not _cfg.AUTO_CLEANUP_AFTER_ORGANIZE
            console.print(f"[green]Auto-Cleanup {'aktiviert' if _cfg.AUTO_CLEANUP_AFTER_ORGANIZE else 'deaktiviert'}.[/green]")

        elif choice == "7":
            _cfg.ENABLE_WEB_HINTS = not _cfg.ENABLE_WEB_HINTS
            console.print(f"[green]Web-Hinweise {'aktiviert' if _cfg.ENABLE_WEB_HINTS else 'deaktiviert'}.[/green]")

        elif choice == "8":
            self.taxonomy.load()
            console.print(f"[green]Taxonomie neu geladen ({len(self.taxonomy.canonical_tags)} Tags).[/green]")

        elif choice == "9":
            _cfg.AUTO_CREATE_TAXONOMY_TAGS = not _cfg.AUTO_CREATE_TAXONOMY_TAGS
            console.print(f"[green]Taxonomie-Tag-Autocreate {'aktiviert' if _cfg.AUTO_CREATE_TAXONOMY_TAGS else 'deaktiviert'}.[/green]")

        elif choice == "10":
            _cfg.RECHECK_ALL_DOCS_IN_AUTO = not _cfg.RECHECK_ALL_DOCS_IN_AUTO
            console.print(f"[green]Auto-Recheck {'aktiviert' if _cfg.RECHECK_ALL_DOCS_IN_AUTO else 'deaktiviert'}.[/green]")

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

        elif choice == "13":
            import csv as csv_mod
            export_path = os.path.join(LOG_DIR, f"processing_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            try:
                with self.run_db._lock, self.run_db._connect() as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT d.doc_id, d.status, d.title_before, d.title_after, "
                        "d.correspondent_before, d.correspondent_after, d.error_text, d.created_at, "
                        "r.action, r.llm_model "
                        "FROM documents d LEFT JOIN runs r ON d.run_id = r.id "
                        "ORDER BY d.created_at DESC LIMIT 5000"
                    ).fetchall()
                with open(export_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv_mod.writer(f)
                    writer.writerow(["doc_id", "status", "title_before", "title_after",
                                     "correspondent_before", "correspondent_after",
                                     "error", "created_at", "action", "llm_model"])
                    for row in rows:
                        writer.writerow([
                            row["doc_id"], row["status"], row["title_before"], row["title_after"],
                            row["correspondent_before"], row["correspondent_after"],
                            row["error_text"], row["created_at"], row["action"], row["llm_model"],
                        ])
                console.print(f"[green]Exportiert: {export_path} ({len(rows)} Eintraege)[/green]")
            except Exception as exc:
                console.print(f"[red]Export-Fehler: {exc}[/red]")

        elif choice == "14":
            result = self.learning_examples.validate()
            s = result["stats"]
            table = Table(title="Learning-Daten Integritaet", show_header=True, width=50)
            table.add_column("Metrik", style="cyan")
            table.add_column("Wert", justify="right", style="bold")
            table.add_row("Eintraege gesamt", str(s["total"]))
            table.add_row("Valide", f"[green]{s['valid']}[/green]")
            table.add_row("Ungültiges JSON", f"[red]{s['invalid_json']}[/red]" if s["invalid_json"] else "0")
            table.add_row("Abgelehnte (Anti-Pattern)", str(s["rejected"]))
            table.add_row("Fehlende Felder", str(s["missing_fields"]))
            table.add_row("Duplikate", f"[yellow]{s['duplicates']}[/yellow]" if s["duplicates"] else "0")
            table.add_row("Korrespondenten", str(s["correspondents"]))
            console.print(table)
            if result["issues"]:
                for issue in result["issues"][:10]:
                    console.print(f"  [yellow]Problem:[/yellow] {issue}")
            else:
                console.print("[green]Keine Probleme gefunden.[/green]")

        elif choice == "15":
            if not self._init_analyzer():
                return
            console.print("[cyan]Benchmarke alle verfuegbaren Modelle...[/cyan]")
            benchmark = self.analyzer.benchmark_available_models(
                max_acceptable_time=LLM_SPEEDCHECK_MAX_TIME,
                timeout_per_model=LLM_SPEEDCHECK_TIMEOUT,
            )
            table = Table(title="LLM-Speedcheck Ergebnisse", show_header=True)
            table.add_column("Modell", style="cyan")
            table.add_column("Zeit", justify="right")
            table.add_column("Status")
            table.add_column("", width=3)
            for r in sorted(benchmark["results"], key=lambda x: x["response_time"] if x["ok"] else 999):
                if r["ok"]:
                    time_str = f"{r['response_time']:.1f}s"
                    if r["response_time"] <= LLM_SPEEDCHECK_MAX_TIME:
                        status = "[green]OK[/green]"
                    else:
                        status = "[yellow]Langsam[/yellow]"
                else:
                    time_str = "-"
                    status = f"[red]{r['error']}[/red]"
                marker = "[bold]*[/bold]" if r.get("is_current") else ""
                table.add_row(r["model"], time_str, status, marker)
            console.print(table)
            console.print(f"[dim]* = aktuelles Modell | Limit: {LLM_SPEEDCHECK_MAX_TIME}s[/dim]")

            best = benchmark["best_model"]
            best_time = benchmark["best_time"]
            if best and best != self.analyzer.model:
                if Confirm.ask(f"Auf schnellstes Modell wechseln ({best}, {best_time:.1f}s)?", default=True):
                    self.analyzer.model = best
                    self.analyzer._original_model = best
                    self.llm_model = best
                    console.print(f"[green]Modell gewechselt: {best}[/green]")

    def menu_knowledge(self):
        """Untermenue: Wissensdatenbank."""
        if not self.knowledge_db:
            if not KNOWLEDGE_DB_URL:
                console.print("[yellow]KNOWLEDGE_DB_URL nicht in .env konfiguriert.[/yellow]")
                console.print("[dim]Beispiel: KNOWLEDGE_DB_URL=postgresql://admin:admin@192.168.178.118:5434/paperless_knowledge[/dim]")
                return
            if not self._init_knowledge_db():
                console.print("[red]Verbindung zur Knowledge-DB fehlgeschlagen.[/red]")
                return

        choice = self._menu("Wissensdatenbank", [
            ("1", "Alle Fakten anzeigen"),
            ("2", "Fakten nach Typ filtern"),
            ("3", "Entitaeten anzeigen"),
            ("4", "Zeitstrahl (letzte Ereignisse)"),
            ("5", "Fakt manuell hinzufuegen"),
            ("6", "Fakt deaktivieren"),
            ("7", "Migration von learning_profile.json"),
            ("8", "Prompt-Kontext Vorschau"),
            ("9", "Statistiken"),
            ("0", "Zurueck"),
        ])

        if choice == "1":
            self._kb_show_facts()
        elif choice == "2":
            self._kb_show_facts_by_type()
        elif choice == "3":
            self._kb_show_entities()
        elif choice == "4":
            self._kb_show_timeline()
        elif choice == "5":
            self._kb_add_manual_fact()
        elif choice == "6":
            self._kb_deactivate_fact()
        elif choice == "7":
            self._kb_run_migration()
        elif choice == "8":
            self._kb_show_prompt_preview()
        elif choice == "9":
            self._kb_show_statistics()

    def _kb_show_facts(self):
        """Show all current facts."""
        facts = self.knowledge_db.get_current_facts()
        if not facts:
            console.print("[dim]Keine Fakten in der Datenbank.[/dim]")
            return
        table = Table(title=f"Aktive Fakten ({len(facts)})", show_header=True)
        table.add_column("ID", style="dim", width=5)
        table.add_column("Typ", style="cyan", width=20)
        table.add_column("Entity", style="green", width=25)
        table.add_column("Zusammenfassung", width=45)
        table.add_column("Gueltig", width=22)
        table.add_column("Konf.", justify="right", width=5)
        for f in facts[:100]:
            valid = ""
            if f["valid_from"]:
                valid = str(f["valid_from"])
            if f["valid_until"]:
                valid += f" - {f['valid_until']}"
            conf_str = f"{f['confidence']:.0%}" if f["confidence"] else ""
            table.add_row(
                str(f["id"]), f["fact_type"],
                f["entity_name"] or "-", f["summary"][:45],
                valid, conf_str,
            )
        console.print(table)

    def _kb_show_facts_by_type(self):
        """Show facts filtered by type."""
        from .knowledge import VALID_FACT_TYPES
        types_sorted = sorted(VALID_FACT_TYPES)
        console.print("[cyan]Verfuegbare Faktentypen:[/cyan]")
        for i, ft in enumerate(types_sorted, 1):
            console.print(f"  {i}. {ft}")
        idx_str = Prompt.ask("Typ-Nummer", default="1")
        try:
            idx = max(0, int(idx_str) - 1)
            ft = types_sorted[idx]
        except (ValueError, IndexError):
            console.print("[red]Ungueltige Auswahl.[/red]")
            return
        facts = self.knowledge_db.get_current_facts(fact_type=ft)
        if not facts:
            console.print(f"[dim]Keine Fakten vom Typ '{ft}'.[/dim]")
            return
        table = Table(title=f"Fakten: {ft} ({len(facts)})", show_header=True)
        table.add_column("ID", style="dim", width=5)
        table.add_column("Entity", style="green", width=25)
        table.add_column("Zusammenfassung", width=50)
        table.add_column("Gueltig", width=22)
        table.add_column("Konf.", justify="right", width=5)
        for f in facts:
            valid = ""
            if f["valid_from"]:
                valid = str(f["valid_from"])
            if f["valid_until"]:
                valid += f" - {f['valid_until']}"
            conf_str = f"{f['confidence']:.0%}" if f["confidence"] else ""
            table.add_row(
                str(f["id"]), f["entity_name"] or "-",
                f["summary"][:50], valid, conf_str,
            )
        console.print(table)

    def _kb_show_entities(self):
        """Show all entities."""
        entities = self.knowledge_db.get_all_entities()
        if not entities:
            console.print("[dim]Keine Entitaeten in der Datenbank.[/dim]")
            return
        table = Table(title=f"Entitaeten ({len(entities)})", show_header=True)
        table.add_column("ID", style="dim", width=5)
        table.add_column("Typ", style="cyan", width=14)
        table.add_column("Name", style="green", width=40)
        table.add_column("Attribute", width=30)
        for e in entities:
            attrs = json.dumps(e["attributes"], ensure_ascii=False)[:30] if e["attributes"] else ""
            table.add_row(str(e["id"]), e["entity_type"], e["name"], attrs)
        console.print(table)

    def _kb_show_timeline(self):
        """Show fact timeline."""
        timeline = self.knowledge_db.get_fact_timeline(limit=30)
        if not timeline:
            console.print("[dim]Keine Ereignisse.[/dim]")
            return
        table = Table(title="Zeitstrahl (letzte 30 Ereignisse)", show_header=True)
        table.add_column("Datum", style="cyan", width=12)
        table.add_column("Typ", width=20)
        table.add_column("Entity", style="green", width=20)
        table.add_column("Zusammenfassung", width=45)
        table.add_column("Aktiv", width=5)
        for f in timeline:
            d = str(f["valid_from"]) if f["valid_from"] else str(f["created_at"])[:10]
            active = "[green]ja[/green]" if f["is_current"] else "[dim]nein[/dim]"
            table.add_row(
                d, f["fact_type"],
                f["entity_name"] or "-",
                f["summary"][:45], active,
            )
        console.print(table)

    def _kb_add_manual_fact(self):
        """Add a fact manually."""
        from .knowledge import VALID_ENTITY_TYPES, VALID_FACT_TYPES
        entity_type = Prompt.ask("Entity-Typ", default="company",
                                 choices=sorted(VALID_ENTITY_TYPES))
        entity_name = Prompt.ask("Entity-Name")
        if not entity_name:
            return
        fact_type = Prompt.ask("Fakten-Typ", default="note",
                               choices=sorted(VALID_FACT_TYPES))
        summary = Prompt.ask("Zusammenfassung")
        if not summary:
            return
        valid_from = Prompt.ask("Gueltig ab (YYYY-MM-DD, leer=unbekannt)", default="")
        valid_until = Prompt.ask("Gueltig bis (YYYY-MM-DD, leer=offen)", default="")

        fact_id = self.knowledge_db.store_manual_fact(
            fact_type=fact_type,
            entity_type=entity_type,
            entity_name=entity_name,
            summary=summary,
            valid_from=valid_from or None,
            valid_until=valid_until or None,
        )
        if fact_id:
            console.print(f"[green]Fakt #{fact_id} gespeichert.[/green]")
        else:
            console.print("[yellow]Fakt wurde nicht gespeichert (Duplikat oder niedrige Konfidenz).[/yellow]")

    def _kb_deactivate_fact(self):
        """Deactivate a fact by ID."""
        id_str = Prompt.ask("Fakt-ID zum Deaktivieren")
        try:
            fact_id = int(id_str)
        except ValueError:
            console.print("[red]Ungueltige ID.[/red]")
            return
        if Confirm.ask(f"Fakt #{fact_id} wirklich deaktivieren?"):
            self.knowledge_db.deactivate_fact(fact_id)
            console.print(f"[green]Fakt #{fact_id} deaktiviert.[/green]")

    def _kb_run_migration(self):
        """Run migration from learning_profile.json."""
        if self.knowledge_db.is_migrated():
            if not Confirm.ask("[yellow]Migration wurde bereits durchgefuehrt. Erneut ausfuehren?[/yellow]"):
                return
        console.print("[cyan]Migriere learning_profile.json -> Knowledge-DB...[/cyan]")
        try:
            stats = self.knowledge_db.migrate_from_learning_profile(self.learning_profile.data)
            table = Table(title="Migration Ergebnis", show_header=True, width=40)
            table.add_column("Metrik", style="cyan")
            table.add_column("Wert", justify="right", style="bold")
            table.add_row("Entitaeten erstellt", str(stats["entities"]))
            table.add_row("Fakten erstellt", str(stats["facts"]))
            table.add_row("Uebersprungen", str(stats["skipped"]))
            table.add_row("OCR-Merges", str(stats["merged_ocr"]))
            console.print(table)
        except Exception as exc:
            console.print(f"[red]Migration fehlgeschlagen: {exc}[/red]")

    def _kb_show_prompt_preview(self):
        """Show the generated prompt context."""
        owner = self.learning_profile.data.get("owner", "Document Owner")
        from . import config as _kcfg
        max_len = getattr(_kcfg, "KNOWLEDGE_PROMPT_MAX_LENGTH", 400)
        context = self.knowledge_db.build_prompt_context(owner, max_len=max_len)
        console.print(Panel(
            context,
            title=f"Prompt-Kontext (max {max_len} Zeichen, aktuell {len(context)})",
            border_style="green",
        ))

    def _kb_show_statistics(self):
        """Show knowledge DB statistics."""
        stats = self.knowledge_db.get_statistics()
        table = Table(title="Wissensdatenbank", show_header=True, width=45)
        table.add_column("Metrik", style="cyan")
        table.add_column("Wert", justify="right", style="bold")
        table.add_row("Entitaeten", str(stats["entities"]))
        table.add_row("Aktive Fakten", str(stats["active_facts"]))
        table.add_row("Ersetzte Fakten", str(stats["superseded_facts"]))
        table.add_row("Dokumente mit Fakten", str(stats["docs_with_facts"]))
        table.add_row("Beziehungen", str(stats["relations"]))
        console.print(table)
        if stats["entity_types"]:
            et_table = Table(title="Entitaeten nach Typ", show_header=True, width=35)
            et_table.add_column("Typ", style="cyan")
            et_table.add_column("Anzahl", justify="right")
            for et, count in stats["entity_types"].items():
                et_table.add_row(et, str(count))
            console.print(et_table)
        if stats["fact_types"]:
            ft_table = Table(title="Fakten nach Typ", show_header=True, width=35)
            ft_table.add_column("Typ", style="cyan")
            ft_table.add_column("Anzahl", justify="right")
            for ft, count in stats["fact_types"].items():
                ft_table.add_row(ft, str(count))
            console.print(ft_table)

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
                show_statistics(self.paperless, run_db=self.run_db)
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
        n_examples = len(self.learning_examples._examples) if self.learning_examples else 0
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
        table.add_column("Alter", width=8)
        table.add_column("Aktualisiert", width=20)
        for item in open_items:
            # Age color-coding: green <7d, yellow 7-30d, red >30d
            age_days = 0
            try:
                created = datetime.fromisoformat(item["created_at"])
                age_days = (datetime.now() - created).days
            except (ValueError, TypeError):
                pass
            if age_days > 30:
                age_str = f"[red]{age_days}d[/red]"
            elif age_days > 7:
                age_str = f"[yellow]{age_days}d[/yellow]"
            else:
                age_str = f"[green]{age_days}d[/green]"
            table.add_row(str(item["id"]), str(item["doc_id"]), item["reason"], age_str, item["updated_at"])
        console.print(table)
        if not open_items:
            return
        action = self._menu("Review-Aktion", [
            ("1", "Einzelnen Review schliessen"),
            ("2", "Alle Reviews batch-pruefen (auto-resolve + lernen)"),
            ("3", "Dokument-Vorschau anzeigen"),
            ("0", "Zurueck"),
        ])
        if action == "1":
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
        elif action == "2":
            if not self._init_paperless():
                return
            resolved = 0
            failed = 0
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console, transient=True,
            ) as progress:
                task = progress.add_task("Reviews pruefen...", total=len(open_items))
                for item in open_items:
                    progress.update(task, description=f"Review #{item['id']} (Doc #{item['doc_id']})")
                    try:
                        if learn_from_review(
                            item["id"], self.run_db, self.paperless,
                            self.learning_profile, self.learning_examples,
                        ):
                            resolved += 1
                    except Exception:
                        failed += 1
                    progress.advance(task)
            console.print(
                f"[green]{resolved} geschlossen + gelernt[/green], "
                f"{len(open_items) - resolved - failed} noch offen, "
                f"{failed} Fehler"
            )
        elif action == "3":
            preview_id_str = Prompt.ask("Review-ID fuer Vorschau")
            try:
                preview_id = int(preview_id_str)
            except ValueError:
                console.print("[red]Ungueltige ID.[/red]")
                return
            review = self.run_db.get_review_with_suggestion(preview_id)
            if not review:
                console.print("[yellow]Kein Review mit dieser ID gefunden.[/yellow]")
                return
            doc_id = review["doc_id"]
            if not self._init_paperless():
                return
            try:
                doc = self.paperless.get_document(doc_id)
            except Exception as exc:
                console.print(f"[red]Dokument #{doc_id} nicht ladbar: {exc}[/red]")
                return
            # Show document preview
            console.print(Panel(
                f"[bold]Titel:[/bold] {doc.get('title', '?')}\n"
                f"[bold]Datei:[/bold] {doc.get('original_file_name', '?')}\n"
                f"[bold]Erstellt:[/bold] {doc.get('created', '?')}\n"
                f"[bold]Korrespondent-ID:[/bold] {doc.get('correspondent', 'keiner')}\n"
                f"[bold]Tags:[/bold] {doc.get('tags', [])}\n"
                f"[bold]Typ:[/bold] {doc.get('document_type', 'keiner')}\n"
                f"[bold]Pfad:[/bold] {doc.get('storage_path', 'keiner')}",
                title=f"Dokument #{doc_id}",
                border_style="cyan",
            ))
            content = (doc.get("content") or "")[:500]
            if content:
                console.print(Panel(content, title="Inhalt (erste 500 Zeichen)", border_style="dim"))
            # Show original suggestion
            sugg = review.get("suggestion_json", "")
            if isinstance(sugg, str) and sugg:
                try:
                    sugg_dict = json.loads(sugg)
                    console.print(Panel(
                        json.dumps(sugg_dict, indent=2, ensure_ascii=False)[:600],
                        title="Urspruenglicher LLM-Vorschlag",
                        border_style="yellow",
                    ))
                except (json.JSONDecodeError, TypeError):
                    pass


# =========================================================================
# Main
# =========================================================================

def _show_startup_health():
    """Quick health check displayed at startup."""
    checks = []
    # State DB size
    if os.path.exists(STATE_DB_FILE):
        db_size_mb = os.path.getsize(STATE_DB_FILE) / (1024 * 1024)
        checks.append(f"  State-DB: {db_size_mb:.1f} MB")
    # Learning examples count
    if os.path.exists(LEARNING_EXAMPLES_FILE):
        try:
            with open(LEARNING_EXAMPLES_FILE, "r", encoding="utf-8") as f:
                n_examples = sum(1 for _ in f)
            checks.append(f"  Learning-Beispiele: {n_examples}")
        except Exception:
            pass
    # Taxonomy tags count
    if os.path.exists(TAXONOMY_FILE):
        try:
            with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
                tax_data = json.load(f)
            checks.append(f"  Taxonomie-Tags: {len(tax_data.get('tags', {}))}")
        except Exception:
            pass
    # Last successful run
    try:
        db = LocalStateDB(STATE_DB_FILE)
        with db._connect() as conn:
            row = conn.execute(
                "SELECT started_at FROM runs WHERE ended_at IS NOT NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                checks.append(f"  Letzter Run: {row[0]}")
            # Open reviews
            row2 = conn.execute("SELECT COUNT(*) FROM review_queue WHERE status = 'open'").fetchone()
            if row2 and row2[0] > 0:
                checks.append(f"  Offene Reviews: [yellow]{row2[0]}[/yellow]")
    except Exception:
        pass
    for c in checks:
        log.info(c)


def main():
    """Start: Vollautomatik direkt, oder --menu fuer das alte Hauptmenue."""
    log.info("=" * 40)
    log.info(f"[bold]Paperless-NGX Organizer v{__version__}[/bold]")
    log.info(f"  Python: {sys.version.split()[0]}")
    log.info(f"  LLM: {LLM_MODEL or '(auto)'} @ {LLM_URL}")
    log.info(f"  Log-Datei: {LOG_FILE}")
    log.info(f"  State-DB: {STATE_DB_FILE}")
    _show_startup_health()
    log.info("=" * 40)
    app = App()
    try:
        if "--menu" in sys.argv:
            app.menu_main()
        else:
            app._show_header()
            app.action_autopilot()
    except KeyboardInterrupt:
        console.print("\n[bold]Abgebrochen.[/bold]")
    finally:
        try:
            if app.knowledge_db:
                app.knowledge_db.close()
            log.info("Paperless-NGX Organizer beendet")
        except KeyboardInterrupt:
            pass
        except Exception:
            pass


if __name__ == "__main__":
    main()
