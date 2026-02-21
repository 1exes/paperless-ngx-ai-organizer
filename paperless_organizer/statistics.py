"""Statistics and monthly reporting."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .config import (
    ENABLE_WEB_HINTS,
    LEARNING_EXAMPLES_FILE,
    LEARNING_PRIOR_MIN_SAMPLES,
    LLM_MODEL,
    RULE_BASED_MIN_RATIO,
    RULE_BASED_MIN_SAMPLES,
    USE_ARCHIVE_SERIAL_NUMBER,
    console,
    log,
)
from .constants import MONTH_NAMES_DE
from .db import LocalStateDB
from .client import PaperlessClient
from .learning import LearningExamples
from .utils import _is_fully_organized


def show_statistics(paperless: PaperlessClient, run_db: LocalStateDB | None = None):
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
    stale_cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    stale_docs = [d for d in documents
                  if not _is_fully_organized(d)
                  and (d.get("added") or d.get("created", "9999")) < stale_cutoff]
    stale_count = len(stale_docs)

    table2.add_row("Waisen (komplett leer)", f"[red]{orphans}[/red]" if orphans else "0")
    table2.add_row("Stale (>30 Tage unorganisiert)", f"[red]{stale_count}[/red]" if stale_count else "0")
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

    # Correspondent activity
    if correspondents and documents:
        corr_last_activity: dict[int, str] = {}
        for doc in documents:
            cid = doc.get("correspondent")
            if cid:
                doc_date = doc.get("added") or doc.get("created") or ""
                if doc_date > corr_last_activity.get(cid, ""):
                    corr_last_activity[cid] = doc_date
        recent_30 = (datetime.now() - timedelta(days=30)).isoformat()
        recent_90 = (datetime.now() - timedelta(days=90)).isoformat()
        active_30 = sum(1 for cid, dt in corr_last_activity.items() if dt >= recent_30)
        active_90 = sum(1 for cid, dt in corr_last_activity.items() if dt >= recent_90)
        dormant = sum(1 for c in correspondents if c["id"] not in corr_last_activity)
        console.print(
            f"\n[bold]Korrespondenten-Aktivitaet:[/bold] "
            f"[green]{active_30}[/green] aktiv (30 Tage) | "
            f"[yellow]{active_90}[/yellow] aktiv (90 Tage) | "
            f"[dim]{dormant} ohne Dokumente[/dim]"
        )

    # Dokumenttyp-Verteilung
    if doc_types:
        dtype_sorted = sorted(doc_types, key=lambda x: x.get("document_count", 0), reverse=True)
        used_types = [dt for dt in dtype_sorted if dt.get("document_count", 0) > 0]
        if used_types:
            table_dt = Table(title="Dokumenttyp-Verteilung", show_header=True)
            table_dt.add_column("Typ", style="green")
            table_dt.add_column("Dokumente", justify="right")
            table_dt.add_column("Anteil", justify="right")
            total_typed = sum(dt.get("document_count", 0) for dt in used_types)
            for dt in used_types:
                count = dt.get("document_count", 0)
                pct = (count / total_typed * 100) if total_typed else 0
                table_dt.add_row(dt["name"], str(count), f"{pct:.1f}%")
            unused_types = len(dtype_sorted) - len(used_types)
            if unused_types:
                table_dt.add_row(f"[dim]({unused_types} ungenutzte Typen)[/dim]", "", "")
            console.print(table_dt)

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
            if len(empty_paths) <= 10:
                for ep in empty_paths:
                    table_paths.add_row(f"  [red]{ep['name']}[/red]", "[red]0[/red]", "[red]leer[/red]")
        console.print(table_paths)

    # Tag co-occurrence analysis
    if tags and documents:
        tag_name_map = {t["id"]: t["name"] for t in tags}
        cooccur = defaultdict(int)
        for doc in documents:
            doc_tag_ids = doc.get("tags") or []
            doc_tag_names = sorted(tag_name_map.get(tid, "") for tid in doc_tag_ids if tid in tag_name_map)
            for i in range(len(doc_tag_names)):
                for j in range(i + 1, len(doc_tag_names)):
                    if doc_tag_names[i] and doc_tag_names[j]:
                        cooccur[(doc_tag_names[i], doc_tag_names[j])] += 1
        if cooccur:
            top_pairs = sorted(cooccur.items(), key=lambda x: x[1], reverse=True)[:10]
            co_table = Table(title="Tag-Kombinationen (Top 10)", show_header=True)
            co_table.add_column("Tag A", style="green")
            co_table.add_column("Tag B", style="green")
            co_table.add_column("Gemeinsam", justify="right")
            for (a, b), cnt in top_pairs:
                co_table.add_row(a, b, str(cnt))
            console.print(co_table)

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

    # Processing statistics from state DB
    if run_db:
        try:
            stats = run_db.get_processing_stats(days=30)
            table6 = Table(title="Verarbeitungsstatistik (letzte 30 Tage)", show_header=True, width=60)
            table6.add_column("Metrik", style="cyan")
            table6.add_column("Wert", justify="right", style="bold")
            table6.add_row("Verarbeitete Dokumente", str(stats["total_docs"]))
            table6.add_row("Erfolgreich aktualisiert", f"[green]{stats['updated']}[/green]")
            table6.add_row("Fehler", f"[red]{stats['errors']}[/red]" if stats["errors"] else "0")
            table6.add_row("Zur Review", str(stats["reviews"]))
            table6.add_row("Regelbasiert (ohne LLM)", str(stats["rule_based"]))
            table6.add_row("Prior-Fallback", str(stats["prior_fallback"]))
            table6.add_row("Erfolgsrate", f"{stats['success_rate']:.1f}%")
            table6.add_row("Laeufe", str(stats["total_runs"]))
            table6.add_row("Offene Reviews", str(stats["open_reviews"]))
            console.print(table6)
            if stats["top_errors"]:
                err_table = Table(title="Haeufigste Fehler", show_header=True)
                err_table.add_column("Fehler", style="red")
                err_table.add_column("Anzahl", justify="right")
                for err in stats["top_errors"]:
                    err_table.add_row(err["error"], str(err["count"]))
                console.print(err_table)

            # Confidence calibration
            try:
                cal = run_db.get_confidence_calibration()
                if cal:
                    cal_table = Table(title="Konfidenz-Kalibrierung", show_header=True, width=60)
                    cal_table.add_column("Konfidenz", style="cyan")
                    cal_table.add_column("Gesamt", justify="right")
                    cal_table.add_column("Korrigiert", justify="right")
                    cal_table.add_column("Genauigkeit", justify="right", style="bold")
                    for level in ["high", "medium", "low"]:
                        if level in cal:
                            d = cal[level]
                            acc_color = "green" if d["accuracy"] >= 80 else ("yellow" if d["accuracy"] >= 60 else "red")
                            cal_table.add_row(
                                level, str(d["total"]), str(d["corrected"]),
                                f"[{acc_color}]{d['accuracy']}%[/{acc_color}]",
                            )
                    console.print(cal_table)
            except Exception:
                pass
        except Exception:
            pass


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
