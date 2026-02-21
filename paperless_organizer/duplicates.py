"""Duplicate detection: filename, title, content checksum, and fingerprint similarity."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from rich.panel import Panel
from rich.table import Table

from .config import console, log
from .client import PaperlessClient
from .utils import (
    _content_fingerprint,
    _content_similarity,
    _lsh_find_candidates,
    _minhash_signature,
)


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

    # Nach normalisiertem Dateinamen gruppieren
    by_norm_filename = defaultdict(list)
    for d in docs:
        fname = d.get("original_file_name", "").strip()
        if fname:
            norm = fname.lower()
            norm = re.sub(r"\.[a-z]{2,4}$", "", norm)
            norm = re.sub(r"\d{4}[-_]\d{2}[-_]\d{2}", "", norm)
            norm = re.sub(r"[0-9a-f]{8}[-_]?[0-9a-f]{4}[-_]?[0-9a-f]{4}[-_]?[0-9a-f]{4}[-_]?[0-9a-f]{12}", "", norm)
            norm = re.sub(r"scan[-_]?\d+", "", norm, flags=re.IGNORECASE)
            norm = re.sub(r"[-_\s]+", " ", norm).strip()
            if norm and len(norm) > 3:
                by_norm_filename[norm].append(d)
    norm_filename_dupes = {k: v for k, v in by_norm_filename.items() if len(v) > 1}
    exact_ids = set()
    for items in filename_dupes.values():
        exact_ids.update(d["id"] for d in items)
    norm_filename_dupes = {
        k: v for k, v in norm_filename_dupes.items()
        if not all(d["id"] in exact_ids for d in v)
    }

    # Nach Inhalts-Checksum gruppieren
    by_checksum = defaultdict(list)
    for d in docs:
        content = (d.get("content") or "").strip()
        if content and len(content) > 50:
            chk = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
            by_checksum[chk].append(d)
    checksum_dupes = {k: v for k, v in by_checksum.items() if len(v) > 1}

    # Nach Titel gruppieren
    by_title = defaultdict(list)
    for d in docs:
        title = d.get("title", "").strip().lower()
        if title and title != "unbekannt":
            by_title[title].append(d)
    title_dupes = {k: v for k, v in by_title.items() if len(v) > 1}

    # Inhalts-Checksum-Duplikate anzeigen
    if checksum_dupes:
        console.print(f"\n[bold red]Exakt gleicher Inhalt (Checksum): {len(checksum_dupes)} Gruppen[/bold red]")
        for chk, items in sorted(checksum_dupes.items(), key=lambda x: -len(x[1])):
            table = Table(title=f"SHA256-Prefix: {chk}", show_header=True)
            table.add_column("ID", width=6)
            table.add_column("Dateiname")
            table.add_column("Titel")
            table.add_column("Erstellt", width=12)
            for d in sorted(items, key=lambda x: x.get("id", 0)):
                table.add_row(
                    str(d["id"]),
                    d.get("original_file_name", "?"),
                    d.get("title", "?"),
                    str(d.get("created", "?"))[:10],
                )
            console.print(table)
    else:
        console.print("\n[green]Keine exakten Inhalts-Duplikate gefunden.[/green]")

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

    if len(fingerprints) <= 2000:
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                sim = _content_similarity(fingerprints[i][1], fingerprints[j][1])
                if sim >= 0.75:
                    doc_a, doc_b = fingerprints[i][0], fingerprints[j][0]
                    content_dupes.append((sim, doc_a, doc_b))
    elif len(fingerprints) <= 20000:
        console.print(f"  [cyan]MinHash/LSH-Modus fuer {len(fingerprints)} Dokumente...[/cyan]")
        sigs = [_minhash_signature(fp) for _, fp in fingerprints]
        candidates = _lsh_find_candidates(sigs)
        console.print(f"  {len(candidates)} Kandidatenpaare gefunden, verifiziere...")
        for i, j in candidates:
            sim = _content_similarity(fingerprints[i][1], fingerprints[j][1])
            if sim >= 0.75:
                doc_a, doc_b = fingerprints[i][0], fingerprints[j][0]
                content_dupes.append((sim, doc_a, doc_b))
    else:
        console.print(f"  [yellow]Zu viele Dokumente ({len(fingerprints)}) - uebersprungen[/yellow]")

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

    # Zusammenfassung
    console.print(f"\n[bold]Zusammenfassung:[/bold]")
    console.print(f"  Exakte Inhalts-Duplikate: {len(checksum_dupes)} Gruppen ({sum(len(v) for v in checksum_dupes.values())} Dokumente)")
    console.print(f"  Dateiname-Duplikate: {len(filename_dupes)} Gruppen ({sum(len(v) for v in filename_dupes.values())} Dokumente)")
    console.print(f"  Titel-Duplikate: {len(title_dupes)} Gruppen ({sum(len(v) for v in title_dupes.values())} Dokumente)")
    console.print(f"  Inhalts-Aehnlichkeit: {len(content_dupes)} Paare")
    console.print("[red]Es wurden KEINE Dokumente geloescht![/red]")
    log.info(
        f"  DUPLIKAT-SCAN fertig - {len(checksum_dupes)} Checksum-Gruppen, "
        f"{len(filename_dupes)} Dateiname-Gruppen, {len(title_dupes)} Titel-Gruppen, "
        f"{len(content_dupes)} Inhalts-Paare"
    )
