"""
Paperless-NGX Cleanup Tool
- Tag-Aufräumung (Duplikate, Ungenutzte)
- Korrespondenten-Zusammenführung
- Duplikat-Erkennung (nur melden, NIE Dokumente löschen!)
- Protokoll aller Änderungen
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

load_dotenv()
console = Console()

LOG_FILE = os.path.join(os.path.dirname(__file__), "cleanup_log.txt")


def log(message: str):
    """Schreibt ins Log und auf die Konsole."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    console.print(f"[dim]{line}[/dim]")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class PaperlessClient:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        })

    def _get_all(self, endpoint: str) -> list:
        results = []
        url = f"{self.url}/api/{endpoint}/?page_size=100"
        while url:
            url = url.replace("http://", "https://")
            resp = self.session.get(url)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))
            url = data.get("next")
        return results

    def get_tags(self): return self._get_all("tags")
    def get_correspondents(self): return self._get_all("correspondents")
    def get_document_types(self): return self._get_all("document_types")

    def get_all_documents(self) -> list:
        results = []
        url = f"{self.url}/api/documents/?page_size=100"
        while url:
            url = url.replace("http://", "https://")
            resp = self.session.get(url)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))
            url = data.get("next")
        return results

    def delete_tag(self, tag_id: int):
        self.session.delete(f"{self.url}/api/tags/{tag_id}/")

    def delete_correspondent(self, corr_id: int):
        self.session.delete(f"{self.url}/api/correspondents/{corr_id}/")

    def delete_document_type(self, type_id: int):
        self.session.delete(f"{self.url}/api/document_types/{type_id}/")

    def update_document(self, doc_id: int, data: dict):
        resp = self.session.patch(f"{self.url}/api/documents/{doc_id}/", json=data)
        resp.raise_for_status()
        return resp.json()


# ── Tag-Aufräumung ────────────────────────────────────────────────────────────

def cleanup_tags(paperless: PaperlessClient, dry_run: bool = True):
    """Löscht ungenutzte und überflüssige Tags."""
    console.print(Panel("[bold]Tag-Aufräumung[/bold]", border_style="yellow"))

    tags = paperless.get_tags()
    console.print(f"Gesamt: {len(tags)} Tags")

    # Ungenutzte Tags
    unused = [t for t in tags if t["document_count"] == 0]
    console.print(f"[yellow]Ungenutzt (0 Dokumente): {len(unused)}[/yellow]")

    if unused and not dry_run:
        for t in unused:
            paperless.delete_tag(t["id"])
            log(f"TAG GELÖSCHT: '{t['name']}' (ID {t['id']}, 0 Dokumente)")
        console.print(f"[green]{len(unused)} ungenutzte Tags gelöscht[/green]")

    # Single-Use Tags (1 Dokument) - wahrscheinlich KI-Müll
    single = [t for t in tags if t["document_count"] == 1]
    console.print(f"[yellow]Single-Use (1 Dokument): {len(single)}[/yellow]")

    if single and not dry_run:
        deleted = 0
        for t in single:
            paperless.delete_tag(t["id"])
            log(f"TAG GELÖSCHT: '{t['name']}' (ID {t['id']}, 1 Dokument)")
            deleted += 1
        console.print(f"[green]{deleted} Single-Use Tags gelöscht[/green]")

    # Verbleibend
    remaining = paperless.get_tags()
    console.print(f"\n[bold]Verbleibend: {len(remaining)} Tags[/bold]")
    if dry_run:
        console.print("[yellow]TESTMODUS: Nichts gelöscht[/yellow]")


# ── Korrespondenten-Aufräumung ────────────────────────────────────────────────

def find_correspondent_groups(correspondents: list) -> dict:
    """Gruppiert Korrespondenten nach Ähnlichkeit."""
    groups = defaultdict(list)
    for c in correspondents:
        name = c["name"].lower().strip()
        # Bekannte Gruppen
        if "drk" in name or "deutsches rotes kreuz" in name or "wasserwacht" in name or "blutspende" in name:
            groups["DRK"].append(c)
        elif "aok" in name:
            groups["AOK"].append(c)
        elif name.startswith("msg") or "msg systems" in name:
            groups["msg systems ag"].append(c)
        elif "amazon" in name:
            groups["Amazon"].append(c)
        elif "baader" in name:
            groups["Baader Bank"].append(c)
        elif "apple" in name:
            groups["Apple"].append(c)
        elif "digistore" in name:
            groups["Digistore24"].append(c)
        elif "check24" in name:
            groups["CHECK24"].append(c)
        elif "axa" in name:
            groups["AXA"].append(c)
        elif "scalable" in name:
            groups["Scalable Capital"].append(c)
        elif "dhl" in name:
            groups["DHL"].append(c)
        elif "corporate benefits" in name:
            groups["Corporate Benefits"].append(c)
        elif "1&1" in name or "1und1" in name:
            groups["1&1"].append(c)
    return groups


def cleanup_correspondents(paperless: PaperlessClient, dry_run: bool = True):
    """Führt doppelte Korrespondenten zusammen."""
    console.print(Panel("[bold]Korrespondenten-Aufräumung[/bold]", border_style="cyan"))

    correspondents = paperless.get_correspondents()
    console.print(f"Gesamt: {len(correspondents)} Korrespondenten")

    # Ungenutzte löschen
    unused = [c for c in correspondents if c["document_count"] == 0]
    console.print(f"[yellow]Ungenutzt: {len(unused)}[/yellow]")

    if unused and not dry_run:
        for c in unused:
            paperless.delete_correspondent(c["id"])
            log(f"KORRESPONDENT GELÖSCHT: '{c['name']}' (ID {c['id']}, 0 Dokumente)")
        console.print(f"[green]{len(unused)} ungenutzte Korrespondenten gelöscht[/green]")

    # Duplikate finden
    groups = find_correspondent_groups(correspondents)
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}

    if duplicates:
        console.print(f"\n[bold]{len(duplicates)} Duplikat-Gruppen gefunden:[/bold]")
        for group_name, items in sorted(duplicates.items()):
            total = sum(c["document_count"] for c in items)
            # Behalte den mit den meisten Dokumenten
            items.sort(key=lambda x: x["document_count"], reverse=True)
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
                # Dokumente vom alten zum neuen Korrespondenten umziehen
                all_docs = paperless.get_all_documents()
                for m in merge:
                    docs_to_move = [d for d in all_docs if d.get("correspondent") == m["id"]]
                    for doc in docs_to_move:
                        paperless.update_document(doc["id"], {"correspondent": keep["id"]})
                        log(f"DOKUMENT #{doc['id']}: Korrespondent '{m['name']}' -> '{keep['name']}'")
                    # Jetzt löschen
                    paperless.delete_correspondent(m["id"])
                    log(f"KORRESPONDENT GELÖSCHT: '{m['name']}' (ID {m['id']}, gemergt in '{keep['name']}')")

    remaining = paperless.get_correspondents()
    console.print(f"\n[bold]Verbleibend: {len(remaining)} Korrespondenten[/bold]")
    if dry_run:
        console.print("[yellow]TESTMODUS: Nichts geändert[/yellow]")


# ── Dokumenttypen-Aufräumung ──────────────────────────────────────────────────

ALLOWED_DOC_TYPES = {
    "vertrag", "rechnung", "bescheinigung", "information", "kontoauszug",
    "zeugnis", "angebot", "kündigung", "mahnung", "versicherungspolice",
    "steuerbescheid", "arztbericht", "gehaltsabrechnung", "bestellung",
    "korrespondenz", "dokumentation", "lizenz", "formular", "urkunde",
    "bewerbung",
}


def cleanup_document_types(paperless: PaperlessClient, dry_run: bool = True):
    """Löscht ungenutzte und überflüssige Dokumenttypen."""
    console.print(Panel("[bold]Dokumenttypen-Aufräumung[/bold]", border_style="magenta"))

    types = paperless.get_document_types()
    console.print(f"Gesamt: {len(types)} Dokumenttypen")

    unused = [t for t in types if t["document_count"] == 0]
    console.print(f"[yellow]Ungenutzt: {len(unused)}[/yellow]")

    if unused and not dry_run:
        for t in unused:
            paperless.delete_document_type(t["id"])
            log(f"DOKUMENTTYP GELÖSCHT: '{t['name']}' (ID {t['id']}, 0 Dokumente)")
        console.print(f"[green]{len(unused)} ungenutzte Dokumenttypen gelöscht[/green]")

    remaining = paperless.get_document_types()
    console.print(f"[bold]Verbleibend: {len(remaining)} Dokumenttypen[/bold]")
    if dry_run:
        console.print("[yellow]TESTMODUS: Nichts gelöscht[/yellow]")


# ── Duplikat-Erkennung ───────────────────────────────────────────────────────

def find_duplicate_documents(paperless: PaperlessClient):
    """Findet potenziell doppelte Dokumente. Löscht NICHTS!"""
    console.print(Panel(
        "[bold]Duplikat-Erkennung[/bold]\n[red]Nur Meldung - es wird NICHTS gelöscht![/red]",
        border_style="red"
    ))

    console.print("Lade alle Dokumente...")
    docs = paperless.get_all_documents()
    console.print(f"{len(docs)} Dokumente geladen")

    # Nach gleichem Dateinamen gruppieren
    by_filename = defaultdict(list)
    for d in docs:
        fname = d.get("original_file_name", "").strip()
        if fname:
            by_filename[fname].append(d)

    filename_dupes = {k: v for k, v in by_filename.items() if len(v) > 1}

    # Nach ähnlichem Titel gruppieren
    by_title = defaultdict(list)
    for d in docs:
        title = d.get("title", "").strip().lower()
        if title and title != "unbekannt":
            by_title[title].append(d)

    title_dupes = {k: v for k, v in by_title.items() if len(v) > 1}

    # Bericht
    report_file = os.path.join(os.path.dirname(__file__), "duplikate_bericht.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Duplikat-Bericht - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"GLEICHER DATEINAME: {len(filename_dupes)} Gruppen\n")
        f.write(f"{'-'*40}\n")
        for fname, items in sorted(filename_dupes.items()):
            f.write(f"\nDatei: {fname}\n")
            for d in items:
                f.write(f"  ID={d['id']} | Titel: {d['title']} | Erstellt: {d.get('created','?')}\n")

        f.write(f"\n\nGLEICHER TITEL: {len(title_dupes)} Gruppen\n")
        f.write(f"{'-'*40}\n")
        for title, items in sorted(title_dupes.items()):
            if len(items) <= 5:  # Nur kleine Gruppen zeigen
                f.write(f"\nTitel: {title}\n")
                for d in items:
                    f.write(f"  ID={d['id']} | Datei: {d.get('original_file_name','?')} | Erstellt: {d.get('created','?')}\n")

    console.print(f"\n[bold]Gleicher Dateiname:[/bold] {len(filename_dupes)} Duplikat-Gruppen")
    console.print(f"[bold]Gleicher Titel:[/bold] {len(title_dupes)} Duplikat-Gruppen")
    console.print(f"\n[green]Bericht gespeichert: {report_file}[/green]")
    console.print("[red]Es wurden KEINE Dokumente gelöscht![/red]")

    log(f"DUPLIKAT-SCAN: {len(filename_dupes)} Dateiname-Duplikate, {len(title_dupes)} Titel-Duplikate gefunden")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paperless-NGX Cleanup Tool")
    parser.add_argument("--apply", action="store_true", help="Änderungen wirklich anwenden")
    parser.add_argument("--tags", action="store_true", help="Tags aufräumen")
    parser.add_argument("--correspondents", action="store_true", help="Korrespondenten aufräumen")
    parser.add_argument("--doc-types", action="store_true", help="Dokumenttypen aufräumen")
    parser.add_argument("--duplicates", action="store_true", help="Duplikate finden (löscht NIE)")
    parser.add_argument("--all", action="store_true", help="Alles ausführen")
    args = parser.parse_args()

    url = os.getenv("PAPERLESS_URL")
    token = os.getenv("PAPERLESS_TOKEN")
    if not all([url, token]):
        console.print("[red]Fehler: .env nötig![/red]")
        sys.exit(1)

    dry_run = not args.apply
    paperless = PaperlessClient(url, token)

    log(f"=== CLEANUP START (Modus: {'LIVE' if not dry_run else 'TEST'}) ===")

    console.print(Panel(
        f"[bold]Paperless-NGX Cleanup[/bold]\n"
        f"Modus: {'[red]LIVE[/red]' if not dry_run else '[green]TEST[/green]'}",
        border_style="blue"
    ))

    if args.tags or args.all:
        cleanup_tags(paperless, dry_run)
    if args.correspondents or args.all:
        cleanup_correspondents(paperless, dry_run)
    if args.doc_types or args.all:
        cleanup_document_types(paperless, dry_run)
    if args.duplicates or args.all:
        find_duplicate_documents(paperless)

    if not any([args.tags, args.correspondents, args.doc_types, args.duplicates, args.all]):
        console.print("[yellow]Bitte eine Option wählen: --tags, --correspondents, --doc-types, --duplicates oder --all[/yellow]")

    log(f"=== CLEANUP ENDE ===")


if __name__ == "__main__":
    main()
