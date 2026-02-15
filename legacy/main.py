"""
Paperless-NGX Organizer
Analysiert Dokumente per Claude Code CLI und organisiert sie komplett:
Titel, Tags, Korrespondent, Dokumenttyp, Speicherpfad, Archivnummer.
Nutzt dein bestehendes Claude Code Abo - kein extra API-Key nötig.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

load_dotenv()
console = Console()


# ── Paperless API Client ─────────────────────────────────────────────────────

# Erlaubte Dokumenttypen - NUR diese werden verwendet/erstellt
ALLOWED_DOC_TYPES = [
    "Vertrag", "Rechnung", "Bescheinigung", "Information", "Kontoauszug",
    "Zeugnis", "Angebot", "Kündigung", "Mahnung", "Versicherungspolice",
    "Steuerbescheid", "Arztbericht", "Gehaltsabrechnung", "Bestellung",
    "Korrespondenz", "Dokumentation", "Lizenz", "Formular", "Urkunde",
    "Bewerbung",
]

# Farbpalette für Tags - gut unterscheidbar, weiße Schrift
TAG_COLORS = [
    "#e74c3c",  # Rot
    "#2ecc71",  # Grün
    "#3498db",  # Blau
    "#f39c12",  # Orange
    "#9b59b6",  # Lila
    "#1abc9c",  # Türkis
    "#e67e22",  # Dunkelorange
    "#2980b9",  # Dunkelblau
    "#27ae60",  # Dunkelgrün
    "#c0392b",  # Dunkelrot
    "#8e44ad",  # Dunkellila
    "#16a085",  # Dunkeltürkis
    "#d35400",  # Rost
    "#2c3e50",  # Nachtblau
    "#f1c40f",  # Gelb
    "#e91e63",  # Pink
    "#00bcd4",  # Cyan
    "#ff5722",  # Tieforange
    "#607d8b",  # Blaugrau
    "#795548",  # Braun
]


class PaperlessClient:
    """Client für die Paperless-NGX REST API."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token
        self._color_index = 0
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

    def get_tags(self) -> list:
        return self._get_all("tags")

    def get_correspondents(self) -> list:
        return self._get_all("correspondents")

    def get_document_types(self) -> list:
        return self._get_all("document_types")

    def get_storage_paths(self) -> list:
        return self._get_all("storage_paths")

    def get_documents(self, page_size: int = 25, page: int = 1) -> dict:
        resp = self.session.get(
            f"{self.url}/api/documents/?page_size={page_size}&page={page}"
        )
        resp.raise_for_status()
        return resp.json()

    def get_document(self, doc_id: int) -> dict:
        resp = self.session.get(f"{self.url}/api/documents/{doc_id}/")
        resp.raise_for_status()
        return resp.json()

    def update_document(self, doc_id: int, data: dict) -> dict:
        resp = self.session.patch(
            f"{self.url}/api/documents/{doc_id}/",
            json=data,
        )
        resp.raise_for_status()
        return resp.json()

    def _with_permissions(self, data: dict) -> dict:
        """Fügt Owner und Berechtigungen hinzu damit nichts als 'Privat' angezeigt wird."""
        data["owner"] = 4  # Edgar
        data["set_permissions"] = {
            "view": {"users": [4], "groups": []},
            "change": {"users": [4], "groups": []},
        }
        return data

    def _next_color(self) -> str:
        """Gibt die nächste Farbe aus der Palette zurück (rotiert)."""
        color = TAG_COLORS[self._color_index % len(TAG_COLORS)]
        self._color_index += 1
        return color

    def create_tag(self, name: str) -> dict:
        color = self._next_color()
        # Gelb (#f1c40f) braucht dunkle Schrift, Rest weiß
        text_color = "#000000" if color == "#f1c40f" else "#ffffff"
        data = {"name": name, "color": color, "text_color": text_color}
        resp = self.session.post(
            f"{self.url}/api/tags/",
            json=self._with_permissions(data),
        )
        resp.raise_for_status()
        return resp.json()

    def create_correspondent(self, name: str) -> dict:
        resp = self.session.post(
            f"{self.url}/api/correspondents/",
            json=self._with_permissions({"name": name}),
        )
        resp.raise_for_status()
        return resp.json()

    def create_document_type(self, name: str) -> dict:
        resp = self.session.post(
            f"{self.url}/api/document_types/",
            json=self._with_permissions({"name": name}),
        )
        resp.raise_for_status()
        return resp.json()

    def create_storage_path(self, name: str, path: str) -> dict:
        resp = self.session.post(
            f"{self.url}/api/storage_paths/",
            json=self._with_permissions({"name": name, "path": path}),
        )
        resp.raise_for_status()
        return resp.json()

    def get_next_asn(self) -> int:
        """Nächste freie Archivnummer ermitteln."""
        resp = self.session.get(
            f"{self.url}/api/documents/?page_size=1&ordering=-archive_serial_number"
        )
        resp.raise_for_status()
        data = resp.json()
        if data["results"] and data["results"][0].get("archive_serial_number"):
            return data["results"][0]["archive_serial_number"] + 1
        return 1


# ── Claude CLI Analyzer ──────────────────────────────────────────────────────

def _find_claude_cli() -> str:
    path = shutil.which("claude") or shutil.which("claude.cmd")
    if path:
        return path
    npm_path = os.path.expandvars(r"%APPDATA%\npm\claude.cmd")
    if os.path.exists(npm_path):
        return npm_path
    raise FileNotFoundError("Claude CLI nicht gefunden!")


class DocumentAnalyzer:
    """Analysiert Dokumente mit der Claude Code CLI."""

    def __init__(self):
        self.claude_path = _find_claude_cli()

    def analyze(self, document: dict, existing_tags: list,
                existing_correspondents: list, existing_types: list,
                existing_paths: list) -> dict:

        # Nur Top-Korrespondenten nach Dokumentanzahl (Token sparen)
        top_corrs = sorted(existing_correspondents, key=lambda c: c["document_count"], reverse=True)
        corr_names = [c["name"] for c in top_corrs[:50]]
        path_names = [f'{p["name"]} ({p["path"]})' for p in existing_paths]

        current_tags = [
            t["name"] for t in existing_tags
            if t["id"] in (document.get("tags") or [])
        ]
        current_corr = next(
            (c["name"] for c in existing_correspondents
             if c["id"] == document.get("correspondent")), "")
        current_type = next(
            (t["name"] for t in existing_types
             if t["id"] == document.get("document_type")), "")
        current_path = next(
            (p["name"] for p in existing_paths
             if p["id"] == document.get("storage_path")), "")

        # Token-effizient: Kurze Dokumente komplett, lange nur Anfang
        content = document.get("content") or ""
        content_len = len(content)
        if content_len > 5000:
            # Langes Dokument (E-Book, Handbuch etc.) -> nur erste ~1500 Zeichen
            content_preview = content[:1500] + f"\n[...{content_len} Zeichen insgesamt, Rest abgeschnitten...]"
        elif content_len > 2000:
            content_preview = content[:2000]
        else:
            content_preview = content

        prompt = f"""Paperless-NGX Dokument organisieren. Besitzer: Edgar Richter, Reichenbach/Sachsen.
Job: Systemadministrator bei WBS TRAINING AG (seit Aug 2025). Vorher: Azubi msg systems ag (2022-2025, Fachinformatiker). DRK-Mitglied. AOK PLUS. Auto: SPV 1109.

DOKUMENT #{document['id']}:
Titel: {document.get('title', '?')} | Datei: {document.get('original_file_name', '?')} | Erstellt: {document.get('created', '?')}
Tags: {current_tags or 'keine'} | Korr: {current_corr or 'keiner'} | Typ: {current_type or 'keiner'} | Pfad: {current_path or 'keiner'}

INHALT:
{content_preview}

KORRESPONDENTEN (bevorzuge vorhandene): {', '.join(corr_names)}

DOKUMENTTYPEN (NUR diese): Vertrag, Rechnung, Bescheinigung, Information, Kontoauszug, Zeugnis, Angebot, Kündigung, Mahnung, Versicherungspolice, Steuerbescheid, Arztbericht, Gehaltsabrechnung, Bestellung, Korrespondenz, Dokumentation, Lizenz, Formular, Urkunde, Bewerbung

SPEICHERPFADE (NUR diese Kategorienamen verwenden, NICHT den vollen Pfad mit Jahr/Titel!):
{', '.join(p['name'] for p in existing_paths if 'Duplikat' not in p['name'])}
WICHTIG: storage_path muss EXAKT einem der obigen Namen entsprechen, z.B. "Auto/Unfall" oder "Finanzen/Bank"

REGELN: msg-Docs->Arbeit/msg, WBS->Arbeit/WBS, DRK->Persönlich/DRK oder Ausbildung/DRK. Max 3-5 Tags (deutsch, kurz). Titel deutsch, aussagekräftig. Korrespondent=Absender/Firma.

NUR JSON, kein anderer Text:
{{"title": "Titel", "tags": ["Tag1", "Tag2"], "correspondent": "Firma", "document_type": "Typ", "storage_path_name": "Kategorie", "storage_path": "Kategorie/Sub", "reasoning": "Kurz"}}"""

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)

        result = subprocess.run(
            [self.claude_path, "-p", "-", "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI Fehler: {result.stderr}")

        text = result.stdout.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        return json.loads(text)


# ── ID-Auflösung ─────────────────────────────────────────────────────────────

def resolve_ids(paperless: PaperlessClient, suggestion: dict,
                tags: list, correspondents: list,
                doc_types: list, storage_paths: list) -> dict:
    """Wandelt Namen in IDs um, erstellt fehlende Einträge."""

    # Tags (mit Fehlerbehandlung für bereits existierende)
    tag_map = {t["name"].lower(): t["id"] for t in tags}
    tag_ids = []
    for tag_name in suggestion.get("tags", []):
        key = tag_name.lower()
        if key in tag_map:
            tag_ids.append(tag_map[key])
        else:
            try:
                console.print(f"  [yellow]+ Neuer Tag:[/yellow] {tag_name}")
                new = paperless.create_tag(tag_name)
                tag_ids.append(new["id"])
                tag_map[key] = new["id"]
                tags.append(new)  # Cache aktualisieren
            except requests.exceptions.HTTPError:
                # Tag existiert vielleicht schon -> neu laden
                fresh_tags = paperless.get_tags()
                tag_map = {t["name"].lower(): t["id"] for t in fresh_tags}
                if key in tag_map:
                    tag_ids.append(tag_map[key])
                    tags.clear()
                    tags.extend(fresh_tags)

    # Korrespondent
    corr_map = {c["name"].lower(): c["id"] for c in correspondents}
    corr_id = None
    corr_name = suggestion.get("correspondent", "")
    if corr_name:
        key = corr_name.lower()
        if key in corr_map:
            corr_id = corr_map[key]
        else:
            try:
                console.print(f"  [yellow]+ Neuer Korrespondent:[/yellow] {corr_name}")
                new = paperless.create_correspondent(corr_name)
                corr_id = new["id"]
                correspondents.append(new)
            except requests.exceptions.HTTPError:
                fresh = paperless.get_correspondents()
                corr_map = {c["name"].lower(): c["id"] for c in fresh}
                corr_id = corr_map.get(key)
                correspondents.clear()
                correspondents.extend(fresh)

    # Dokumenttyp (nur aus erlaubter Liste!)
    type_map = {t["name"].lower(): t["id"] for t in doc_types}
    type_id = None
    type_name = suggestion.get("document_type", "")
    if type_name:
        allowed_lower = {t.lower(): t for t in ALLOWED_DOC_TYPES}
        key = type_name.lower()
        if key not in allowed_lower:
            console.print(f"  [red]Dokumenttyp '{type_name}' nicht erlaubt, übersprungen.[/red]")
        elif key in type_map:
            type_id = type_map[key]
        else:
            canonical_name = allowed_lower[key]
            try:
                console.print(f"  [yellow]+ Neuer Dokumenttyp:[/yellow] {canonical_name}")
                new = paperless.create_document_type(canonical_name)
                type_id = new["id"]
                doc_types.append(new)
            except requests.exceptions.HTTPError:
                fresh = paperless.get_document_types()
                type_map = {t["name"].lower(): t["id"] for t in fresh}
                type_id = type_map.get(key)
                doc_types.clear()
                doc_types.extend(fresh)

    # Speicherpfad - Suche nach Name UND Pfad (flexibel)
    path_name_map = {p["name"].lower(): p["id"] for p in storage_paths}
    path_path_map = {p["path"].lower(): p["id"] for p in storage_paths}
    path_id = None
    path_value = suggestion.get("storage_path", "")
    path_name = suggestion.get("storage_path_name", "")
    if path_value:
        # Zuerst nach exaktem Namen suchen (z.B. "Auto/Unfall")
        key = path_value.lower()
        if key in path_name_map:
            path_id = path_name_map[key]
        elif key in path_path_map:
            path_id = path_path_map[key]
        else:
            # Auch nach Teilmatch suchen (z.B. "Auto/Unfall" in "Auto/Unfall/{{ ... }}")
            for p in storage_paths:
                if p["name"].lower().startswith(key) or key.startswith(p["name"].lower()):
                    path_id = p["id"]
                    break
                if p["path"].lower().startswith(key) or key in p["path"].lower():
                    path_id = p["id"]
                    break
        if path_id is None:
            # Neuen Pfad erstellen mit richtigem Template
            template = f"{path_value}/{{{{ created_year }}}}/{{{{ title }}}}"
            try:
                console.print(f"  [yellow]+ Neuer Speicherpfad:[/yellow] {path_value}")
                new = paperless.create_storage_path(path_value, template)
                path_id = new["id"]
                storage_paths.append(new)
            except requests.exceptions.HTTPError as e:
                console.print(f"  [red]Speicherpfad-Fehler: {e}[/red]")

    # Archivnummer
    asn = paperless.get_next_asn()

    update_data = {"title": suggestion.get("title", "")}
    if tag_ids:
        update_data["tags"] = tag_ids
    if corr_id is not None:
        update_data["correspondent"] = corr_id
    if type_id is not None:
        update_data["document_type"] = type_id
    if path_id is not None:
        update_data["storage_path"] = path_id
    update_data["archive_serial_number"] = asn

    return update_data


# ── Anzeige ───────────────────────────────────────────────────────────────────

def show_suggestion(document: dict, suggestion: dict, asn: int,
                    tags: list, correspondents: list,
                    doc_types: list, storage_paths: list):

    current_tags = [t["name"] for t in tags if t["id"] in (document.get("tags") or [])]
    current_corr = next((c["name"] for c in correspondents if c["id"] == document.get("correspondent")), "Keiner")
    current_type = next((t["name"] for t in doc_types if t["id"] == document.get("document_type")), "Keiner")
    current_path = next((p["name"] for p in storage_paths if p["id"] == document.get("storage_path")), "Keiner")

    table = Table(title=f"Dokument #{document['id']}", show_header=True, width=80)
    table.add_column("Feld", style="cyan", width=18)
    table.add_column("Aktuell", style="red", width=28)
    table.add_column("Vorschlag", style="green", width=30)

    table.add_row("Titel", document.get("title", ""), suggestion.get("title", ""))
    table.add_row("Tags", ", ".join(current_tags) or "Keine", ", ".join(suggestion.get("tags", [])))
    table.add_row("Korrespondent", current_corr, suggestion.get("correspondent", ""))
    table.add_row("Dokumenttyp", current_type, suggestion.get("document_type", ""))
    table.add_row("Speicherpfad", current_path, suggestion.get("storage_path", ""))
    table.add_row("Archivnummer", str(document.get("archive_serial_number") or "Keine"), str(asn))

    console.print(table)
    if suggestion.get("reasoning"):
        console.print(Panel(suggestion["reasoning"], title="Begründung", border_style="blue"))


# ── Dokument verarbeiten ─────────────────────────────────────────────────────

def process_document(doc_id: int, paperless: PaperlessClient,
                     analyzer: DocumentAnalyzer, tags: list,
                     correspondents: list, doc_types: list,
                     storage_paths: list, dry_run: bool = True,
                     batch_mode: bool = False) -> bool:

    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print(f"[bold]Lade Dokument #{doc_id}...[/bold]")
    document = paperless.get_document(doc_id)

    console.print(f"[dim]Titel: {document.get('title', 'Unbekannt')}[/dim]")
    console.print(f"[dim]Inhalt: {(document.get('content') or '')[:100]}...[/dim]")

    console.print("[bold]Analysiere mit Claude AI...[/bold]")
    try:
        suggestion = analyzer.analyze(document, tags, correspondents, doc_types, storage_paths)
    except json.JSONDecodeError as e:
        console.print(f"[red]JSON-Parse-Fehler: {e}[/red]")
        return False
    except subprocess.TimeoutExpired:
        console.print("[red]Timeout: Claude hat zu lange gebraucht.[/red]")
        return False
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        return False

    asn = paperless.get_next_asn()
    show_suggestion(document, suggestion, asn, tags, correspondents, doc_types, storage_paths)

    if dry_run:
        console.print("[yellow]TESTMODUS: Keine Änderungen werden angewendet.[/yellow]")
        return False

    if not batch_mode and not Confirm.ask("Änderungen anwenden?"):
        console.print("[dim]Übersprungen.[/dim]")
        return False

    update_data = resolve_ids(paperless, suggestion, tags, correspondents, doc_types, storage_paths)
    console.print("[bold]Wende Änderungen an...[/bold]")
    result = paperless.update_document(doc_id, update_data)
    console.print(f"[green]Dokument #{doc_id} erfolgreich aktualisiert![/green]")
    console.print(f"[dim]Neuer Archivname: {result.get('archived_file_name', '?')}[/dim]")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paperless-NGX Organizer - Dokumente per KI organisieren"
    )
    parser.add_argument("--doc-id", type=int, help="Einzelnes Dokument per ID")
    parser.add_argument("--apply", action="store_true", help="Änderungen wirklich anwenden")
    parser.add_argument("--batch", action="store_true", help="Automatisch ohne Bestätigung")
    parser.add_argument("--limit", type=int, default=5, help="Max Dokumente pro Durchlauf")
    parser.add_argument("--untagged-only", action="store_true", help="Nur Dokumente ohne Tags")
    parser.add_argument("--skip-organized", action="store_true",
                        help="Überspringe Dokumente die bereits Typ+Pfad+Tags haben")
    parser.add_argument("--skip-duplicates", action="store_true",
                        help="Überspringe Dokumente mit Duplikat-Tag")
    args = parser.parse_args()

    paperless_url = os.getenv("PAPERLESS_URL")
    paperless_token = os.getenv("PAPERLESS_TOKEN")
    if not all([paperless_url, paperless_token]):
        console.print("[red]Fehler: .env mit PAPERLESS_URL und PAPERLESS_TOKEN nötig![/red]")
        sys.exit(1)

    try:
        _find_claude_cli()
    except FileNotFoundError:
        console.print("[red]Claude CLI nicht gefunden![/red]")
        sys.exit(1)

    dry_run = not args.apply
    batch_mode = args.batch and args.apply

    console.print(Panel(
        f"[bold]Paperless-NGX Organizer[/bold]\n"
        f"Server: {paperless_url}\n"
        f"KI: Claude Code CLI (dein Abo)\n"
        f"Modus: {'[red]LIVE BATCH[/red]' if batch_mode else '[red]LIVE[/red]' if not dry_run else '[green]TEST[/green]'}",
        border_style="blue"
    ))

    paperless = PaperlessClient(paperless_url, paperless_token)
    analyzer = DocumentAnalyzer()

    console.print("Lade Stammdaten...")
    tags = paperless.get_tags()
    correspondents = paperless.get_correspondents()
    doc_types = paperless.get_document_types()
    storage_paths = paperless.get_storage_paths()

    # Duplikat-Tag ID für Filter
    duplikat_tag_id = next((t["id"] for t in tags if t["name"] == "Duplikat"), None)

    console.print(f"  {len(tags)} Tags | {len(correspondents)} Korrespondenten | {len(doc_types)} Typen | {len(storage_paths)} Speicherpfade")

    if args.doc_id:
        process_document(args.doc_id, paperless, analyzer, tags, correspondents,
                         doc_types, storage_paths, dry_run, batch_mode)
    else:
        # Alle Dokumente laden (seitenweise)
        console.print(f"\nLade Dokumente...")
        documents = paperless._get_all("documents")
        console.print(f"  {len(documents)} Dokumente total")

        # Filter anwenden
        if args.skip_duplicates and duplikat_tag_id:
            before = len(documents)
            documents = [d for d in documents if duplikat_tag_id not in (d.get("tags") or [])]
            console.print(f"  {before - len(documents)} Duplikate übersprungen")

        if args.untagged_only:
            documents = [d for d in documents if not d.get("tags")]
            console.print(f"  {len(documents)} ohne Tags")

        if args.skip_organized:
            documents = [d for d in documents
                         if not (d.get("document_type") and d.get("storage_path")
                                 and d.get("tags"))]
            console.print(f"  {len(documents)} noch nicht vollständig organisiert")

        # Limit anwenden
        if args.limit and len(documents) > args.limit:
            documents = documents[:args.limit]

        console.print(f"  Verarbeite {len(documents)} Dokumente\n")

        applied = 0
        errors = 0
        for i, doc in enumerate(documents, 1):
            console.print(f"[dim]--- Dokument {i}/{len(documents)} ---[/dim]")
            try:
                if process_document(doc["id"], paperless, analyzer, tags, correspondents,
                                    doc_types, storage_paths, dry_run, batch_mode):
                    applied += 1
            except Exception as e:
                errors += 1
                console.print(f"[red]Fehler bei #{doc['id']}: {e}[/red]")

        console.print(f"\n[bold]Fertig! {applied}/{len(documents)} aktualisiert, {errors} Fehler.[/bold]")


if __name__ == "__main__":
    main()
