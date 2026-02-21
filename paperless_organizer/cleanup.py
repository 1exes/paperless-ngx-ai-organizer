"""Cleanup functions: tags, correspondents, document types."""

from __future__ import annotations

import difflib

import requests
from rich.panel import Panel
from rich.table import Table

from .config import (
    CORRESPONDENT_MERGE_MIN_GROUP_DOCS,
    DELETE_UNUSED_CORRESPONDENTS,
    DELETE_USED_TAGS,
    KEEP_UNUSED_TAXONOMY_TAGS,
    NON_TAXONOMY_DELETE_THRESHOLD,
    REVIEW_TAG_NAME,
    TAG_DELETE_THRESHOLD,
    TAG_ENGLISH_THRESHOLD,
    TAXONOMY_FILE,
    console,
    log,
)
from .constants import ALLOWED_DOC_TYPES, TAG_WHITELIST
from .client import PaperlessClient
from .taxonomy import TagTaxonomy
from .utils import (
    _correspondent_core_name,
    _is_correspondent_duplicate_name,
    _normalize_correspondent_name,
    _normalize_tag_name,
    _strip_diacritics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_ascii_only(name: str) -> bool:
    """Prueft ob Name nur ASCII-Zeichen enthaelt (englisch)."""
    try:
        name.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def _correspondent_keep_sort_key(item: dict) -> tuple:
    """Waehlt den stabilsten Kanonischen Namen innerhalb einer Duplikatgruppe."""
    name = (item.get("name") or "").strip()
    name_lower = name.lower()
    core = _correspondent_core_name(name)
    doc_count = int(item.get("document_count", 0) or 0)

    quality = 0
    if " via " not in name_lower:
        quality += 2
    if "@" not in name_lower:
        quality += 2
    if "http://" not in name_lower and "https://" not in name_lower:
        quality += 2
    if len(core.split()) >= 2:
        quality += 1
    if len(name) <= 70:
        quality += 1

    return (doc_count, quality, -len(name))


def _find_correspondent_groups(correspondents: list) -> dict:
    """Findet echte Namensduplikate (generisch, konservativ)."""
    ordered = sorted(correspondents, key=lambda c: int(c.get("document_count", 0) or 0), reverse=True)
    by_id = {int(c["id"]): c for c in ordered if c.get("id") is not None}
    unassigned = set(by_id.keys())
    groups: dict[str, list] = {}
    group_no = 1

    for seed in ordered:
        seed_id = int(seed["id"])
        if seed_id not in unassigned:
            continue

        cluster_ids = {seed_id}
        queue = [seed_id]
        unassigned.remove(seed_id)

        while queue:
            current_id = queue.pop(0)
            current = by_id[current_id]
            for cand_id in list(unassigned):
                cand = by_id[cand_id]
                if _is_correspondent_duplicate_name(current.get("name", ""), cand.get("name", "")):
                    cluster_ids.add(cand_id)
                    queue.append(cand_id)
                    unassigned.remove(cand_id)

        cluster = [by_id[cid] for cid in cluster_ids]
        if len(cluster) > 1:
            cluster.sort(key=_correspondent_keep_sort_key, reverse=True)
            label = cluster[0].get("name") or f"Duplikatgruppe {group_no}"
            groups[label] = cluster
            group_no += 1

    return groups


# ---------------------------------------------------------------------------
# Cleanup functions
# ---------------------------------------------------------------------------

def cleanup_tags(paperless: PaperlessClient, dry_run: bool = True):
    """Tags aufraeumen: Taxonomie + Whitelist + Schwellwerte."""
    console.print(Panel("[bold]Tag-Aufraeumung[/bold]", border_style="yellow"))
    log.info("[bold]CLEANUP TAGS[/bold] gestartet")

    tags = paperless.get_tags()
    log.info(f"  {len(tags)} Tags geladen")

    taxonomy = TagTaxonomy(TAXONOMY_FILE)
    taxonomy_set = {_normalize_tag_name(t) for t in taxonomy.canonical_tags}
    protected_tags = {_normalize_tag_name(t) for t in TAG_WHITELIST}
    protected_tags.add(_normalize_tag_name(REVIEW_TAG_NAME))
    protected_tags.add(_normalize_tag_name("Duplikat"))

    to_delete = []
    for t in tags:
        name = t["name"]
        doc_count = t.get("document_count", 0)
        normalized_name = _normalize_tag_name(name)
        in_taxonomy = normalized_name in taxonomy_set if taxonomy_set else True

        if normalized_name in protected_tags:
            continue
        if KEEP_UNUSED_TAXONOMY_TAGS and in_taxonomy and doc_count == 0:
            continue

        reason = ""
        if doc_count == 0:
            reason = "0 Dokumente" if in_taxonomy else "nicht in Taxonomie, 0 Dokumente"
        elif DELETE_USED_TAGS:
            if not in_taxonomy and doc_count <= NON_TAXONOMY_DELETE_THRESHOLD:
                reason = f"nicht in Taxonomie, <= {NON_TAXONOMY_DELETE_THRESHOLD} Dokumente"
            elif TAG_DELETE_THRESHOLD > 0 and doc_count <= TAG_DELETE_THRESHOLD:
                reason = f"<= {TAG_DELETE_THRESHOLD} Dokumente"
            elif (
                TAG_ENGLISH_THRESHOLD > 0
                and not in_taxonomy
                and doc_count <= TAG_ENGLISH_THRESHOLD
                and _is_ascii_only(name)
            ):
                reason = f"ASCII/englisch, <= {TAG_ENGLISH_THRESHOLD} Dokumente"

        if reason:
            to_delete.append((t, reason))

    console.print(f"[yellow]Zum Loeschen markiert: {len(to_delete)}[/yellow]")

    if to_delete:
        table = Table(title="Tags zum Loeschen", show_header=True)
        table.add_column("Name", style="red")
        table.add_column("Dokumente", justify="right")
        table.add_column("Grund")

        for item, reason in sorted(to_delete, key=lambda x: x[0]["name"].lower()):
            doc_count = item.get("document_count", 0)
            table.add_row(item["name"], str(doc_count), reason)
        console.print(table)

        if not dry_run:
            deleted = paperless.batch_delete("tags", [item for item, _ in to_delete], "Tags")
            log.info(f"  [green]{deleted} Tags geloescht[/green]")

    remaining = paperless.get_tags()
    log.info(f"  CLEANUP TAGS fertig - {len(remaining)} Tags verbleibend")
    if dry_run:
        console.print("[yellow]TESTMODUS: Nichts geloescht[/yellow]")


def cleanup_correspondents(paperless: PaperlessClient, dry_run: bool = True):
    """Konservatives Korrespondenten-Cleanup: optional ungenutzte loeschen + echte Duplikate mergen."""
    console.print(Panel("[bold]Korrespondenten-Aufraeumung[/bold]", border_style="cyan"))
    log.info("[bold]CLEANUP KORRESPONDENTEN[/bold] gestartet")

    correspondents = paperless.get_correspondents()
    log.info(f"  {len(correspondents)} Korrespondenten geladen")

    unused = [c for c in correspondents if c.get("document_count", 0) == 0]
    console.print(f"[yellow]Ungenutzt (0 Dokumente): {len(unused)}[/yellow]")

    if unused and not dry_run:
        if DELETE_UNUSED_CORRESPONDENTS:
            deleted = paperless.batch_delete("correspondents", unused, "Korrespondenten")
            console.print(f"[green]{deleted} ungenutzte Korrespondenten geloescht[/green]")
        else:
            console.print("[cyan]Info:[/cyan] Ungenutzte Korrespondenten bleiben erhalten (Policy).")

    groups = _find_correspondent_groups(correspondents)
    duplicates = {}
    for name, items in groups.items():
        total_docs = sum(int(c.get("document_count", 0) or 0) for c in items)
        if len(items) > 1 and total_docs >= CORRESPONDENT_MERGE_MIN_GROUP_DOCS:
            duplicates[name] = items

    if duplicates:
        console.print(f"\n[bold]{len(duplicates)} Duplikat-Gruppen gefunden:[/bold]")
        # Load documents once before processing all groups
        all_docs = paperless.get_documents() if not dry_run else []
        for group_name, items in sorted(duplicates.items()):
            total = sum(c.get("document_count", 0) for c in items)
            items.sort(key=_correspondent_keep_sort_key, reverse=True)
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
                for m in merge:
                    docs_to_move = [d for d in all_docs if d.get("correspondent") == m["id"]]
                    for doc in docs_to_move:
                        paperless.update_document(doc["id"], {"correspondent": keep["id"]})
                        console.print(f"    Dokument #{doc['id']}: {m['name']} -> {keep['name']}")
                    paperless.delete_correspondent(m["id"])
                    console.print(f"    [green]Geloescht:[/green] {m['name']} (gemergt in {keep['name']})")

    remaining = paperless.get_correspondents()
    log.info(f"  CLEANUP KORRESPONDENTEN fertig - {len(remaining)} verbleibend")
    if dry_run:
        console.print("[yellow]TESTMODUS: Nichts geaendert[/yellow]")


def cleanup_document_types(paperless: PaperlessClient, dry_run: bool = True):
    """Leere Dokumenttypen loeschen."""
    console.print(Panel("[bold]Dokumenttypen-Aufraeumung[/bold]", border_style="magenta"))
    log.info("[bold]CLEANUP DOKUMENTTYPEN[/bold] gestartet")

    types = paperless.get_document_types()
    log.info(f"  {len(types)} Dokumenttypen geladen")

    # 1) Canonical-Namen aus ALLOWED_DOC_TYPES erzwingen
    allowed_lower = {name.lower(): name for name in ALLOWED_DOC_TYPES}
    by_exact_name = {t["name"]: t for t in types}
    to_normalize: list[tuple[dict, dict]] = []

    for t in types:
        current_name = (t.get("name") or "").strip()
        key = current_name.lower()
        canonical_name = allowed_lower.get(key)
        if not canonical_name or current_name == canonical_name:
            continue

        target = by_exact_name.get(canonical_name)
        if target is None and not dry_run:
            try:
                target = paperless.create_document_type(canonical_name)
                types.append(target)
                by_exact_name[canonical_name] = target
                console.print(f"  [green]+ Canonical-Dokumenttyp erstellt:[/green] {canonical_name}")
            except requests.exceptions.HTTPError:
                fresh = paperless.get_document_types()
                types = fresh
                by_exact_name = {x["name"]: x for x in types}
                target = by_exact_name.get(canonical_name)

        if target and target["id"] != t["id"]:
            to_normalize.append((t, target))

    if to_normalize:
        table = Table(title="Dokumenttypen-Normalisierung", show_header=True)
        table.add_column("Von", style="yellow")
        table.add_column("Nach", style="green")
        table.add_column("Dokumente", justify="right")
        for src, dst in to_normalize:
            table.add_row(src["name"], dst["name"], str(src.get("document_count", 0)))
        console.print(table)

        if not dry_run:
            all_docs = paperless.get_documents()
            for src, dst in to_normalize:
                docs_to_move = [d for d in all_docs if d.get("document_type") == src["id"]]
                for doc in docs_to_move:
                    paperless.update_document(doc["id"], {"document_type": dst["id"]})
                    console.print(f"    Dokument #{doc['id']}: {src['name']} -> {dst['name']}")
                paperless.delete_document_type(src["id"])
                console.print(f"    [green]Geloescht:[/green] {src['name']} (normalisiert auf {dst['name']})")

            types = paperless.get_document_types()
            log.info(f"  [green]{len(to_normalize)} Dokumenttypen normalisiert[/green]")

    unused = [t for t in types if t.get("document_count", 0) == 0]
    console.print(f"[yellow]Ungenutzt (0 Dokumente): {len(unused)}[/yellow]")

    if unused:
        table = Table(title="Dokumenttypen zum Loeschen", show_header=True)
        table.add_column("ID", width=6)
        table.add_column("Name")
        for t in unused:
            table.add_row(str(t["id"]), t["name"])
        console.print(table)

        if not dry_run:
            deleted = paperless.batch_delete("document_types", unused, "Dokumenttypen")
            console.print(f"[green]{deleted} Dokumenttypen geloescht[/green]")

    remaining = paperless.get_document_types()
    console.print(f"\n[bold]Verbleibend: {len(remaining)} Dokumenttypen[/bold]")

    if remaining:
        table = Table(title="Verbleibende Dokumenttypen", show_header=True)
        table.add_column("ID", width=6)
        table.add_column("Name")
        table.add_column("Dokumente", justify="right")
        for t in sorted(remaining, key=lambda x: x.get("document_count", 0), reverse=True):
            table.add_row(str(t["id"]), t["name"], str(t.get("document_count", 0)))
        console.print(table)

    log.info(f"  CLEANUP DOKUMENTTYPEN fertig - {len(remaining)} verbleibend")
    if dry_run:
        console.print("[yellow]TESTMODUS: Nichts geloescht[/yellow]")


def cleanup_all(paperless: PaperlessClient, dry_run: bool = True):
    """Alle drei Cleanup-Funktionen nacheinander."""
    cleanup_tags(paperless, dry_run)
    console.print()
    cleanup_correspondents(paperless, dry_run)
    console.print()
    cleanup_document_types(paperless, dry_run)
