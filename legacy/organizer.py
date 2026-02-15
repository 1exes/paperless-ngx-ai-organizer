#!/usr/bin/env python3
"""Paperless-NGX Organizer helper."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("paperless_organizer")


@dataclasses.dataclass
class Config:
    base_url: str
    token: str
    max_tags: int = 60
    workers: int = 4
    timeout: int = 25
    enable_web_check: bool = False


class PaperlessClient:
    def __init__(self, config: Config) -> None:
        self.config = config

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{path}"

    def _request_json(self, method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Token {self.config.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            LOG.error("HTTP %s for %s %s: %s", exc.code, method, url, body)
            raise

    def get_document(self, document_id: int) -> Dict[str, Any]:
        return self._request_json("GET", self._url(f"/api/documents/{document_id}/"))

    def patch_document(self, document_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request_json("PATCH", self._url(f"/api/documents/{document_id}/"), payload)

    def _list_entities(self, endpoint: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        url: Optional[str] = self._url(endpoint)
        while url:
            data = self._request_json("GET", url)
            if isinstance(data, dict) and "results" in data:
                items.extend(data["results"])
                url = data.get("next")
            elif isinstance(data, list):
                items.extend(data)
                url = None
            else:
                url = None
        return items

    def resolve_id(self, endpoint: str, label: str) -> Optional[int]:
        label_clean = label.strip().casefold()
        for item in self._list_entities(endpoint):
            name = str(item.get("name") or item.get("path") or "").strip().casefold()
            if name == label_clean:
                return int(item["id"])
        return None

    def resolve_tag_ids(self, names: List[str]) -> List[int]:
        all_tags = self._list_entities("/api/tags/")
        lookup = {str(t.get("name", "")).strip().casefold(): int(t["id"]) for t in all_tags}
        resolved: List[int] = []
        for n in names:
            tag_id = lookup.get(n.strip().casefold())
            if tag_id is not None:
                resolved.append(tag_id)
            else:
                LOG.warning("Tag not found and skipped: %s", n)
        return resolved


def extract_keywords(content: str, limit: int = 8) -> List[str]:
    words = re.findall(r"[A-Za-zÄÖÜäöüß][\w\-]{2,}", content)
    freq: Dict[str, int] = {}
    for w in words:
        key = w.lower()
        if key in {"und", "oder", "der", "die", "das", "invoice", "document"}:
            continue
        freq[key] = freq.get(key, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [t[0] for t in top]


def web_verify_claim(title: str, content: str, timeout: int = 8) -> str:
    query_terms = " ".join([title] + extract_keywords(content, limit=5)).strip()
    query = urllib.parse.urlencode({"q": query_terms, "format": "json", "no_html": 1})
    url = f"https://api.duckduckgo.com/?{query}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"Web-Check nicht verfügbar: {exc}"

    abstract = (data.get("AbstractText") or "").strip()
    heading = (data.get("Heading") or "").strip()
    if abstract:
        return f"Web-Check: {heading} – {abstract[:240]}"
    return "Web-Check: Keine klare externe Bestätigung gefunden (nur Plausibilitätsprüfung)."


def sanitize_proposal(proposal: Dict[str, Any], max_tags: int) -> Dict[str, Any]:
    cleaned = dict(proposal)
    tags = cleaned.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    unique_tags: List[str] = []
    seen = set()
    for t in tags:
        key = t.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            unique_tags.append(t.strip())
    cleaned["tags"] = unique_tags[:max_tags]

    if "archive_serial_number" in cleaned and cleaned["archive_serial_number"] in {"", "Keine", "None", None}:
        cleaned["archive_serial_number"] = None
    return cleaned


def build_patch_payload(client: PaperlessClient, proposal: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if "title" in proposal:
        payload["title"] = proposal["title"]
    if proposal.get("tags"):
        payload["tags"] = client.resolve_tag_ids(proposal["tags"])
    if proposal.get("correspondent"):
        cid = client.resolve_id("/api/correspondents/", proposal["correspondent"])
        if cid is not None:
            payload["correspondent"] = cid
    if proposal.get("document_type"):
        did = client.resolve_id("/api/document_types/", proposal["document_type"])
        if did is not None:
            payload["document_type"] = did
    if proposal.get("storage_path"):
        sid = client.resolve_id("/api/storage_paths/", proposal["storage_path"])
        if sid is not None:
            payload["storage_path"] = sid
    if "archive_serial_number" in proposal:
        payload["archive_serial_number"] = proposal["archive_serial_number"]
    return payload


def process_document(client: PaperlessClient, document_id: int, proposal: Dict[str, Any]) -> Tuple[int, str]:
    cleaned = sanitize_proposal(proposal, max_tags=client.config.max_tags)
    if client.config.enable_web_check:
        current_doc = client.get_document(document_id)
        cleaned["reason_web_check"] = web_verify_claim(
            title=cleaned.get("title") or current_doc.get("title") or "",
            content=current_doc.get("content") or "",
        )
    patch_payload = build_patch_payload(client, cleaned)
    if not patch_payload:
        return document_id, "übersprungen (keine auflösbaren Änderungen)"
    client.patch_document(document_id, patch_payload)
    return document_id, "ok"


def run_batch(client: PaperlessClient, plan: Dict[int, Dict[str, Any]]) -> None:
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=client.config.workers) as pool:
        futures = {pool.submit(process_document, client, doc_id, proposal): doc_id for doc_id, proposal in plan.items()}
        for future in concurrent.futures.as_completed(futures):
            doc_id = futures[future]
            try:
                _, msg = future.result()
                LOG.info("Dokument #%s: %s", doc_id, msg)
            except Exception as exc:  # noqa: BLE001
                LOG.error("Fehler bei #%s: %s", doc_id, exc)
    LOG.info("Fertig in %.1fs", time.time() - start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paperless-NGX Organizer helper")
    parser.add_argument("--base-url", default=os.getenv("PAPERLESS_BASE_URL", ""), required=False)
    parser.add_argument("--token", default=os.getenv("PAPERLESS_TOKEN", ""), required=False)
    parser.add_argument("--plan", required=True, help="JSON-Datei mit Dokument-ID -> Vorschlag")
    parser.add_argument("--max-tags", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--enable-web-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.base_url or not args.token:
        raise SystemExit("Bitte --base-url und --token setzen (oder PAPERLESS_BASE_URL / PAPERLESS_TOKEN).")
    with open(args.plan, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    plan = {int(k): v for k, v in loaded.items()}
    client = PaperlessClient(
        Config(
            base_url=args.base_url,
            token=args.token,
            max_tags=args.max_tags,
            workers=args.workers,
            enable_web_check=args.enable_web_check,
        )
    )
    run_batch(client, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
