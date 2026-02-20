"""Paperless-NGX REST API client."""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .config import (
    MAX_RETRIES,
    MAX_WORKERS,
    OWNER_ID,
    console,
    log,
)
from .constants import TAG_COLORS


class PaperlessClient:
    """Vereinter Client fuer die Paperless-NGX REST API."""

    _CACHE_TTL_SEC = 120
    _MIN_WRITE_INTERVAL = 0.15

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token
        self._color_index = 0
        self._cache: dict[str, tuple[float, list]] = {}
        self._last_write_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        })

    def _rate_limit_write(self):
        """Ensure minimum interval between write operations."""
        now = time.monotonic()
        elapsed = now - self._last_write_time
        if elapsed < self._MIN_WRITE_INTERVAL:
            time.sleep(self._MIN_WRITE_INTERVAL - elapsed)
        self._last_write_time = time.monotonic()

    def _get_cached(self, endpoint: str) -> list:
        """Return cached result if still valid, otherwise fetch and cache."""
        now = time.monotonic()
        cached = self._cache.get(endpoint)
        if cached and (now - cached[0]) < self._CACHE_TTL_SEC:
            return cached[1]
        result = self._get_all(endpoint)
        self._cache[endpoint] = (now, result)
        return result

    def invalidate_cache(self, endpoint: str | None = None):
        """Clear cache for a specific endpoint or all."""
        if endpoint:
            self._cache.pop(endpoint, None)
        else:
            self._cache.clear()

    def _get_all(self, endpoint: str, max_results: int = 0) -> list:
        """Alle Eintraege mit Paginierung laden."""
        results = []
        page_num = 0
        url = f"{self.url}/api/{endpoint}/?page_size=100"
        while url:
            url = url.replace("http://", "https://")
            last_exc = None
            resp = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = self.session.get(url, timeout=30)
                    resp.raise_for_status()
                    last_exc = None
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as exc:
                    last_exc = exc
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise
                except requests.exceptions.RequestException:
                    raise
            if resp is None:
                if last_exc:
                    raise last_exc
                raise RuntimeError(f"GET fehlgeschlagen: {url}")
            data = resp.json()
            results.extend(data.get("results", []))
            page_num += 1
            total_count = data.get("count", 0)
            if total_count > 500 and page_num % 5 == 0:
                log.info(f"  Lade {endpoint}: {len(results)}/{total_count} ({len(results) * 100 // max(1, total_count)}%)")
            url = data.get("next")
            if max_results > 0 and len(results) >= max_results:
                results = results[:max_results]
                break
        return results

    # --- Lesen ---
    def get_tags(self) -> list:
        return self._get_cached("tags")

    def get_correspondents(self) -> list:
        return self._get_cached("correspondents")

    def get_document_types(self) -> list:
        return self._get_cached("document_types")

    def get_storage_paths(self) -> list:
        return self._get_cached("storage_paths")

    def get_documents(self) -> list:
        return self._get_all("documents")

    def get_document(self, doc_id: int) -> dict:
        url = f"{self.url}/api/documents/{doc_id}/"
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    backoff = min(30, (2 ** attempt) + random.uniform(0, 1))
                    time.sleep(backoff)
                    continue
                raise
            except requests.exceptions.RequestException:
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Dokument konnte nicht geladen werden: {doc_id}")

    # --- Schreiben ---
    def _with_permissions(self, data: dict) -> dict:
        """Owner und Berechtigungen hinzufuegen."""
        data["owner"] = OWNER_ID
        data["set_permissions"] = {
            "view": {"users": [OWNER_ID], "groups": []},
            "change": {"users": [OWNER_ID], "groups": []},
        }
        return data

    def _next_color(self) -> str:
        color = TAG_COLORS[self._color_index % len(TAG_COLORS)]
        self._color_index += 1
        return color

    @staticmethod
    def _text_color_for_background(color: str) -> str:
        value = (color or "").strip().lower()
        if not value.startswith("#") or len(value) != 7:
            return "#ffffff"
        try:
            r = int(value[1:3], 16)
            g = int(value[3:5], 16)
            b = int(value[5:7], 16)
        except ValueError:
            return "#ffffff"
        luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
        return "#000000" if luminance >= 170 else "#ffffff"

    def _write_with_retry(self, method: str, url: str, data: dict, timeout: int = 30) -> requests.Response:
        """Schreiboperation mit Retry + exponential backoff bei transienten Fehlern."""
        self._rate_limit_write()
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = getattr(self.session, method)(url, json=data, timeout=timeout)
                resp.raise_for_status()
                return resp
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    backoff = min(30, (2 ** attempt) + random.uniform(0, 1))
                    time.sleep(backoff)
                    continue
                raise
            except requests.exceptions.RequestException:
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Schreiboperation fehlgeschlagen: {method.upper()} {url}")

    def update_document(self, doc_id: int, data: dict) -> dict:
        resp = self._write_with_retry("patch", f"{self.url}/api/documents/{doc_id}/", data)
        return resp.json()

    def create_tag(self, name: str, color: str | None = None, text_color: str | None = None) -> dict:
        color = (color or self._next_color()).strip().lower()
        text_color = (text_color or self._text_color_for_background(color)).strip().lower()
        data = {"name": name, "color": color, "text_color": text_color}
        resp = self._write_with_retry("post", f"{self.url}/api/tags/", self._with_permissions(data))
        self.invalidate_cache("tags")
        return resp.json()

    def create_correspondent(self, name: str) -> dict:
        resp = self._write_with_retry("post", f"{self.url}/api/correspondents/", self._with_permissions({"name": name}))
        self.invalidate_cache("correspondents")
        return resp.json()

    def create_document_type(self, name: str) -> dict:
        resp = self._write_with_retry("post", f"{self.url}/api/document_types/", self._with_permissions({"name": name}))
        self.invalidate_cache("document_types")
        return resp.json()

    def create_storage_path(self, name: str, path: str) -> dict:
        resp = self._write_with_retry("post", f"{self.url}/api/storage_paths/", self._with_permissions({"name": name, "path": path}))
        return resp.json()

    def get_next_asn(self) -> int:
        """Naechste freie Archivnummer."""
        resp = self.session.get(
            f"{self.url}/api/documents/?page_size=1&ordering=-archive_serial_number",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data["results"] and data["results"][0].get("archive_serial_number"):
            return data["results"][0]["archive_serial_number"] + 1
        return 1

    # --- Loeschen (einzeln mit Retry) ---
    def _delete_item(self, endpoint: str, item_id: int) -> bool:
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.delete(
                    f"{self.url}/api/{endpoint}/{item_id}/", timeout=15,
                )
                if resp.status_code != 204:
                    log.warning(f"DELETE {endpoint}/{item_id} unerwartet: Status {resp.status_code}")
                return resp.status_code == 204
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                time.sleep(2 * (attempt + 1))
        log.error(f"DELETE {endpoint}/{item_id} fehlgeschlagen nach {MAX_RETRIES} Versuchen: {last_exc}")
        return False

    def delete_tag(self, tag_id: int) -> bool:
        result = self._delete_item("tags", tag_id)
        if result:
            self.invalidate_cache("tags")
        return result

    def delete_correspondent(self, corr_id: int) -> bool:
        result = self._delete_item("correspondents", corr_id)
        if result:
            self.invalidate_cache("correspondents")
        return result

    def delete_document_type(self, type_id: int) -> bool:
        result = self._delete_item("document_types", type_id)
        if result:
            self.invalidate_cache("document_types")
        return result

    # --- Batch-Loeschung (parallel) ---
    def batch_delete(self, endpoint: str, items: list, label: str) -> int:
        """Parallele Loeschung mit ThreadPoolExecutor + Retry."""
        if not items:
            console.print(f"  Keine {label} zu loeschen.")
            return 0

        console.print(f"  Loesche {len(items)} {label}...")
        deleted = 0
        aborted = False
        pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        try:
            futures = {
                pool.submit(self._delete_item, endpoint, item["id"]): item
                for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    if future.result():
                        deleted += 1
                        console.print(f"    [green]Geloescht:[/green] {item['name']} (ID {item['id']})")
                    else:
                        console.print(f"    [red]Fehlgeschlagen:[/red] {item['name']} (ID {item['id']})")
                except Exception as e:
                    console.print(f"    [red]Fehler:[/red] {item['name']}: {e}")
        except KeyboardInterrupt:
            aborted = True
            log.warning("Batch-Loeschung durch Benutzer abgebrochen")
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if not aborted:
                pool.shutdown(wait=True, cancel_futures=False)

        console.print(f"  Fertig: {deleted}/{len(items)} {label} geloescht")
        return deleted
