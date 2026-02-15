"""
Batch-delete tags from a Paperless-NGX instance.

Rules:
  - Delete all tags where document_count <= 1,
    EXCEPT a whitelist (case-insensitive).
  - Delete all tags where document_count is 2 or 3
    AND the name looks English (ASCII-only, no German characters).
  - Uses ThreadPoolExecutor(max_workers=10) for fast concurrent DELETEs.
"""

import requests
import concurrent.futures
import re
import time
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("PAPERLESS_URL", "https://papier.sunnexsrv.de")
API_TAGS_URL = f"{BASE_URL}/api/tags/"
TOKEN = os.getenv("PAPERLESS_TOKEN", "")
AUTH_HEADER = {"Authorization": f"Token {TOKEN}"}

WHITELIST = {"kfz", "service", "termin", "aufhebungsvertrag", "ausbildung", "fachinformatiker"}

# German-specific characters (umlauts, sharp s, etc.)
GERMAN_CHARS_RE = re.compile(r'[äöüÄÖÜß]')


def is_ascii_only(name: str) -> bool:
    """Return True if the name contains only ASCII characters."""
    try:
        name.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def load_all_tags() -> list:
    """Load every tag from the API, handling pagination."""
    tags = []
    url = f"{API_TAGS_URL}?page_size=100"
    page = 1
    while url:
        # Always force https
        url = url.replace("http://", "https://")
        print(f"  Fetching page {page}: {url}")
        resp = requests.get(url, headers=AUTH_HEADER, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        tags.extend(results)
        print(f"    -> got {len(results)} tags (total so far: {len(tags)})")
        url = data.get("next")
        page += 1
    return tags


def should_delete(tag: dict) -> bool:
    """Decide whether a tag should be deleted."""
    name = tag["name"]
    doc_count = tag.get("document_count", 0)

    # Rule 1: document_count <= 1, unless whitelisted
    if doc_count <= 1:
        if name.strip().lower() in WHITELIST:
            return False
        return True

    # Rule 2: document_count 2 or 3 AND name is English (ASCII-only)
    if doc_count in (2, 3):
        if is_ascii_only(name) and not GERMAN_CHARS_RE.search(name):
            return True

    return False


def delete_tag(tag: dict) -> dict:
    """Send DELETE for a single tag. Returns a result dict."""
    tag_id = tag["id"]
    tag_name = tag["name"]
    url = f"{API_TAGS_URL}{tag_id}/"
    try:
        resp = requests.delete(url, headers=AUTH_HEADER, timeout=30)
        if resp.status_code == 204:
            return {"id": tag_id, "name": tag_name, "status": "deleted"}
        else:
            return {"id": tag_id, "name": tag_name, "status": f"failed ({resp.status_code})"}
    except Exception as exc:
        return {"id": tag_id, "name": tag_name, "status": f"error ({exc})"}


def main():
    if not TOKEN:
        raise SystemExit("PAPERLESS_TOKEN fehlt in .env")

    print("=" * 60)
    print("Paperless-NGX  --  Batch Tag Deletion")
    print("=" * 60)

    # 1. Load all tags
    print("\n[1/3] Loading all tags ...")
    all_tags = load_all_tags()
    print(f"\n  Total tags loaded: {len(all_tags)}")

    # 2. Decide which to delete
    print("\n[2/3] Analysing tags ...")
    to_delete = [t for t in all_tags if should_delete(t)]
    to_keep = [t for t in all_tags if not should_delete(t)]

    print(f"  Tags to DELETE : {len(to_delete)}")
    print(f"  Tags to KEEP   : {len(to_keep)}")

    if not to_delete:
        print("\nNothing to delete. Done.")
        return

    # Show what will be deleted, grouped by reason
    print("\n  Tags marked for deletion:")
    for t in sorted(to_delete, key=lambda x: x["name"].lower()):
        reason = "doc_count<=1" if t.get("document_count", 0) <= 1 else "English name, doc_count 2-3"
        print(f"    - {t['name']!r}  (id={t['id']}, docs={t.get('document_count',0)}, reason={reason})")

    # 3. Delete concurrently
    print(f"\n[3/3] Deleting {len(to_delete)} tags with 10 workers ...")
    start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(delete_tag, tag): tag for tag in to_delete}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            symbol = "OK" if result["status"] == "deleted" else "FAIL"
            print(f"    [{symbol}] {result['name']!r} (id={result['id']}): {result['status']}")
    elapsed = time.perf_counter() - start

    # Summary
    deleted_count = sum(1 for r in results if r["status"] == "deleted")
    failed_count = len(results) - deleted_count
    remaining = len(all_tags) - deleted_count

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total tags before : {len(all_tags)}")
    print(f"  Deleted           : {deleted_count}")
    print(f"  Failed            : {failed_count}")
    print(f"  Remaining (est.)  : {remaining}")
    print(f"  Elapsed time      : {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
