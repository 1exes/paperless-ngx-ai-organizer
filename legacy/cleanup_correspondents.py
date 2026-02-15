"""
Paperless-NGX Correspondent Cleanup Script
- Deletes all correspondents with 0 documents (using ThreadPoolExecutor for speed)
- Identifies duplicate groups among remaining correspondents
"""

import sys
import io
import time
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_URL = f"{os.getenv('PAPERLESS_URL', 'https://papier.sunnexsrv.de').rstrip('/')}/api"
TOKEN = os.getenv("PAPERLESS_TOKEN", "")
HEADERS = {
    "Authorization": f"Token {TOKEN}",
    "Content-Type": "application/json",
}


def load_all_correspondents():
    """Load ALL correspondents with pagination (page_size=100)."""
    correspondents = []
    url = f"{BASE_URL}/correspondents/?page_size=100"
    page = 1
    while url:
        print(f"  Loading page {page}...")
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        correspondents.extend(data["results"])
        next_url = data.get("next")
        if next_url:
            next_url = next_url.replace("http://", "https://")
        url = next_url
        page += 1
    return correspondents


def delete_correspondent(c):
    """Delete a single correspondent with retries. Returns (id, name, success, error)."""
    cid = c["id"]
    name = c["name"]
    for attempt in range(3):
        try:
            resp = requests.delete(f"{BASE_URL}/correspondents/{cid}/", headers=HEADERS)
            resp.raise_for_status()
            return (cid, name, True, None)
        except Exception as e:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
            else:
                return (cid, name, False, str(e))


def main():
    if not TOKEN:
        raise SystemExit("PAPERLESS_TOKEN fehlt in .env")

    print("=" * 70)
    print("PAPERLESS-NGX CORRESPONDENT CLEANUP")
    print("=" * 70)

    # --- Step 1: Load all correspondents ---
    print("\n[1] Loading all correspondents...")
    all_correspondents = load_all_correspondents()
    total_loaded = len(all_correspondents)
    print(f"    Loaded {total_loaded} correspondents total.\n")

    # --- Step 2: Delete correspondents with 0 documents ---
    empty = [c for c in all_correspondents if c["document_count"] == 0]
    non_empty = [c for c in all_correspondents if c["document_count"] > 0]

    print(f"[2] Found {len(empty)} correspondents with 0 documents -- deleting them...")
    deleted_count = 0
    failed_count = 0
    failed_list = []

    if empty:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(delete_correspondent, c): c for c in empty}
            for future in as_completed(futures):
                result = future.result()
                if result[2]:
                    deleted_count += 1
                    print(f"    Deleted: {result[1]} (id={result[0]})")
                else:
                    failed_count += 1
                    failed_list.append(result)
                    print(f"    FAILED:  {result[1]} (id={result[0]}): {result[3]}")

    print(f"\n    Successfully deleted: {deleted_count}")
    if failed_count:
        print(f"    Failed to delete:    {failed_count}")
    print(f"    Remaining (with docs): {len(non_empty)}")

    # --- Step 3: Identify duplicate groups ---
    print("\n" + "=" * 70)
    print("[3] DUPLICATE GROUP ANALYSIS (correspondents with documents)")
    print("=" * 70)

    remaining = sorted(non_empty, key=lambda c: c["name"].lower())

    groups_def = [
        ("DRK / Deutsches Rotes Kreuz / Wasserwacht / Blutspende", [
            "drk", "deutsches rotes kreuz", "rotes kreuz", "wasserwacht", "blutspende"
        ]),
        ("AOK / AOK PLUS", [
            "aok"
        ]),
        ("msg / msg systems", [
            "msg"
        ]),
        ("Amazon", [
            "amazon"
        ]),
        ("Baader Bank", [
            "baader"
        ]),
        ("Apple", [
            "apple"
        ]),
        ("CHECK24", [
            "check24"
        ]),
        ("AXA", [
            "axa"
        ]),
        ("DHL", [
            "dhl"
        ]),
        ("Dr. med. (various doctors)", [
            "dr. med", "dr.med"
        ]),
        ("1&1", [
            "1&1", "1und1", "1 & 1"
        ]),
    ]

    found_any_group = False
    for group_label, keywords in groups_def:
        matches = []
        for c in remaining:
            name_lower = c["name"].lower()
            for kw in keywords:
                if kw.lower() in name_lower:
                    matches.append(c)
                    break
        if len(matches) >= 1:
            found_any_group = True
            is_dup = len(matches) > 1
            tag = "[DUPLICATES]" if is_dup else "[single]"
            print(f"\n  --- {group_label} {tag} ---")
            for m in sorted(matches, key=lambda x: x["name"]):
                print(f"      id={m['id']:>4}  docs={m['document_count']:>3}  name=\"{m['name']}\"")

    if not found_any_group:
        print("\n  No matching groups found among remaining correspondents.")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total correspondents loaded:       {total_loaded}")
    print(f"  Deleted (0 documents):             {deleted_count}")
    if failed_count:
        print(f"  Failed to delete (server error):   {failed_count}")
    print(f"  Remaining (with documents):        {len(non_empty)}")
    print(f"  Total after cleanup:               {len(non_empty) + failed_count}")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
