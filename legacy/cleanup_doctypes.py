"""
Paperless-NGX: Delete all document types with document_count == 0.
Uses pagination to load all types and ThreadPoolExecutor for parallel deletion.
"""

import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

BASE_URL = f"{os.getenv('PAPERLESS_URL', 'https://papier.sunnexsrv.de').rstrip('/')}/api"
TOKEN = os.getenv("PAPERLESS_TOKEN", "")
HEADERS = {
    "Authorization": f"Token {TOKEN}",
    "Accept": "application/json",
}


def fetch_all_document_types():
    """Load every document_type page by page (page_size=100)."""
    url = f"{BASE_URL}/document_types/?page_size=100"
    all_types = []

    while url:
        # Enforce HTTPS on pagination URLs
        url = url.replace("http://", "https://")
        print(f"  Fetching: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        all_types.extend(data.get("results", []))
        url = data.get("next")  # None when last page

    return all_types


def delete_document_type(dt):
    """Delete a single document type by id. Returns (id, name, success, detail)."""
    dt_id = dt["id"]
    dt_name = dt["name"]
    try:
        resp = requests.delete(
            f"{BASE_URL}/document_types/{dt_id}/",
            headers=HEADERS,
            timeout=30,
        )
        if resp.status_code == 204:
            return (dt_id, dt_name, True, "deleted")
        else:
            return (dt_id, dt_name, False, f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        return (dt_id, dt_name, False, str(exc))


def main():
    if not TOKEN:
        raise SystemExit("PAPERLESS_TOKEN fehlt in .env")

    # --- 1. Load all document types ---
    print("=" * 60)
    print("STEP 1: Loading all document types ...")
    print("=" * 60)
    all_types = fetch_all_document_types()
    print(f"\n  Total document types loaded: {len(all_types)}\n")

    # --- Partition into empty / non-empty ---
    empty = [dt for dt in all_types if dt.get("document_count", 0) == 0]
    non_empty = [dt for dt in all_types if dt.get("document_count", 0) > 0]

    print(f"  Document types with 0 documents (to delete): {len(empty)}")
    print(f"  Document types with documents (to keep):     {len(non_empty)}\n")

    # --- 2. Delete empty types in parallel ---
    print("=" * 60)
    print("STEP 2: Deleting empty document types (max_workers=10) ...")
    print("=" * 60)

    deleted_count = 0
    failed = []

    if not empty:
        print("  Nothing to delete.\n")
    else:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(delete_document_type, dt): dt for dt in empty}
            for future in as_completed(futures):
                dt_id, dt_name, success, detail = future.result()
                if success:
                    deleted_count += 1
                    print(f"  [OK]   Deleted id={dt_id}  \"{dt_name}\"")
                else:
                    failed.append((dt_id, dt_name, detail))
                    print(f"  [FAIL] id={dt_id}  \"{dt_name}\"  -> {detail}")

        if failed:
            print(f"\n  WARNING: {len(failed)} deletion(s) failed.")

    # --- 3. Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Deleted:   {deleted_count}")
    print(f"  Failed:    {len(failed)}")
    print(f"  Remaining: {len(non_empty)} (all have documents assigned)")
    print()

    # --- 4. Remaining types sorted by document_count desc ---
    print("=" * 60)
    print("REMAINING DOCUMENT TYPES (sorted by document_count DESC)")
    print("=" * 60)
    non_empty.sort(key=lambda dt: dt.get("document_count", 0), reverse=True)

    if not non_empty:
        print("  (none)")
    else:
        print(f"  {'ID':>5}  {'COUNT':>6}  NAME")
        print(f"  {'-'*5}  {'-'*6}  {'-'*40}")
        for dt in non_empty:
            print(f"  {dt['id']:>5}  {dt['document_count']:>6}  {dt['name']}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
