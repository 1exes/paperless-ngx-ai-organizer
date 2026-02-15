"""Schnelle Aufräumung - sanfter mit Retry und weniger Threads."""
import time
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("PAPERLESS_TOKEN", "")
BASE = os.getenv("PAPERLESS_URL", "https://papier.sunnexsrv.de").rstrip("/")
HEADERS = {"Authorization": f"Token {TOKEN}"}

KEEP_TAGS = {"kfz", "service", "termin", "aufhebungsvertrag", "ausbildung", "fachinformatiker"}


def get_all(endpoint):
    results = []
    url = f"{BASE}/api/{endpoint}/?page_size=100"
    while url:
        url = url.replace("http://", "https://")
        r = requests.get(url, headers=HEADERS, timeout=30)
        data = r.json()
        results.extend(data.get("results", []))
        url = data.get("next")
    return results


def delete_item(endpoint, item_id):
    for attempt in range(3):
        try:
            r = requests.delete(f"{BASE}/api/{endpoint}/{item_id}/", headers=HEADERS, timeout=15)
            return r.status_code == 204
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(2 * (attempt + 1))
    return False


def batch_delete(endpoint, items, label):
    if not items:
        print(f"  Keine {label} zu löschen")
        return 0
    print(f"  Lösche {len(items)} {label}...")
    deleted = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(delete_item, endpoint, i["id"]): i for i in items}
        for f in as_completed(futures):
            try:
                if f.result():
                    deleted += 1
                if deleted % 25 == 0 and deleted > 0:
                    print(f"    ...{deleted}/{len(items)}")
            except Exception:
                pass
    print(f"  Fertig: {deleted}/{len(items)} {label} gelöscht")
    return deleted


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("PAPERLESS_TOKEN fehlt in .env")

    # === TAGS ===
    print("=== TAGS ===")
    tags = get_all("tags")
    print(f"  Aktuell: {len(tags)} Tags")

    to_delete = [t for t in tags
                 if t["document_count"] <= 5
                 and t["name"].lower() not in KEEP_TAGS]
    batch_delete("tags", to_delete, "Tags")

    # === KORRESPONDENTEN ===
    print("\n=== KORRESPONDENTEN ===")
    corrs = get_all("correspondents")
    print(f"  Aktuell: {len(corrs)} Korrespondenten")

    unused_corrs = [c for c in corrs if c["document_count"] == 0]
    batch_delete("correspondents", unused_corrs, "Korrespondenten")

    # === DOKUMENTTYPEN ===
    print("\n=== DOKUMENTTYPEN ===")
    types = get_all("document_types")
    print(f"  Aktuell: {len(types)} Dokumenttypen")

    unused_types = [t for t in types if t["document_count"] == 0]
    batch_delete("document_types", unused_types, "Dokumenttypen")

    # === ZUSAMMENFASSUNG ===
    print("\n" + "=" * 50)
    t = get_all("tags")
    c = get_all("correspondents")
    d = get_all("document_types")
    print(f"ERGEBNIS:")
    print(f"  Tags:            {len(t)}")
    print(f"  Korrespondenten: {len(c)}")
    print(f"  Dokumenttypen:   {len(d)}")
    print(f"\nVerbleibende Tags:")
    for tag in sorted(t, key=lambda x: x["document_count"], reverse=True):
        print(f"  {tag['document_count']:4d}x  {tag['name']}")
