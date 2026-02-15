"""
Paperless-NGX Duplicate Document Finder
READ-ONLY: This script only reads and analyzes documents. It does NOT modify or delete anything.
"""

import requests
import json
import os
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_URL = f"{os.getenv('PAPERLESS_URL', 'https://papier.sunnexsrv.de').rstrip('/')}/api"
TOKEN = os.getenv("PAPERLESS_TOKEN", "")
HEADERS = {
    "Authorization": f"Token {TOKEN}",
    "Accept": "application/json",
}
REPORT_PATH = os.path.join(os.path.dirname(__file__), "duplikate_bericht.txt")
PAGE_SIZE = 100
FIELDS = "id,title,original_file_name,created,content"


def load_all_documents():
    """Load ALL documents from Paperless-NGX using pagination."""
    documents = []
    url = f"{BASE_URL}/documents/?page_size={PAGE_SIZE}&fields={FIELDS}"
    page = 1

    while url:
        # Always enforce HTTPS in pagination URLs
        url = url.replace("http://", "https://")

        print(f"  Loading page {page}...")
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        documents.extend(results)

        total = data.get("count", "?")
        print(f"    -> Got {len(results)} documents (total so far: {len(documents)}/{total})")

        url = data.get("next")
        page += 1

    return documents


def find_duplicates_by_filename(documents):
    """Group documents by original_file_name (exact match)."""
    groups = defaultdict(list)
    for doc in documents:
        fname = doc.get("original_file_name")
        if fname:
            groups[fname].append(doc)
    # Only keep groups with more than one document
    return {k: v for k, v in groups.items() if len(v) > 1}


def find_duplicates_by_title(documents):
    """Group documents by title (case-insensitive)."""
    groups = defaultdict(list)
    for doc in documents:
        title = doc.get("title", "")
        if title:
            groups[title.lower()].append(doc)
    return {k: v for k, v in groups.items() if len(v) > 1}


def format_doc(doc):
    """Format a single document for the report."""
    doc_id = doc.get("id", "?")
    title = doc.get("title", "(no title)")
    fname = doc.get("original_file_name", "(no filename)")
    created = doc.get("created", "(no date)")
    # Truncate content preview
    content = (doc.get("content") or "")[:150].replace("\n", " ").strip()
    if len(doc.get("content") or "") > 150:
        content += "..."
    return (
        f"    ID: {doc_id}\n"
        f"    Title: {title}\n"
        f"    Filename: {fname}\n"
        f"    Created: {created}\n"
        f"    Content preview: {content}\n"
    )


def write_report(documents, filename_dupes, title_dupes):
    """Write the full report to a text file."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 80)
    lines.append("PAPERLESS-NGX DUPLICATE DOCUMENT REPORT")
    lines.append(f"Generated: {now}")
    lines.append(f"Server: https://papier.sunnexsrv.de")
    lines.append(f"Total documents analyzed: {len(documents)}")
    lines.append("READ-ONLY analysis - no documents were modified or deleted.")
    lines.append("=" * 80)
    lines.append("")

    # --- Section 1: Same original_file_name ---
    lines.append("-" * 80)
    lines.append("SECTION 1: Documents with the SAME original_file_name")
    lines.append(f"  Found {len(filename_dupes)} group(s) with duplicate filenames")
    total_fn_dupes = sum(len(v) for v in filename_dupes.values())
    lines.append(f"  Total documents involved: {total_fn_dupes}")
    lines.append("-" * 80)
    lines.append("")

    for i, (fname, docs) in enumerate(sorted(filename_dupes.items()), 1):
        lines.append(f"  Group {i}: \"{fname}\" ({len(docs)} documents)")
        lines.append("")
        for doc in sorted(docs, key=lambda d: d.get("id", 0)):
            lines.append(format_doc(doc))
        lines.append("")

    # --- Section 2: Same title (case-insensitive) ---
    lines.append("-" * 80)
    lines.append("SECTION 2: Documents with the SAME title (case-insensitive)")
    lines.append(f"  Found {len(title_dupes)} group(s) with duplicate titles")
    total_t_dupes = sum(len(v) for v in title_dupes.values())
    lines.append(f"  Total documents involved: {total_t_dupes}")
    lines.append("-" * 80)
    lines.append("")

    for i, (title_key, docs) in enumerate(sorted(title_dupes.items()), 1):
        display_title = docs[0].get("title", title_key)
        lines.append(f"  Group {i}: \"{display_title}\" ({len(docs)} documents)")
        lines.append("")
        for doc in sorted(docs, key=lambda d: d.get("id", 0)):
            lines.append(format_doc(doc))
        lines.append("")

    report_text = "\n".join(lines)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def main():
    if not TOKEN:
        raise SystemExit("PAPERLESS_TOKEN fehlt in .env")

    print("=" * 60)
    print("Paperless-NGX Duplicate Document Finder")
    print("READ-ONLY - no documents will be modified or deleted")
    print("=" * 60)
    print()

    # Step 1: Load all documents
    print("[1/4] Loading all documents...")
    documents = load_all_documents()
    print(f"  => Loaded {len(documents)} documents total.\n")

    # Step 2: Find filename duplicates
    print("[2/4] Searching for duplicate filenames...")
    filename_dupes = find_duplicates_by_filename(documents)
    fn_count = sum(len(v) for v in filename_dupes.values())
    print(f"  => Found {len(filename_dupes)} groups ({fn_count} documents) with duplicate filenames.\n")

    # Step 3: Find title duplicates
    print("[3/4] Searching for duplicate titles (case-insensitive)...")
    title_dupes = find_duplicates_by_title(documents)
    t_count = sum(len(v) for v in title_dupes.values())
    print(f"  => Found {len(title_dupes)} groups ({t_count} documents) with duplicate titles.\n")

    # Step 4: Write report
    print("[4/4] Writing report...")
    write_report(documents, filename_dupes, title_dupes)
    print(f"  => Report written to: {REPORT_PATH}\n")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total documents:              {len(documents)}")
    print(f"  Duplicate filename groups:    {len(filename_dupes)} groups ({fn_count} docs)")
    print(f"  Duplicate title groups:       {len(title_dupes)} groups ({t_count} docs)")
    print()

    if filename_dupes:
        print("  Top filename duplicate groups:")
        for fname, docs in sorted(filename_dupes.items(), key=lambda x: -len(x[1]))[:10]:
            ids = [str(d["id"]) for d in docs]
            print(f"    [{len(docs)}x] \"{fname}\" -> IDs: {', '.join(ids)}")
        print()

    if title_dupes:
        print("  Top title duplicate groups:")
        for title_key, docs in sorted(title_dupes.items(), key=lambda x: -len(x[1]))[:10]:
            display_title = docs[0].get("title", title_key)
            ids = [str(d["id"]) for d in docs]
            print(f"    [{len(docs)}x] \"{display_title}\" -> IDs: {', '.join(ids)}")
        print()

    print("Done. No documents were modified or deleted.")


if __name__ == "__main__":
    main()
