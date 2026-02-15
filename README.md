# Paperless-NGX Organizer

Zentrale Datei: `paperless_organizer.py`

Das Tool organisiert Dokumente in Paperless-NGX per lokalem LLM (LM Studio), fuehrt Cleanup aus und speichert
alle Laeufe in einer lokalen SQLite-Datenbank.

Der Organizer arbeitet in 2 Phasen:
1. Kontextdaten sammeln (bestehende Korrespondenten/Pfade/Dokumente)
2. Entscheidung pro Dokument mit diesen Kontextdaten

## Start

```powershell
python .\paperless_organizer.py
```

## Wichtige Dateien

- `paperless_organizer.py` - Hauptanwendung (Menues, Organize, Cleanup, Duplikate, Statistiken)
- `taxonomy_tags.json` - feste erlaubte Tags inkl. Synonyme
- `organizer_state.db` - lokale SQLite-DB mit Run-Historie (`runs`, `documents`, `tag_events`, `review_queue`)
- `organizer.log` - laufendes Text-Log
- `legacy/` - alte Einzel-Skripte (nur bei Bedarf nutzen)

## .env

```env
PAPERLESS_URL=https://dein-paperless-server
PAPERLESS_TOKEN=dein-token
LLM_URL=http://10.0.0.220:1234/api/v1/chat
LLM_MODEL=
LLM_API_KEY=
LLM_SYSTEM_PROMPT=
LLM_KEEP_ALIVE=30m
```

`LLM_MODEL` kann leer bleiben. Dann nutzt der Organizer automatisch das aktuell geladene Modell
oder den Server-Default.

Ollama (z. B. Android/Docker) funktioniert ebenfalls:

```env
LLM_URL=http://192.168.178.118:11434/
LLM_MODEL=qwen2.5:7b
LLM_KEEP_ALIVE=30m
AGENT_WORKERS=1
LLM_TIMEOUT=240
```

Optionale Steuerung:

```env
MAX_TAGS_PER_DOC=4
DEFAULT_DRY_RUN=1
AGENT_WORKERS=3
ALLOW_NEW_TAGS=0
ENFORCE_TAG_TAXONOMY=1
AUTO_CREATE_TAXONOMY_TAGS=1
MAX_TOTAL_TAGS=100
RECHECK_ALL_DOCS_IN_AUTO=1
ALLOW_NEW_STORAGE_PATHS=0
AUTO_CLEANUP_AFTER_ORGANIZE=1
USE_ARCHIVE_SERIAL_NUMBER=0
CORRESPONDENT_MATCH_THRESHOLD=0.86
NON_TAXONOMY_DELETE_THRESHOLD=5
REVIEW_TAG_NAME=Manuell-Pruefen
AUTO_APPLY_REVIEW_TAG=1
REVIEW_ON_MEDIUM_CONFIDENCE=0
ENABLE_WEB_HINTS=1
WEB_HINT_TIMEOUT=6
WEB_HINT_MAX_ENTITIES=2
```

## Empfehlung fuer stabilen Vollautomatik-Betrieb

- `ENFORCE_TAG_TAXONOMY=1` + `ALLOW_NEW_TAGS=0` (verhindert Tag-Explosion)
- `AUTO_CREATE_TAXONOMY_TAGS=1` erstellt fehlende Taxonomie-Tags automatisch (mit Farben aus `taxonomy_tags.json`)
- `MAX_TOTAL_TAGS=100` verhindert unkontrolliertes Anwachsen des Tag-Bestands
- `RECHECK_ALL_DOCS_IN_AUTO=1` laesst Menuepunkt `1` alle Dokumente erneut pruefen
- `AUTO_CLEANUP_AFTER_ORGANIZE=1` (raemt nach Batch/Auto-Lauf direkt auf)
- `NON_TAXONOMY_DELETE_THRESHOLD=5` (raeumt alte/falsche Nicht-Taxonomie-Tags automatisch ab)
- `USE_ARCHIVE_SERIAL_NUMBER=0` (schneller und weniger API-Fehler bei ASN)
- bei harten Blockern wird automatisch `Manuell-Pruefen` gesetzt und ein Eintrag in `review_queue` angelegt
- Web-Hinweise helfen bei Anbieter-Konflikten (z. B. GitHub/Google Cloud vs. Arbeit/WBS)
- SQLite ist rein lokal (keine Netzwerk-DB, keine offene Schnittstelle)

