# Paperless-NGX Organizer

Zentrale Datei: `paperless_organizer.py`

Das Tool organisiert Dokumente in Paperless-NGX per lokalem LLM (Ollama/LM Studio/OpenAI-kompatibel), fuehrt Cleanup aus und speichert
alle Laeufe in einer lokalen SQLite-Datenbank.

Der Organizer arbeitet in 2 Phasen:
1. Kontextdaten sammeln (bestehende Korrespondenten/Pfade/Dokumente)
2. Entscheidung pro Dokument mit diesen Kontextdaten

Hinweis: Bei `Alles sortieren` fragt das Tool pro Lauf, ob wirklich alle Dokumente neu geprueft werden sollen.
Hinweis: Es gibt einen Live-Watch-Modus im Hauptmenue (`8`), der regelmaessig neue Dokumente erkennt und automatisch verarbeitet.
Hinweis: Es gibt einen Vollautomatik-Modus im Hauptmenue (`9`), der dauerhaft alles in einem Modus ausfuehrt (Sortieren + Watch + Wartung).

## Start

```powershell
python .\paperless_organizer.py
```

## Wichtige Dateien

- `paperless_organizer.py` - Hauptanwendung (Menues, Organize, Cleanup, Duplikate, Statistiken)
- `taxonomy_tags.json` - feste erlaubte Tags inkl. Synonyme
- `organizer_state.db` - lokale SQLite-DB mit Run-Historie (`runs`, `documents`, `tag_events`, `review_queue`)
- `learning_profile.json` - lernendes Profil (Beschaeftigungsverlauf, bekannte Privat-/Firmenfahrzeuge, Hinweise)
- `learning_examples.jsonl` - bestaetigte Few-Shot-Beispiele fuer stabilere Entscheidungen bei kleinen Modellen
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

Der Organizer lernt lokal aus bestaetigten Updates:
- Beschaeftigungsverlauf (z. B. Wechsel von `msg systems ag` zu `WBS TRAINING AG`)
- bekannte Privat- und Firmenfahrzeuge (fuer Guardrails)
- aehnliche, bestaetigte Dokumententscheidungen als Few-Shot-Kontext fuer kleine LLMs
- Kontext landet in `learning_profile.json` und wird im Prompt wiederverwendet.

Ollama (z. B. Android/Docker) funktioniert ebenfalls:

```env
LLM_URL=http://192.168.178.118:11434/
LLM_MODEL=qwen2.5:7b
LLM_KEEP_ALIVE=30m
AGENT_WORKERS=1
LLM_CONNECT_TIMEOUT=6
LLM_TIMEOUT=90
LLM_COMPACT_TIMEOUT=35
LLM_COMPACT_TIMEOUT_RETRY=90
LLM_RETRY_COUNT=1
LLM_MAX_TOKENS=320
LLM_COMPACT_MAX_TOKENS=220
LLM_COMPACT_PROMPT_MAX_TAGS=24
LLM_COMPACT_PROMPT_MAX_PATHS=20
LLM_VERIFY_ON_LOW_CONFIDENCE=0
ENABLE_WEB_HINTS=0
SKIP_RECENT_LLM_ERRORS_MINUTES=240
SKIP_RECENT_LLM_ERRORS_THRESHOLD=1
LIVE_WATCH_INTERVAL_SEC=45
LIVE_WATCH_CONTEXT_REFRESH_CYCLES=5
LIVE_WATCH_COMPACT_FIRST=1
MAX_CONTEXT_EMPLOYER_HINTS=20
WORK_CORR_EMPLOYER_MIN_DOCS=8
AUTOPILOT_INTERVAL_SEC=45
AUTOPILOT_CONTEXT_REFRESH_CYCLES=5
AUTOPILOT_START_WITH_AUTO_ORGANIZE=1
AUTOPILOT_RECHECK_ALL_ON_START=0
AUTOPILOT_CLEANUP_EVERY_CYCLES=10
AUTOPILOT_DUPLICATE_SCAN_EVERY_CYCLES=0
AUTOPILOT_MAX_NEW_DOCS_PER_CYCLE=25
WATCH_RECONNECT_ERROR_THRESHOLD=3
WATCH_ERROR_BACKOFF_BASE_SEC=10
WATCH_ERROR_BACKOFF_MAX_SEC=180
```

Optionale Steuerung:

```env
MAX_TAGS_PER_DOC=5
DEFAULT_DRY_RUN=1
AGENT_WORKERS=3
LLM_MAX_TOKENS=320
LLM_TIMEOUT=90
LLM_COMPACT_TIMEOUT=35
LLM_COMPACT_TIMEOUT_RETRY=90
LLM_RETRY_COUNT=1
ALLOW_NEW_TAGS=0
ALLOW_NEW_CORRESPONDENTS=0
DELETE_UNUSED_CORRESPONDENTS=0
CORRESPONDENT_MERGE_MIN_GROUP_DOCS=2
CORRESPONDENT_MERGE_MIN_NAME_SIMILARITY=0.96
LEARNING_EXAMPLE_LIMIT=3
LEARNING_MAX_EXAMPLES=2000
ENABLE_LEARNING_PRIORS=1
LEARNING_PRIOR_MAX_HINTS=2
LEARNING_PRIOR_MIN_SAMPLES=3
LEARNING_PRIOR_MIN_RATIO=0.70
LEARNING_PRIOR_ENABLE_TAG_SUGGESTION=0
ENFORCE_TAG_TAXONOMY=1
AUTO_CREATE_TAXONOMY_TAGS=1
KEEP_UNUSED_TAXONOMY_TAGS=1
MAX_TOTAL_TAGS=100
RECHECK_ALL_DOCS_IN_AUTO=0
ALLOW_NEW_STORAGE_PATHS=0
AUTO_CLEANUP_AFTER_ORGANIZE=1
USE_ARCHIVE_SERIAL_NUMBER=0
CORRESPONDENT_MATCH_THRESHOLD=0.90
NON_TAXONOMY_DELETE_THRESHOLD=5
DELETE_USED_TAGS=0
REVIEW_TAG_NAME=Manuell-Pruefen
AUTO_APPLY_REVIEW_TAG=1
REVIEW_ON_MEDIUM_CONFIDENCE=0
ENABLE_WEB_HINTS=1
WEB_HINT_TIMEOUT=6
WEB_HINT_MAX_ENTITIES=2
LLM_VERIFY_ON_LOW_CONFIDENCE=0
SKIP_RECENT_LLM_ERRORS_MINUTES=240
SKIP_RECENT_LLM_ERRORS_THRESHOLD=1
WATCH_RECONNECT_ERROR_THRESHOLD=3
WATCH_ERROR_BACKOFF_BASE_SEC=10
WATCH_ERROR_BACKOFF_MAX_SEC=180
```

## Empfehlung fuer stabilen Vollautomatik-Betrieb

- `ENFORCE_TAG_TAXONOMY=1` + `ALLOW_NEW_TAGS=0` (verhindert Tag-Explosion)
- `AUTO_CREATE_TAXONOMY_TAGS=1` erstellt fehlende Taxonomie-Tags automatisch (mit Farben aus `taxonomy_tags.json`)
- `KEEP_UNUSED_TAXONOMY_TAGS=1` verhindert, dass auto-erstellte Taxonomie-Tags mit `0 Dokumenten` direkt wieder geloescht werden
- `MAX_TOTAL_TAGS=100` verhindert unkontrolliertes Anwachsen des Tag-Bestands
- `RECHECK_ALL_DOCS_IN_AUTO=0` fuer schnellen Normalbetrieb; `1` nur fuer Voll-Revision
- `AUTO_CLEANUP_AFTER_ORGANIZE=1` (raemt nach Batch/Auto-Lauf direkt auf)
- `NON_TAXONOMY_DELETE_THRESHOLD=5` (raeumt alte/falsche Nicht-Taxonomie-Tags automatisch ab)
- `USE_ARCHIVE_SERIAL_NUMBER=0` (schneller und weniger API-Fehler bei ASN)
- bei harten Blockern wird automatisch `Manuell-Pruefen` gesetzt und ein Eintrag in `review_queue` angelegt
- `LLM_TIMEOUT/LLM_COMPACT_TIMEOUT` moderat halten; mit `LLM_COMPACT_TIMEOUT_RETRY` gibt es jetzt einen zweiten Kompakt-Versuch
- `SKIP_RECENT_LLM_ERRORS_*` verhindert, dass dieselben Timeout-Dokumente direkt wiederholt werden
- `ENABLE_LEARNING_PRIORS=1` nutzt bestaetigte Beispiele als Korrespondent-Priors (z. B. Scalable -> Kontoauszug/Finanzen)
- `LEARNING_PRIOR_ENABLE_TAG_SUGGESTION=0` verhindert riskante Tag-Uebernahme aus Priors (empfohlen)
- `LIVE_WATCH_COMPACT_FIRST=1` verhindert lange Blocker im Watcher (schneller Kompaktmodus zuerst)
- `WATCH_RECONNECT_ERROR_THRESHOLD` + `WATCH_ERROR_BACKOFF_*` sorgen fuer robustes Reconnect/Backoff bei Netzwerk- oder Server-Stoerungen
- `MAX_CONTEXT_EMPLOYER_HINTS` und `WORK_CORR_EMPLOYER_MIN_DOCS` begrenzen Prompt-Kontext und vermeiden Overfitting auf falsche Arbeitgeber
- Vollautomatik (`Menue 9`) kann dauerhaft laufen; Wartungstakt ueber `AUTOPILOT_*` steuern
- Web-Hinweise nur bei Bedarf aktivieren (`ENABLE_WEB_HINTS=1`), da sie Laufzeit kosten koennen
- Tag-Cleanup ist konservativ: standardmaessig nur ungenutzte Tags (`0 Dokumente`) loeschen; aggressiver nur mit `DELETE_USED_TAGS=1`
- SQLite ist rein lokal (keine Netzwerk-DB, keine offene Schnittstelle)

