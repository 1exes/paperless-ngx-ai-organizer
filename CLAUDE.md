# Paperless-NGX AI Organizer - Projektregeln

## Sprache
- Kommunikation mit dem User: **Deutsch**
- Code-Kommentare und Variablen: Englisch
- Git-Commits: Englisch

## Projekt-Architektur
- **Hauptdatei**: `paperless_organizer.py` (7250 Zeilen, unified Terminal App v2.2.1)
- **Python**: 3.13 (`C:\Users\edgar\AppData\Local\Programs\Python\Python313\python.exe`)
- **LLM**: Lokal via Ollama/LM Studio (OpenAI-kompatibel) @ konfigurierbar in `.env`
- **Paperless-NGX**: REST API mit Token-Auth
- **State-DB**: SQLite lokal (`organizer_state.db`)
- **Config**: `.env` Datei (kopiert von `.env.example`)
- **Tag-Taxonomie**: `taxonomy_tags.json` (38 feste Kategorien)
- **UI**: Rich Terminal (interaktives Menuesystem)
- **Legacy-Scripts**: `legacy/` Ordner (alte Einzelskripte, nicht mehr aktiv)

## Wichtige Klassen (in paperless_organizer.py)
- `LocalStateDB` - SQLite State-Tracking (Runs, Dokumente, Reviews)
- `PaperlessClient` - REST API Client mit Caching + Rate-Limiting
- `LocalLLMAnalyzer` - LLM-Wrapper (Ollama/LM Studio/OpenAI)
- `LearningExamples` - Few-Shot JSONL-Speicher (max 500 Beispiele)
- `LearningProfile` - Arbeitgeber/Fahrzeug/Kontext-Profil
- `TagTaxonomy` - Tag-Verwaltung aus taxonomy_tags.json
- `DecisionContext` - Runtime-Kontext fuer Dokumentverarbeitung
- `App` - Hauptklasse mit Menuesystem und allen Actions

## Code-Regeln
- Alles in EINER Datei (`paperless_organizer.py`) - so beibehalten
- Deutsche Terminal-Ausgaben (Rich Console)
- Keine unnötigen Kommentare oder Docstrings hinzufügen
- Bestehende Patterns und Konventionen beibehalten
- Fehler-Logging: `log.warning()` fuer wichtige Fehler, `log.debug()` fuer Details
- Thread-Safety beachten (SQLite + Lock-Pattern)

## Sicherheit
- Keine API-Tokens oder Passwörter in Code commiten
- Secrets gehören in `.env` (ist in .gitignore)
- Paperless-Token und LLM-URL nur in `.env`

## Testing
- Test-PDFs in `test_dokumente/` (16 Beispiel-Dokumente)
- `generate_test_invoices.py` zum Erstellen neuer Test-PDFs
- Dry-Run Modus: `DEFAULT_DRY_RUN=1` in `.env`

## Starten
```bash
python paperless_organizer.py
```
