# Paperless-NGX Organizer

Automatische Dokumentenorganisation fuer Paperless-NGX mit lokalem LLM (Ollama/LM Studio/OpenAI-kompatibel).

## Features

- **LLM-basierte Klassifikation**: Automatische Zuordnung von Titel, Korrespondent, Dokumenttyp, Speicherpfad und Tags
- **Ollama Structured JSON Output**: Erzwingt valides JSON via JSON-Schema (keine Parse-Fehler mehr)
- **Lernfaehiges System**: Few-Shot-Beispiele + Korrespondent-Priors aus bestaetigten Entscheidungen
- **Regelbasierter Fast-Path**: Bekannte Korrespondenten (10+ Beispiele, >80% konsistent) werden ohne LLM verarbeitet
- **Korrespondent-Merge**: Aehnliche Namen (z.B. "Baader Bank" + "Baader Bank AG") werden automatisch zusammengefuehrt
- **Echte Websuche**: ddgs-Paket fuer echte Suchergebnisse bei unbekannten Entitaeten (statt nur DuckDuckGo Instant API)
- **OCR-Qualitaetspruefung**: Erkennt schlechte OCR und markiert betroffene Dokumente zur Review
- **Datumserkennung**: Extrahiert Dokumentdaten aus dem Inhalt (deutsche Formate)
- **Content-Fingerprinting**: Erkennt inhaltsaehnliche Dokumente (Near-Duplicates) via Trigram-Hashing
- **Titel-Verbesserung**: Bereinigt LLM-generierte Titel (Dateiendungen, Formatierung, Laenge)
- **Fallback-Kette**: LLM full -> LLM compact -> LLM+Websuche -> Learning-Priors -> Review-Queue
- **Autopilot-Modus**: Vollautomatischer Dauerbetrieb mit konfigurierbaren Wartungszyklen
- **Rich TUI**: Interaktives Terminal-Menue mit Statistiken, Duplikat-Scan und Reports
- **Guardrails**: Arbeit/Privat-Trennung, Fahrzeug-Erkennung, Anbieter-Schutz

## Voraussetzungen

- Python 3.10+
- Paperless-NGX Instanz mit API-Token
- LLM-Server (Ollama empfohlen, z.B. `qwen2.5:14b`)

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# .env anpassen: PAPERLESS_URL, PAPERLESS_TOKEN, LLM_URL, LLM_MODEL
```

## Start

```bash
python paperless_organizer.py
```

## Wichtige Dateien

| Datei | Beschreibung |
|-------|-------------|
| `paperless_organizer.py` | Hauptanwendung (Menues, Organize, Cleanup, Duplikate, Statistiken) |
| `.env` | Konfiguration (nicht committen!) |
| `.env.example` | Konfigurationsvorlage mit Dokumentation |
| `taxonomy_tags.json` | Erlaubte Tags inkl. Synonyme und Farben |
| `organizer_state.db` | SQLite-DB mit Run-Historie (automatisch erstellt) |
| `learning_profile.json` | Lernprofil: Beschaeftigungsverlauf, Fahrzeuge, Hinweise |
| `learning_examples.jsonl` | Bestaetigte Few-Shot-Beispiele |
| `organizer.log` | Laufendes Text-Log |
| `legacy/` | Alte Einzelskripte (nur als Referenz) |
| `test_dokumente/` | Test-PDFs fuer Entwicklung |

## Empfohlene Konfiguration

```env
# Modell: qwen2.5:14b bietet gutes JSON-Verstaendnis bei akzeptabler Geschwindigkeit
LLM_MODEL=qwen2.5:14b
LLM_TIMEOUT=120
LLM_COMPACT_TIMEOUT=50

# Aggressives Lernen: schon ab 2 Beispielen Priors nutzen
LEARNING_PRIOR_MIN_SAMPLES=2
LEARNING_PRIOR_ENABLE_TAG_SUGGESTION=1
LEARNING_EXAMPLE_LIMIT=5

# Websuche fuer unbekannte Firmen (benoetigt: pip install ddgs)
ENABLE_WEB_HINTS=1

# Taxonomie erzwingen (verhindert Tag-Explosion)
ENFORCE_TAG_TAXONOMY=1
AUTO_CREATE_TAXONOMY_TAGS=1
```

## Verarbeitungskette

```
Dokument geladen
    |
    v
OCR-Qualitaetspruefung
    |
    v
Regelbasiert? (10+ Samples, >80% konsistent)
    |-- ja --> Suggestion ohne LLM
    |-- nein --v
               |
    LLM Full-Mode (mit Few-Shot + Learning-Hints + Web-Hints)
        |-- Timeout/Fehler --> LLM Compact-Mode
        |-- Fehler --> LLM + Websuche-Kontext
        |-- Fehler --> Learning-Prior Fallback
        |-- Fehler --> Review-Queue
    |
    v
Low-Confidence? + Kein Korrespondent? --> Websuche-Verifikation
    |
    v
Guardrails (Vendor, Vehicle, Topic, Learning)
    |
    v
Tag-Selektion + Taxonomie-Pruefung
    |
    v
Review-Pruefung (OCR-Qualitaet, fehlende Felder, Konflikte)
    |
    v
Aenderungen anwenden
```

## Menue-Uebersicht

1. Alles sortieren (alle unorganisierten Dokumente)
2. Einzelnes Dokument sortieren
3. Cleanup (Tags, Korrespondenten, Dokumenttypen)
4. Duplikat-Erkennung (Dateiname + Titel + Content-Fingerprint)
5. Statistiken / Uebersicht
6. Monatsreport
7. Erweiterte Einstellungen
8. Live-Watch (neue Dokumente automatisch verarbeiten)
9. Vollautomatik / Autopilot
