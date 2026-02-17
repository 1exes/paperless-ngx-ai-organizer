# Paperless-NGX Organizer v2.0

Automatische Dokumentenorganisation fuer Paperless-NGX mit lokalem LLM (Ollama/LM Studio/OpenAI-kompatibel).

## Features

### Kernfunktionen
- **LLM-basierte Klassifikation**: Automatische Zuordnung von Titel, Korrespondent, Dokumenttyp, Speicherpfad und Tags
- **Ollama Structured JSON Output**: Erzwingt valides JSON via JSON-Schema (keine Parse-Fehler mehr)
- **Lernfaehiges System**: Few-Shot-Beispiele + Korrespondent-Priors aus bestaetigten Entscheidungen
- **Regelbasierter Fast-Path**: Bekannte Korrespondenten (10+ Beispiele, >80% konsistent) werden ohne LLM verarbeitet
- **Fallback-Kette**: Regelbasiert -> LLM full -> LLM compact -> LLM+Websuche -> Learning-Priors -> Review-Queue

### Intelligenz
- **Korrespondent-Merge**: Aehnliche Namen (z.B. "Baader Bank" + "Baader Bank AG") automatisch zusammenfuehren
- **Echte Websuche**: ddgs-Paket fuer echte Suchergebnisse bei unbekannten Entitaeten
- **IBAN-Bank-Erkennung**: Deutsche BLZ automatisch zu Banknamen aufloesen (50+ Banken)
- **Entity-Extraktion**: Email-Domains, URLs, IBANs -> Firmen identifizieren
- **Spracherkennung**: Automatische Deutsch/Englisch-Erkennung
- **Smarte Inhaltsvorschau**: Header + Footer fuer bessere Klassifikation (Briefkopf + Signatur)

### Qualitaet
- **OCR-Qualitaetspruefung**: Erkennt schlechte OCR und markiert betroffene Dokumente zur Review
- **Datumserkennung**: Extrahiert Dokumentdaten aus dem Inhalt (deutsche Formate)
- **Content-Fingerprinting**: Erkennt inhaltsaehnliche Dokumente (Near-Duplicates) via Trigram-Hashing
- **Duplikat-Erkennung**: Exakt + normalisierte Dateinamen + Titel + Inhaltsaehnlichkeit
- **Titel-Verbesserung**: Bereinigt LLM-generierte Titel (Endungen, Duplikate, OCR-Artefakte, Laenge)
- **Guardrails**: Arbeit/Privat-Trennung, Fahrzeug-Erkennung, Anbieter-Schutz

### Betrieb
- **Autopilot-Modus**: Vollautomatischer Dauerbetrieb mit konfigurierbaren Wartungszyklen
- **Rich TUI**: Interaktives Terminal-Menue mit Statistiken, Progress-Bars und Reports
- **Log-Rotation**: Automatische Rotation bei 5MB (3 Backups)
- **Graceful Shutdown**: Sauberes Herunterfahren bei SIGTERM
- **API-Rate-Limiting**: Schutz vor Server-Ueberlastung
- **Master-Data-Caching**: Tags/Korrespondenten mit TTL gecached (weniger API-Calls)
- **DB-Cleanup**: Automatische Bereinigung alter Runs (>90 Tage)
- **Config-Validierung**: Prueft Einstellungen beim Start
- **Learning-Backup**: Sicherung der Lerndaten per Menue

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
| `paperless_organizer.py` | Hauptanwendung (~6200 Zeilen) |
| `.env` | Konfiguration (nicht committen!) |
| `.env.example` | Konfigurationsvorlage mit Dokumentation |
| `taxonomy_tags.json` | Erlaubte Tags inkl. Synonyme und Farben |
| `organizer_state.db` | SQLite-DB mit Run-Historie (automatisch erstellt) |
| `learning_profile.json` | Lernprofil: Beschaeftigungsverlauf, Fahrzeuge, Hinweise |
| `learning_examples.jsonl` | Bestaetigte Few-Shot-Beispiele |
| `organizer.log` | Laufendes Text-Log (rotiert bei 5MB) |
| `backups/` | Learning-Daten Backups |
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
OCR-Qualitaetspruefung + Spracherkennung
    |
    v
Regelbasiert? (10+ Samples, >80% konsistent)
    |-- ja --> Suggestion ohne LLM
    |-- nein --v
               |
    LLM Full-Mode (mit Few-Shot + Learning-Hints + Web-Hints + IBAN-Lookup)
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
Titel-Verbesserung + Tag-Selektion + Taxonomie-Pruefung
    |
    v
Review-Pruefung (OCR-Qualitaet, fehlende Felder, Konflikte)
    |
    v
Aenderungen anwenden + Learning-Feedback
```

## Menue-Uebersicht

1. Alles sortieren (alle unorganisierten Dokumente, mit Progress-Bar)
2. Dokumente organisieren (erweiterte Optionen)
3. Aufraeumen (Tags, Korrespondenten, Dokumenttypen)
4. Duplikat-Erkennung (Dateiname + normalisiert + Titel + Content-Fingerprint)
5. Statistiken (Uebersicht + Organisationsgrad + Speicherpfad-Verteilung + Learning-Stats)
6. Review-Queue (offene Nachpruefungen + Auto-Resolve)
7. Einstellungen (Modus, LLM, Tags, Web-Hints, Backup, DB-Cleanup)
8. Live-Watch (neue Dokumente automatisch verarbeiten)
9. Vollautomatik / Autopilot (mit Status-Summary alle 10 Zyklen)
