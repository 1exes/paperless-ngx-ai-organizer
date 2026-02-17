# Paperless-NGX Organizer v2.2

Automatische Dokumentenorganisation fuer Paperless-NGX mit lokalem LLM (Ollama/LM Studio/OpenAI-kompatibel).

## Features

### Kernfunktionen
- **LLM-basierte Klassifikation**: Automatische Zuordnung von Titel, Korrespondent, Dokumenttyp, Speicherpfad und Tags
- **Ollama Structured JSON Output**: Erzwingt valides JSON via JSON-Schema mit Enum-Constraints (keine Halluzinationen bei Dokumenttypen/Pfaden)
- **Lernfaehiges System**: Few-Shot-Beispiele + Korrespondent-Priors + Negative Learning (Anti-Patterns) + Similarity-basierte Auswahl
- **Regelbasierter Fast-Path**: Bekannte Korrespondenten (10+ Beispiele, >80% konsistent) werden ohne LLM verarbeitet
- **Fallback-Kette**: Regelbasiert -> LLM full -> LLM compact -> LLM+Websuche -> Learning-Priors -> Review-Queue
- **Automatischer Modell-Fallback**: Wechsel auf kleineres Modell nach wiederholten Fehlern

### Intelligenz
- **Korrespondent-Merge**: Aehnliche Namen (z.B. "Baader Bank" + "Baader Bank AG") automatisch zusammenfuehren
- **Echte Websuche**: ddgs-Paket fuer echte Suchergebnisse bei unbekannten Entitaeten
- **IBAN-Bank-Erkennung**: Deutsche BLZ automatisch zu Banknamen aufloesen (50+ Banken)
- **Entity-Extraktion**: Email-Domains, URLs, IBANs -> Firmen identifizieren
- **Spracherkennung**: Automatische Deutsch/Englisch-Erkennung mit mehrsprachigen Prompts
- **Smarte Inhaltsvorschau**: Header + Footer fuer bessere Klassifikation (Briefkopf + Signatur)
- **Inhaltsmuster-Erkennung**: IBAN, Rechnungsnummer, Vertrag, Steuer, Versicherung, Gehalt automatisch erkannt
- **Titel-Anreicherung**: Rechnungsnummern und Betraege automatisch im Titel ergaenzt
- **Learning Decay**: Aeltere Beispiele werden weniger stark gewichtet (neuere zaehlen mehr)
- **Similarity-basierte Few-Shot-Auswahl**: Content-Fingerprint-Aehnlichkeit fuer bessere Beispielwahl

### Qualitaet
- **OCR-Qualitaetspruefung**: Erkennt schlechte OCR und markiert betroffene Dokumente zur Review
- **Datumserkennung**: Extrahiert Dokumentdaten aus dem Inhalt (deutsche Formate)
- **Content-Fingerprinting**: Near-Duplicate-Erkennung via Trigram-Hashing + MinHash/LSH (skaliert bis 20.000 Dokumente)
- **Checksum-Duplikate**: SHA256-basierte exakte Inhalts-Duplikat-Erkennung
- **Duplikat-Erkennung**: Exakt + normalisierte Dateinamen + Titel + Checksum + Inhaltsaehnlichkeit
- **Titel-Verbesserung**: Bereinigt LLM-generierte Titel (Endungen, Duplikate, OCR-Artefakte, Laenge)
- **Guardrails**: Arbeit/Privat-Trennung, Fahrzeug-Erkennung, Anbieter-Schutz
- **Negative Learning**: Falsche Vorschlaege werden als Anti-Patterns gespeichert und kuenftig vermieden
- **Konfidenz-Kalibrierung**: Vergleicht LLM-Konfidenz mit tatsaechlicher Genauigkeit

### Betrieb
- **Autopilot-Modus**: Vollautomatischer Dauerbetrieb mit konfigurierbaren Wartungszyklen + Ruhestunden + Smart Scheduling
- **Rich TUI**: Interaktives Terminal-Menue mit Statistiken, Progress-Bars, ETA-Anzeige und Reports
- **Log-Rotation**: Automatische Rotation bei 5MB (3 Backups)
- **Graceful Shutdown**: Sauberes Herunterfahren bei SIGTERM
- **API-Rate-Limiting**: Schutz vor Server-Ueberlastung mit Exponential Backoff
- **Master-Data-Caching**: Tags/Korrespondenten mit TTL gecached (weniger API-Calls)
- **Prompt-Caching**: Statische Prompt-Teile werden gecached (weniger Rechenaufwand)
- **DB-Cleanup**: Automatische Bereinigung alter Runs (>90 Tage)
- **Config-Validierung**: Prueft Einstellungen beim Start
- **Health-Check**: Systemstatus beim Start (DB-Groesse, Learning-Daten, letzter Run)
- **Learning-Backup**: Sicherung der Lerndaten per Menue
- **Learning-Integritaetspruefung**: Validiert Learning-Daten auf Fehler und Duplikate
- **CSV-Export**: Verarbeitungshistorie als CSV exportierbar
- **LLM-Performance-Tracking**: Antwortzeiten und Erfolgsrate automatisch erfasst
- **Fehler-Kategorisierung**: Timeout, JSON, Connection, API aufgeschluesselt

### Statistiken
- **Organisationsgrad**: Vollstaendigkeit aller Dokumente
- **Tag-Kombinationen**: Top-10 haeufigste Tag-Paare
- **Dokumenttyp-Verteilung**: Anteil jedes Typs mit Prozenten
- **Speicherpfad-Analyse**: Nutzung und leere Pfade auf einen Blick
- **Korrespondenten-Aktivitaet**: Aktiv (30/90 Tage) vs. dormant
- **Stale-Erkennung**: Dokumente >30 Tage unorganisiert
- **Verarbeitungsstatistik**: Erfolgsrate, Fehler, Review-Quote (letzte 30 Tage)
- **Konfidenz-Kalibrierung**: Genauigkeit pro Konfidenz-Level

### Review-Queue
- **Batch-Verarbeitung**: Alle offenen Reviews auf einmal pruefen + lernen
- **Alters-Tracking**: Farbcodierung nach Alter (gruen <7d, gelb 7-30d, rot >30d)
- **Dokument-Vorschau**: Inhalt und LLM-Vorschlag direkt in der Review-Queue ansehen
- **Auto-Resolve**: Periodisches Pruefen ob Reviews inzwischen manuell erledigt wurden

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
| `paperless_organizer.py` | Hauptanwendung (~7100 Zeilen) |
| `.env` | Konfiguration (nicht committen!) |
| `.env.example` | Konfigurationsvorlage mit Dokumentation |
| `taxonomy_tags.json` | Erlaubte Tags inkl. Synonyme und Farben |
| `organizer_state.db` | SQLite-DB mit Run-Historie + Konfidenz-Kalibrierung (automatisch erstellt) |
| `learning_profile.json` | Lernprofil: Beschaeftigungsverlauf, Fahrzeuge, Hinweise |
| `learning_examples.jsonl` | Bestaetigte Few-Shot-Beispiele (inkl. Anti-Patterns) |
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

# Fallback-Modell bei wiederholten Fehlern
LLM_FALLBACK_MODEL=qwen2.5:7b
LLM_FALLBACK_AFTER_ERRORS=3

# Aggressives Lernen: schon ab 2 Beispielen Priors nutzen
LEARNING_PRIOR_MIN_SAMPLES=2
LEARNING_PRIOR_ENABLE_TAG_SUGGESTION=1
LEARNING_EXAMPLE_LIMIT=5

# Websuche fuer unbekannte Firmen (benoetigt: pip install ddgs)
ENABLE_WEB_HINTS=1

# Taxonomie erzwingen (verhindert Tag-Explosion)
ENFORCE_TAG_TAXONOMY=1
AUTO_CREATE_TAXONOMY_TAGS=1

# Ruhestunden (optional, z.B. 23:00-06:00)
# QUIET_HOURS_START=23
# QUIET_HOURS_END=6
```

## Verarbeitungskette

```
Dokument geladen
    |
    v
OCR-Qualitaetspruefung + Spracherkennung + Inhaltsmuster-Erkennung
    |
    v
Regelbasiert? (10+ Samples, >80% konsistent)
    |-- ja --> Suggestion ohne LLM
    |-- nein --v
               |
    LLM Full-Mode (mit Enum-Constraints + Few-Shot + Learning-Hints + Web-Hints + IBAN-Lookup)
        |-- Timeout/Fehler --> LLM Compact-Mode
        |-- Fehler --> LLM + Websuche-Kontext
        |-- Fehler --> Modell-Fallback (wenn konfiguriert)
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
Titel-Verbesserung (+ Rechnungsnr/Betrag) + Tag-Selektion + Taxonomie-Pruefung
    |
    v
Review-Pruefung (OCR-Qualitaet, fehlende Felder, Konflikte)
    |
    v
Aenderungen anwenden + Learning-Feedback (positiv + negativ) + Konfidenz-Tracking
```

## Menue-Uebersicht

1. Alles sortieren (alle unorganisierten Dokumente, mit Progress-Bar + ETA, neueste zuerst)
2. Dokumente organisieren (erweiterte Optionen)
3. Aufraeumen (Tags, Korrespondenten, Dokumenttypen)
4. Duplikat-Erkennung (Checksum + Dateiname + normalisiert + Titel + MinHash/LSH-Fingerprint)
5. Statistiken (Uebersicht + Tag-Kombinationen + Dokumenttyp-Verteilung + Speicherpfade + Korrespondenten-Aktivitaet + Konfidenz-Kalibrierung)
6. Review-Queue (offene Nachpruefungen + Batch-Verarbeitung + Dokument-Vorschau + Auto-Resolve)
7. Einstellungen (Modus, LLM, Tags, Web-Hints, Backup, DB-Cleanup, CSV-Export, Learning-Integritaet)
8. Live-Watch (neue Dokumente automatisch verarbeiten)
9. Vollautomatik / Autopilot (mit Smart Scheduling, Status-Summary alle 10 Zyklen, Ruhestunden)
