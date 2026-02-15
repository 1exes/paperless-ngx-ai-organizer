# Automation Plan

## Ziel

Ein stabiler Vollautomatik-Workflow ohne Tag-Wildwuchs.

## Stand (bereits umgesetzt)

1. Einheitlicher Hauptworkflow in `paperless_organizer.py`
2. Lokale SQLite-Historie (`organizer_state.db`) fuer Runs und Dokumententscheidungen
3. Strenge Tag-Policy (standardmaessig keine neuen Tags)
4. Optionales Auto-Cleanup nach Batch/Auto-Lauf
5. Feste Tag-Taxonomie mit Synonymen (`taxonomy_tags.json`)
6. Review-Queue (`review_queue`) + `Manuell-Pruefen` Tag bei harten Blockern

## Naechste Schritte

1. Monatlicher Report aus SQLite erzeugen (z. B. neue Korrespondenten, verworfene Tags, offene Reviews)
