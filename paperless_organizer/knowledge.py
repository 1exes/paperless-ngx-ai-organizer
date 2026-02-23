"""Personal Knowledge Database for fact extraction and context enrichment.

Stores entities (persons, vehicles, companies, addresses, insurances, memberships),
facts (time-bounded knowledge assertions), entity relations, and document sources.
Uses PostgreSQL as backend via psycopg2.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import date, datetime
from typing import Any

from .config import log

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _fuzzy_ratio(a: str, b: str) -> float:
    """Simple character-level similarity ratio (0.0-1.0)."""
    if not a or not b:
        return 0.0
    import difflib
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    attributes JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_type_name
    ON entities (entity_type, name_normalized);

CREATE TABLE IF NOT EXISTS facts (
    id SERIAL PRIMARY KEY,
    fact_type TEXT NOT NULL,
    subject_entity_id INTEGER REFERENCES entities(id),
    summary TEXT NOT NULL,
    detail JSONB DEFAULT '{}',
    valid_from DATE,
    valid_until DATE,
    confidence REAL DEFAULT 0.7,
    is_current BOOLEAN DEFAULT TRUE,
    superseded_by_id INTEGER REFERENCES facts(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_facts_type ON facts (fact_type);
CREATE INDEX IF NOT EXISTS idx_facts_current ON facts (is_current) WHERE is_current = TRUE;

CREATE TABLE IF NOT EXISTS fact_sources (
    id SERIAL PRIMARY KEY,
    fact_id INTEGER NOT NULL REFERENCES facts(id),
    doc_id INTEGER NOT NULL,
    doc_title TEXT,
    doc_date DATE,
    extraction_method TEXT DEFAULT 'llm',
    extracted_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fact_sources_doc ON fact_sources (doc_id);

CREATE TABLE IF NOT EXISTS entity_relations (
    id SERIAL PRIMARY KEY,
    from_entity_id INTEGER NOT NULL REFERENCES entities(id),
    relation_type TEXT NOT NULL,
    to_entity_id INTEGER NOT NULL REFERENCES entities(id),
    valid_from DATE,
    valid_until DATE,
    is_current BOOLEAN DEFAULT TRUE,
    detail JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

# ---------------------------------------------------------------------------
# Valid fact types
# ---------------------------------------------------------------------------

VALID_FACT_TYPES = {
    # Arbeit
    "employment_start", "employment_end", "employment_active",
    # Ausbildung / Weiterbildung
    "education_start", "education_end", "education_active", "education_exam",
    # Fahrzeuge
    "vehicle_acquired", "vehicle_disposed", "vehicle_accident", "vehicle_service",
    # Adressen / Umzug
    "address_move", "address_current",
    # Versicherungen
    "insurance_start", "insurance_end", "insurance_change", "insurance_claim",
    # Mitgliedschaften
    "membership_start", "membership_end", "membership_active",
    # Gesundheit
    "health_provider", "health_event",
    # Finanzen
    "bank_account", "financial_contract",
    # Regelmaessige Dienste / Abos
    "regular_service",
    # Lebensereignisse
    "life_event",
    # Sonstiges
    "note",
}

VALID_ENTITY_TYPES = {
    "person", "vehicle", "company", "address", "insurance",
    "membership", "education", "authority",
}

VALID_RELATION_TYPES = {
    "belongs_to", "insures", "covers", "employs", "member_of",
    "trains_at", "supervises", "located_in",
}


# ---------------------------------------------------------------------------
# KnowledgeDB
# ---------------------------------------------------------------------------

class KnowledgeDB:
    """PostgreSQL-backed personal knowledge database."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._conn = None
        self._ensure_schema()

    # -- connection ----------------------------------------------------------

    def _get_conn(self):
        """Get or create a psycopg2 connection."""
        import psycopg2
        import psycopg2.extras
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
            self._conn.autocommit = False
            psycopg2.extras.register_default_jsonb(self._conn)
        return self._conn

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Split by statement since psycopg2 doesn't do multi-statement well
                for stmt in _SCHEMA_SQL.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        cur.execute(stmt)
            conn.commit()
            log.debug("KnowledgeDB: Schema sichergestellt")
        except Exception:
            conn.rollback()
            raise

    def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    # -- Entity CRUD ---------------------------------------------------------

    def get_or_create_entity(self, entity_type: str, name: str,
                             attributes: dict | None = None) -> int:
        """Find or create an entity. Returns entity ID.

        Deduplicates via name_normalized. If entity exists, merges attributes.
        """
        name_norm = _normalize_name(name)
        attrs = attributes or {}
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, attributes FROM entities "
                    "WHERE entity_type = %s AND name_normalized = %s",
                    (entity_type, name_norm),
                )
                row = cur.fetchone()
                if row:
                    entity_id, existing_attrs = row
                    if attrs:
                        merged = {**(existing_attrs or {}), **attrs}
                        cur.execute(
                            "UPDATE entities SET attributes = %s, updated_at = NOW() WHERE id = %s",
                            (json.dumps(merged), entity_id),
                        )
                    conn.commit()
                    return entity_id
                else:
                    cur.execute(
                        "INSERT INTO entities (entity_type, name, name_normalized, attributes) "
                        "VALUES (%s, %s, %s, %s) RETURNING id",
                        (entity_type, name.strip(), name_norm, json.dumps(attrs)),
                    )
                    entity_id = cur.fetchone()[0]
                    conn.commit()
                    return entity_id
        except Exception:
            conn.rollback()
            raise

    def find_entity(self, entity_type: str, name: str) -> int | None:
        """Find entity by exact normalized name. Returns ID or None."""
        name_norm = _normalize_name(name)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM entities WHERE entity_type = %s AND name_normalized = %s",
                (entity_type, name_norm),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def find_entity_fuzzy(self, entity_type: str, name: str,
                          threshold: float = 0.85) -> int | None:
        """Find entity by fuzzy name match. Returns best match ID or None."""
        name_norm = _normalize_name(name)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name_normalized FROM entities WHERE entity_type = %s",
                (entity_type,),
            )
            best_id, best_ratio = None, 0.0
            for row in cur.fetchall():
                ratio = _fuzzy_ratio(name_norm, row[1])
                if ratio > best_ratio and ratio >= threshold:
                    best_id, best_ratio = row[0], ratio
            return best_id

    def get_all_entities(self, entity_type: str | None = None) -> list[dict]:
        """Get all entities, optionally filtered by type."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            if entity_type:
                cur.execute(
                    "SELECT id, entity_type, name, attributes, created_at FROM entities "
                    "WHERE entity_type = %s ORDER BY name",
                    (entity_type,),
                )
            else:
                cur.execute(
                    "SELECT id, entity_type, name, attributes, created_at FROM entities "
                    "ORDER BY entity_type, name"
                )
            return [
                {"id": r[0], "entity_type": r[1], "name": r[2],
                 "attributes": r[3] or {}, "created_at": r[4]}
                for r in cur.fetchall()
            ]

    # -- Fact CRUD -----------------------------------------------------------

    def store_fact(self, fact_type: str, summary: str,
                   detail: dict | None = None,
                   entity_id: int | None = None,
                   valid_from: date | str | None = None,
                   valid_until: date | str | None = None,
                   confidence: float = 0.7,
                   doc_id: int | None = None,
                   doc_title: str | None = None,
                   doc_date: date | str | None = None,
                   extraction_method: str = "llm") -> int | None:
        """Store a fact with duplicate check (fuzzy 85% on summary).

        Returns fact ID, or None if duplicate was found.
        """
        from . import config as _cfg

        min_confidence = getattr(_cfg, "KNOWLEDGE_MIN_CONFIDENCE", 0.4)
        if confidence < min_confidence:
            log.debug("KnowledgeDB: Fakt verworfen (Konfidenz %.2f < %.2f): %s",
                       confidence, min_confidence, summary[:60])
            return None

        detail = detail or {}
        valid_from_d = _parse_date(valid_from)
        valid_until_d = _parse_date(valid_until)
        doc_date_d = _parse_date(doc_date)

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Duplicate check: same fact_type + same entity + fuzzy summary
                cur.execute(
                    "SELECT id, summary FROM facts "
                    "WHERE fact_type = %s AND subject_entity_id IS NOT DISTINCT FROM %s "
                    "AND is_current = TRUE",
                    (fact_type, entity_id),
                )
                for existing_id, existing_summary in cur.fetchall():
                    if _fuzzy_ratio(summary, existing_summary) >= 0.85:
                        log.debug("KnowledgeDB: Duplikat-Fakt erkannt (id=%s): %s",
                                   existing_id, summary[:60])
                        # Still link the document as additional source
                        if doc_id:
                            cur.execute(
                                "INSERT INTO fact_sources (fact_id, doc_id, doc_title, doc_date, extraction_method) "
                                "VALUES (%s, %s, %s, %s, %s)",
                                (existing_id, doc_id, doc_title, doc_date_d, extraction_method),
                            )
                        conn.commit()
                        return existing_id

                # Insert new fact
                cur.execute(
                    "INSERT INTO facts (fact_type, subject_entity_id, summary, detail, "
                    "valid_from, valid_until, confidence, is_current) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE) RETURNING id",
                    (fact_type, entity_id, summary, json.dumps(detail),
                     valid_from_d, valid_until_d, confidence),
                )
                fact_id = cur.fetchone()[0]

                if doc_id:
                    cur.execute(
                        "INSERT INTO fact_sources (fact_id, doc_id, doc_title, doc_date, extraction_method) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (fact_id, doc_id, doc_title, doc_date_d, extraction_method),
                    )
                conn.commit()
                log.debug("KnowledgeDB: Fakt #%s gespeichert: %s", fact_id, summary[:60])
                return fact_id
        except Exception:
            conn.rollback()
            raise

    def supersede_fact(self, old_id: int, new_id: int):
        """Mark old fact as superseded by new one."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE facts SET is_current = FALSE, superseded_by_id = %s, "
                    "updated_at = NOW() WHERE id = %s",
                    (new_id, old_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def get_current_facts(self, fact_type: str | None = None,
                          entity_id: int | None = None) -> list[dict]:
        """Get active (is_current=TRUE) facts, optionally filtered."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            conditions = ["f.is_current = TRUE"]
            params: list = []
            if fact_type:
                conditions.append("f.fact_type = %s")
                params.append(fact_type)
            if entity_id:
                conditions.append("f.subject_entity_id = %s")
                params.append(entity_id)
            where = " AND ".join(conditions)
            cur.execute(
                f"SELECT f.id, f.fact_type, f.subject_entity_id, f.summary, f.detail, "
                f"f.valid_from, f.valid_until, f.confidence, f.created_at, "
                f"e.name AS entity_name, e.entity_type "
                f"FROM facts f "
                f"LEFT JOIN entities e ON f.subject_entity_id = e.id "
                f"WHERE {where} "
                f"ORDER BY f.valid_from DESC NULLS LAST, f.created_at DESC",
                params,
            )
            return [
                {"id": r[0], "fact_type": r[1], "entity_id": r[2], "summary": r[3],
                 "detail": r[4] or {}, "valid_from": r[5], "valid_until": r[6],
                 "confidence": r[7], "created_at": r[8],
                 "entity_name": r[9], "entity_type": r[10]}
                for r in cur.fetchall()
            ]

    def get_facts_at_date(self, ref_date: date) -> list[dict]:
        """Get facts that were valid at a specific date."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT f.id, f.fact_type, f.subject_entity_id, f.summary, f.detail, "
                "f.valid_from, f.valid_until, f.confidence, "
                "e.name AS entity_name, e.entity_type "
                "FROM facts f "
                "LEFT JOIN entities e ON f.subject_entity_id = e.id "
                "WHERE (f.valid_from IS NULL OR f.valid_from <= %s) "
                "AND (f.valid_until IS NULL OR f.valid_until >= %s) "
                "AND f.is_current = TRUE "
                "ORDER BY f.valid_from DESC NULLS LAST",
                (ref_date, ref_date),
            )
            return [
                {"id": r[0], "fact_type": r[1], "entity_id": r[2], "summary": r[3],
                 "detail": r[4] or {}, "valid_from": r[5], "valid_until": r[6],
                 "confidence": r[7], "entity_name": r[8], "entity_type": r[9]}
                for r in cur.fetchall()
            ]

    def deactivate_fact(self, fact_id: int):
        """Set is_current=FALSE on a fact."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE facts SET is_current = FALSE, updated_at = NOW() WHERE id = %s",
                    (fact_id,),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # -- Relations -----------------------------------------------------------

    def store_relation(self, from_entity_id: int, relation_type: str,
                       to_entity_id: int, valid_from: date | str | None = None,
                       valid_until: date | str | None = None,
                       detail: dict | None = None) -> int:
        """Store an entity relation. Returns relation ID."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Check for existing identical relation
                cur.execute(
                    "SELECT id FROM entity_relations "
                    "WHERE from_entity_id = %s AND relation_type = %s AND to_entity_id = %s "
                    "AND is_current = TRUE",
                    (from_entity_id, relation_type, to_entity_id),
                )
                row = cur.fetchone()
                if row:
                    conn.commit()
                    return row[0]
                cur.execute(
                    "INSERT INTO entity_relations "
                    "(from_entity_id, relation_type, to_entity_id, valid_from, valid_until, detail) "
                    "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                    (from_entity_id, relation_type, to_entity_id,
                     _parse_date(valid_from), _parse_date(valid_until),
                     json.dumps(detail or {})),
                )
                rel_id = cur.fetchone()[0]
                conn.commit()
                return rel_id
        except Exception:
            conn.rollback()
            raise

    def get_relations(self, entity_id: int | None = None) -> list[dict]:
        """Get relations, optionally for a specific entity (as from or to)."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            if entity_id:
                cur.execute(
                    "SELECT r.id, r.from_entity_id, r.relation_type, r.to_entity_id, "
                    "r.valid_from, r.valid_until, r.is_current, r.detail, "
                    "e1.name AS from_name, e2.name AS to_name "
                    "FROM entity_relations r "
                    "JOIN entities e1 ON r.from_entity_id = e1.id "
                    "JOIN entities e2 ON r.to_entity_id = e2.id "
                    "WHERE (r.from_entity_id = %s OR r.to_entity_id = %s) "
                    "AND r.is_current = TRUE "
                    "ORDER BY r.created_at DESC",
                    (entity_id, entity_id),
                )
            else:
                cur.execute(
                    "SELECT r.id, r.from_entity_id, r.relation_type, r.to_entity_id, "
                    "r.valid_from, r.valid_until, r.is_current, r.detail, "
                    "e1.name AS from_name, e2.name AS to_name "
                    "FROM entity_relations r "
                    "JOIN entities e1 ON r.from_entity_id = e1.id "
                    "JOIN entities e2 ON r.to_entity_id = e2.id "
                    "WHERE r.is_current = TRUE "
                    "ORDER BY r.created_at DESC"
                )
            return [
                {"id": r[0], "from_entity_id": r[1], "relation_type": r[2],
                 "to_entity_id": r[3], "valid_from": r[4], "valid_until": r[5],
                 "is_current": r[6], "detail": r[7] or {},
                 "from_name": r[8], "to_name": r[9]}
                for r in cur.fetchall()
            ]

    # -- LLM Fact Extraction -------------------------------------------------

    def extract_and_store_facts(self, document: dict, suggestion: dict,
                                analyzer, owner: str):
        """Extract facts from a document via LLM and store them.

        This is a secondary LLM call after classification, kept compact and fast.
        On timeout or error, it's silently skipped (classification is unaffected).
        """
        from . import config as _cfg

        if not getattr(_cfg, "ENABLE_KNOWLEDGE_EXTRACTION", True):
            return

        timeout = getattr(_cfg, "KNOWLEDGE_EXTRACTION_TIMEOUT", 30)
        max_tokens = getattr(_cfg, "KNOWLEDGE_EXTRACTION_MAX_TOKENS", 400)

        doc_id = document.get("id", 0)
        title = suggestion.get("title") or document.get("title", "")
        correspondent = suggestion.get("correspondent", "")
        doc_type = suggestion.get("document_type", "")
        content = document.get("content") or ""

        # Compact content preview (max 1500 chars)
        if len(content) > 1500:
            content_preview = content[:700] + "\n[...]\n" + content[-500:]
        else:
            content_preview = content

        # Build known facts summary for context
        known = self.build_prompt_context(owner, max_len=300)

        prompt = (
            f"Analysiere Dokument #{doc_id} und extrahiere ALLE persoenlichen Fakten ueber {owner}.\n"
            f"Titel: {title} | Korrespondent: {correspondent} | Typ: {doc_type}\n"
            f"INHALT:\n{content_preview}\n\n"
            f"BEREITS BEKANNT:\n{known}\n\n"
            "EXTRAHIERE (falls im Dokument erkennbar):\n"
            "- ADRESSE/UMZUG: Neue Wohnadresse, Stadt, PLZ -> fact_type 'address_current' oder 'address_move'\n"
            "  detail: {strasse, plz, stadt} - WICHTIG fuer lokale Zuordnung!\n"
            "- ARBEITGEBER: Neuer Job, Kuendigung, Arbeitsvertrag -> 'employment_start/end/active'\n"
            "- AUSBILDUNG: Ausbildungsvertrag, IHK-Pruefung, Berufsschule, Umschulung,\n"
            "  Weiterbildung, Zertifikat -> 'education_start/end/active/exam'\n"
            "  detail: {typ: 'Ausbildung'/'Umschulung'/'Weiterbildung', fach, institution}\n"
            "  WICHTIG: IHK, Berufsschule, Pruefungen -> gehoeren zur Ausbildung!\n"
            "- FAHRZEUG: Kauf, Verkauf, Unfall, Werkstatt -> 'vehicle_acquired/disposed/accident/service'\n"
            "  detail: {marke, modell, kennzeichen, kategorie: 'privat'/'firma'}\n"
            "- VERSICHERUNG: Neue Police, Kuendigung, Schadenfall -> 'insurance_start/end/claim'\n"
            "- GESUNDHEIT: Arzt, Krankenhaus, Krankenkasse -> 'health_provider/health_event'\n"
            "- MITGLIEDSCHAFT: Verein, Fitnessstudio, Organisation -> 'membership_start/end/active'\n"
            "- FINANZEN: Bankkonto, Vertrag, Abo, Kredit -> 'bank_account/financial_contract'\n"
            "- ABO/DIENST: Regelmaessige Kosten (Strom, Internet, Handy) -> 'regular_service'\n"
            "  detail: {anbieter, betrag, intervall: 'monatlich'/'jaehrlich'}\n"
            "- LEBENSEREIGNIS: Heirat, Scheidung, Geburt, Todesfall -> 'life_event'\n"
            "- SONSTIGES: Alles andere Relevante -> 'note'\n\n"
            "REGELN:\n"
            "- NUR Fakten die DIREKT aus dem Dokument hervorgehen\n"
            "- Bei Adressen: IMMER Stadt und PLZ extrahieren (wichtig fuer lokale Zuordnung)\n"
            "- Bei Ausbildung: Auch VERBUNDENE Institutionen erfassen (IHK bei Ausbildung, Berufsschule)\n"
            "- Beziehungen benennen: z.B. IHK 'gehoert zur Ausbildung bei Firma X'\n"
            "- Nichts wiederholen was schon bekannt ist\n"
            "- confidence: 0.9 bei klaren Fakten, 0.6-0.8 bei Vermutungen\n\n"
            "Antwort als JSON-Array:\n"
            '[{"fact_type": "...", "entity_type": "...", "entity_name": "...", '
            '"summary": "Kurze deutsche Beschreibung", "valid_from": "YYYY-MM-DD", '
            '"valid_until": null, "confidence": 0.8, '
            '"detail": {"stadt": "...", "typ": "...", "related_to": "..."}}]\n\n'
            f"Erlaubte fact_types: {', '.join(sorted(VALID_FACT_TYPES))}\n"
            f"Erlaubte entity_types: {', '.join(sorted(VALID_ENTITY_TYPES))}\n"
            "Wenn KEINE neuen Fakten: antworte mit []\n"
            "NUR JSON-Array, kein anderer Text."
        )

        try:
            t0 = time.perf_counter()
            response = analyzer._call_llm(
                prompt,
                read_timeout=timeout,
                retries=0,
                max_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - t0
            log.debug("KnowledgeDB: Extraktion fuer #%s in %.1fs", doc_id, elapsed)
        except Exception as exc:
            log.debug("KnowledgeDB: Extraktion uebersprungen fuer #%s: %s", doc_id, exc)
            return

        # Parse the LLM response
        facts_list = self._parse_facts_response(response)
        if not facts_list:
            return

        doc_date_str = document.get("created") or ""
        doc_date = _parse_date(doc_date_str[:10]) if doc_date_str else None

        stored = 0
        for fact_data in facts_list:
            try:
                ft = fact_data.get("fact_type", "note")
                if ft not in VALID_FACT_TYPES:
                    ft = "note"

                et = fact_data.get("entity_type", "company")
                if et not in VALID_ENTITY_TYPES:
                    et = "company"

                entity_name = fact_data.get("entity_name", "").strip()
                if not entity_name:
                    entity_name = correspondent or owner

                entity_id = self.get_or_create_entity(et, entity_name,
                                                       fact_data.get("detail"))

                summary = fact_data.get("summary", "").strip()
                if not summary:
                    continue

                conf = float(fact_data.get("confidence", 0.7))
                fact_id = self.store_fact(
                    fact_type=ft,
                    summary=summary,
                    detail=fact_data.get("detail") or {},
                    entity_id=entity_id,
                    valid_from=fact_data.get("valid_from"),
                    valid_until=fact_data.get("valid_until"),
                    confidence=conf,
                    doc_id=doc_id,
                    doc_title=title,
                    doc_date=doc_date,
                    extraction_method="llm",
                )
                if fact_id:
                    stored += 1
            except Exception as exc:
                log.debug("KnowledgeDB: Fakt-Speicherung fehlgeschlagen: %s", exc)
                continue

        if stored:
            log.info("KnowledgeDB: %s Fakten aus Dokument #%s extrahiert", stored, doc_id)

    def _parse_facts_response(self, response: str) -> list[dict]:
        """Parse LLM response containing a JSON array of facts."""
        if not response:
            return []

        text = response.strip()

        # Strip markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Find JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end <= start:
            return []

        try:
            result = json.loads(text[start:end])
            if isinstance(result, list):
                return [f for f in result if isinstance(f, dict)]
        except json.JSONDecodeError:
            log.debug("KnowledgeDB: JSON-Parse-Fehler bei Fakten-Antwort")
        return []

    # -- Prompt Context Generation -------------------------------------------

    def build_prompt_context(self, owner: str, max_len: int = 400,
                             ref_date: date | None = None) -> str:
        """Generate knowledge context string for LLM prompts.

        Returns a compact text with facts AND actionable classification hints.
        """
        from . import config as _cfg
        max_facts = getattr(_cfg, "KNOWLEDGE_MAX_FACTS_IN_PROMPT", 12)

        if ref_date:
            facts = self.get_facts_at_date(ref_date)
        else:
            facts = self.get_current_facts()

        if not facts:
            return f"Dokumentenbesitzer: {owner}. Keine weiteren Fakten bekannt."

        category_map = {
            "employment": "Arbeitgeber",
            "education": "Ausbildung",
            "vehicle": "Fahrzeuge",
            "address": "Adressen",
            "insurance": "Versicherungen",
            "membership": "Mitgliedschaften",
            "health": "Gesundheit",
            "bank": "Finanzen",
            "financial": "Finanzen",
            "regular": "Abos/Dienste",
            "life": "Lebensereignisse",
            "note": "Sonstiges",
        }

        groups: dict[str, list[str]] = {}
        count = 0
        for fact in facts:
            if count >= max_facts:
                break
            ft = fact["fact_type"]
            category = "Sonstiges"
            for prefix, cat_name in category_map.items():
                if ft.startswith(prefix):
                    category = cat_name
                    break
            line = fact["summary"]
            if fact.get("valid_from"):
                line += f" (seit {fact['valid_from']})"
            if fact.get("valid_until"):
                line += f" (bis {fact['valid_until']})"
            groups.setdefault(category, []).append(line)
            count += 1

        parts = [f"BESITZER: {owner}"]
        for cat, lines in groups.items():
            parts.append(f"{cat}: " + " | ".join(lines))

        # Append classification hints
        hints = self.build_classification_hints(owner, facts)
        if hints:
            parts.append("ZUORDNUNGS-HINWEISE: " + " | ".join(hints))

        text = "\n".join(parts)
        if len(text) > max_len:
            text = text[:max_len - 3] + "..."
        return text

    def build_classification_hints(self, owner: str,
                                   facts: list[dict] | None = None) -> list[str]:
        """Generate actionable classification hints from ALL known facts.

        Fully dynamic: iterates all facts and generates routing hints based on
        whatever the DB has learned. No hardcoded company names or categories.
        Returns a list of short hint strings for the LLM prompt.
        """
        if facts is None:
            facts = self.get_current_facts()

        hints: list[str] = []

        # Mapping: fact_type prefix -> (label, path hint)
        _CATEGORY_LABELS = {
            "address": ("Wohnort", "Privat"),
            "employment": ("Arbeitgeber", "Arbeit"),
            "education": ("Ausbildung", "Arbeit/Ausbildung"),
            "insurance": ("Versicherung", "Versicherungen/[Typ]"),
            "health": ("Gesundheit", "Gesundheit"),
            "vehicle": ("Fahrzeug", "Auto/[Typ]"),
            "membership": ("Mitgliedschaft", "je nach Kontext"),
            "bank": ("Bank", "Finanzen/Bank"),
            "financial": ("Vertrag", "Finanzen"),
            "regular_service": ("Laufender Dienst", "Finanzen oder Freizeit"),
            "life_event": ("Lebensereignis", "je nach Kontext"),
        }

        # Group facts by category for compact output
        category_entities: dict[str, list[tuple[str, str, dict]]] = {}
        # (entity_name, summary, detail)
        education_active = False
        education_employer = ""
        education_institutions: list[str] = []
        current_city = ""
        current_employer = ""

        for fact in facts:
            ft = fact["fact_type"]
            detail = fact.get("detail") or {}
            entity_name = fact.get("entity_name", "")

            # Determine category
            category = None
            for prefix in _CATEGORY_LABELS:
                if ft.startswith(prefix):
                    category = prefix
                    break
            if not category:
                continue

            category_entities.setdefault(category, []).append(
                (entity_name, fact["summary"], detail)
            )

            # Track special context
            if ft in ("address_current", "address_move"):
                city = detail.get("stadt") or detail.get("city") or ""
                if city:
                    current_city = city

            if ft == "employment_active":
                current_employer = entity_name

            if ft in ("education_start", "education_active"):
                education_active = True
                inst = detail.get("institution") or ""
                if inst:
                    education_institutions.append(inst)
                rel = detail.get("related_to") or detail.get("arbeitgeber") or ""
                if rel:
                    education_employer = rel
                elif current_employer:
                    education_employer = current_employer

        # --- Generate hints per category ---

        # Address
        if current_city:
            hints.append(
                f"Wohnort={current_city} -> lokale Dokumente aus {current_city} = Privat"
            )

        # Education (special: includes related institutions)
        if education_active:
            edu_parts = ["IHK", "Berufsschule", "Handelskammer", "Pruefungsausschuss"]
            edu_parts.extend(education_institutions)
            context = f"Ausbildung aktiv"
            if education_employer:
                context += f" bei {education_employer}"
            hints.append(
                f"{context} -> Dokumente von {'/'.join(list(dict.fromkeys(edu_parts))[:5])} = Arbeit/Ausbildung"
            )

        # Employment
        if current_employer:
            hints.append(
                f"Aktueller AG={current_employer} -> dessen Dokumente = Arbeit"
            )

        # All other categories: dynamically generate hints from entities
        for category, entries in category_entities.items():
            if category in ("address", "employment", "education"):
                continue  # Already handled above

            label, path_hint = _CATEGORY_LABELS.get(category, ("", ""))
            if not label:
                continue

            # Collect unique entity names for this category
            names = list(dict.fromkeys(
                name for name, _, _ in entries if name
            ))
            if not names:
                continue

            # Vehicle: split by private/company
            if category == "vehicle":
                private_v = []
                company_v = []
                for name, _, detail in entries:
                    cat_val = detail.get("category") or detail.get("kategorie") or ""
                    if cat_val in ("firma", "company"):
                        company_v.append(name)
                    else:
                        private_v.append(name)
                if private_v:
                    hints.append(f"Privatfahrzeuge: {', '.join(private_v[:3])} -> Auto/*")
                if company_v:
                    hints.append(f"Firmenwagen: {', '.join(company_v[:3])} -> Arbeit/Auto")
                continue

            # Generic: "Known X: entity1, entity2 -> Path"
            names_str = ", ".join(names[:4])
            if len(names) > 4:
                names_str += f" (+{len(names)-4})"
            hints.append(f"{label}: {names_str} -> {path_hint}")

        return hints[:10]  # Max 10 hints to keep prompt compact

    # -- Migration -----------------------------------------------------------

    def migrate_from_learning_profile(self, profile_data: dict) -> dict:
        """One-time migration from learning_profile.json to Knowledge DB.

        Returns migration statistics.
        """
        stats = {"entities": 0, "facts": 0, "skipped": 0, "merged_ocr": 0}

        owner = profile_data.get("owner", "Edgar Richter")
        owner_id = self.get_or_create_entity("person", owner)
        stats["entities"] += 1

        # --- Known real employers ---
        real_employers = {
            "msg systems ag": {"start": "1990-01-01", "end": "2025-07-31"},
            "wbs training ag": {"start": "1990-01-01", "end": "2026-02-12"},
            "autohaus chemnitz gmbh": {"start": "2026-02-13", "end": ""},
        }

        # --- Entities to classify from jobs list ---
        # Known service providers / not employers
        service_companies = {
            "vodafone", "dhl", "check24", "sachsenenergie", "ecoflow",
            "lenovo", "galaxus", "softwarehunter", "oem-vertriebs",
            "plug & play", "battleground", "repurpose", "ticket i/o",
        }
        insurance_keywords = {"versicherung", "vvag"}
        # OCR duplicates of KRAILLMANN (use normalized forms)
        ocr_variants = {
            _normalize_name("KRAILLMANN AG"),
            _normalize_name("KRAßLMANN AG"),
            _normalize_name("KRAULLMANN AG"),
        }
        # Emails and persons to skip
        discard_patterns = [
            lambda n: "@" in n,  # email addresses
            lambda n: n in {"john doe", "estefania cassingena navone"},
        ]

        jobs = profile_data.get("jobs", [])
        for job in jobs:
            company = job.get("company", "").strip()
            if not company:
                continue
            company_norm = _normalize_name(company)

            # Check real employer
            is_real_employer = False
            for re_norm, dates in real_employers.items():
                if _fuzzy_ratio(company_norm, re_norm) >= 0.85:
                    is_real_employer = True
                    eid = self.get_or_create_entity("company", company)
                    stats["entities"] += 1
                    # Employment fact
                    start = dates["start"] or job.get("start")
                    end = dates["end"] or job.get("end")
                    ft = "employment_active" if not end else "employment_start"
                    self.store_fact(
                        fact_type=ft,
                        summary=f"{owner} arbeitet bei {company}",
                        detail={"source": "migration"},
                        entity_id=eid,
                        valid_from=start,
                        valid_until=end if end else None,
                        confidence=0.95,
                        extraction_method="migration",
                    )
                    stats["facts"] += 1
                    # Relation
                    self.store_relation(owner_id, "employs", eid,
                                        valid_from=start, valid_until=end if end else None)
                    break

            if is_real_employer:
                continue

            # Check discard patterns
            should_discard = False
            for pattern in discard_patterns:
                if pattern(company_norm):
                    should_discard = True
                    break
            if should_discard:
                stats["skipped"] += 1
                continue

            # OCR duplicates -> merge to KRAILLMANN AG
            if company_norm in ocr_variants:
                eid = self.get_or_create_entity("company", "KRAILLMANN AG")
                stats["merged_ocr"] += 1
                continue

            # Insurance
            if any(kw in company_norm for kw in insurance_keywords):
                eid = self.get_or_create_entity("insurance", company)
                self.store_fact(
                    fact_type="insurance_start",
                    summary=f"Versicherung bei {company}",
                    detail={"source": "migration"},
                    entity_id=eid,
                    valid_from=job.get("start"),
                    confidence=0.6,
                    extraction_method="migration",
                )
                stats["entities"] += 1
                stats["facts"] += 1
                continue

            # Service companies -> entity only, no employment fact
            is_service = False
            for svc in service_companies:
                if svc in company_norm:
                    is_service = True
                    break
            if is_service:
                self.get_or_create_entity("company", company)
                stats["entities"] += 1
                continue

            # Remaining: create as company, no employment fact
            # Includes: aconso, RSG Group, New Work SE, Baader Bank, Vivid, cronos,
            # MPD Fahrdienst, AT.U, corporate benefits, Scrum.org, etc.
            self.get_or_create_entity("company", company)
            stats["entities"] += 1

        # --- Vehicles ---
        vehicles = profile_data.get("vehicles", {})
        # Known bad vehicle names to discard
        bad_vehicle_patterns = [
            "beitragssatz",  # "Volkswagen Beitragssatz" = not a car
        ]

        for category, vehicle_list in vehicles.items():
            is_company = category == "company"
            for v_name in vehicle_list:
                v_name = v_name.strip()
                if not v_name:
                    continue
                v_norm = _normalize_name(v_name)

                # Discard known bad entries
                skip = False
                for bad in bad_vehicle_patterns:
                    if bad in v_norm:
                        stats["skipped"] += 1
                        skip = True
                        break
                if skip:
                    continue

                # Low confidence for truncated names (ends with -, or < 4 chars after brand)
                conf = 0.8
                if v_name.endswith("-") or len(v_name) < 4:
                    conf = 0.4

                # Check if name looks truncated (ends with 2-letter fragment)
                parts = v_name.split()
                if len(parts) >= 2 and len(parts[-1]) <= 2:
                    conf = 0.4

                vid = self.get_or_create_entity("vehicle", v_name,
                                                 {"category": "company" if is_company else "private"})
                stats["entities"] += 1

                self.store_fact(
                    fact_type="vehicle_acquired",
                    summary=f"Fahrzeug: {v_name} ({'Firmenwagen' if is_company else 'Privat'})",
                    detail={"category": "company" if is_company else "private",
                            "source": "migration"},
                    entity_id=vid,
                    confidence=conf,
                    extraction_method="migration",
                )
                stats["facts"] += 1

                # Relation: vehicle belongs_to owner (or employer if company)
                self.store_relation(vid, "belongs_to", owner_id)

        # --- Notes -> proper fact types ---
        notes = profile_data.get("notes", [])
        note_mappings = {
            "drk-mitglied": ("membership_active", "membership", "DRK"),
            "aok plus": ("health_provider", "company", "AOK PLUS"),
        }

        for note in notes:
            note_norm = _normalize_name(note)
            mapped = None
            for key, mapping in note_mappings.items():
                if key in note_norm:
                    mapped = mapping
                    break

            if mapped:
                ft, et, entity_name = mapped
                eid = self.get_or_create_entity(et, entity_name)
                self.store_fact(
                    fact_type=ft,
                    summary=f"{note}",
                    detail={"source": "migration"},
                    entity_id=eid,
                    confidence=0.8,
                    extraction_method="migration",
                )
                stats["entities"] += 1
                stats["facts"] += 1
                self.store_relation(owner_id, "member_of", eid)
            else:
                # Generic note
                self.store_fact(
                    fact_type="note",
                    summary=note,
                    detail={"source": "migration"},
                    entity_id=owner_id,
                    confidence=0.6,
                    extraction_method="migration",
                )
                stats["facts"] += 1

        log.info(
            "KnowledgeDB: Migration abgeschlossen: %s Entities, %s Fakten, %s uebersprungen, %s OCR-Merges",
            stats["entities"], stats["facts"], stats["skipped"], stats["merged_ocr"],
        )
        return stats

    # -- Statistics ----------------------------------------------------------

    def get_statistics(self) -> dict:
        """Get summary statistics for dashboard."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entities")
            entity_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM facts WHERE is_current = TRUE")
            active_facts = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM facts WHERE is_current = FALSE")
            superseded_facts = cur.fetchone()[0]

            cur.execute("SELECT COUNT(DISTINCT doc_id) FROM fact_sources")
            docs_with_facts = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM entity_relations WHERE is_current = TRUE")
            relations = cur.fetchone()[0]

            cur.execute(
                "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type ORDER BY COUNT(*) DESC"
            )
            entity_types = {r[0]: r[1] for r in cur.fetchall()}

            cur.execute(
                "SELECT fact_type, COUNT(*) FROM facts WHERE is_current = TRUE "
                "GROUP BY fact_type ORDER BY COUNT(*) DESC"
            )
            fact_types = {r[0]: r[1] for r in cur.fetchall()}

            return {
                "entities": entity_count,
                "active_facts": active_facts,
                "superseded_facts": superseded_facts,
                "docs_with_facts": docs_with_facts,
                "relations": relations,
                "entity_types": entity_types,
                "fact_types": fact_types,
            }

    def get_fact_timeline(self, limit: int = 50) -> list[dict]:
        """Get recent facts chronologically."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT f.id, f.fact_type, f.summary, f.valid_from, f.valid_until, "
                "f.confidence, f.is_current, f.created_at, "
                "e.name AS entity_name, e.entity_type "
                "FROM facts f "
                "LEFT JOIN entities e ON f.subject_entity_id = e.id "
                "ORDER BY COALESCE(f.valid_from, f.created_at::date) DESC, f.created_at DESC "
                "LIMIT %s",
                (limit,),
            )
            return [
                {"id": r[0], "fact_type": r[1], "summary": r[2],
                 "valid_from": r[3], "valid_until": r[4],
                 "confidence": r[5], "is_current": r[6], "created_at": r[7],
                 "entity_name": r[8], "entity_type": r[9]}
                for r in cur.fetchall()
            ]

    def store_manual_fact(self, fact_type: str, entity_type: str,
                          entity_name: str, summary: str,
                          valid_from: str | None = None,
                          valid_until: str | None = None) -> int | None:
        """Store a manually entered fact."""
        if fact_type not in VALID_FACT_TYPES:
            fact_type = "note"
        if entity_type not in VALID_ENTITY_TYPES:
            entity_type = "company"
        eid = self.get_or_create_entity(entity_type, entity_name)
        return self.store_fact(
            fact_type=fact_type,
            summary=summary,
            detail={"source": "manual"},
            entity_id=eid,
            valid_from=valid_from,
            valid_until=valid_until,
            confidence=1.0,
            extraction_method="manual",
        )

    def is_migrated(self) -> bool:
        """Check if migration has already been performed (any migration facts exist)."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM facts WHERE detail @> '{\"source\": \"migration\"}'::jsonb"
            )
            return cur.fetchone()[0] > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(val: Any) -> date | None:
    """Parse a date from various formats. Returns None on failure."""
    if val is None:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        val = val.strip()
        if not val or val == "null":
            return None
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(val[:10], fmt[:min(len(fmt), 10)]).date()
            except ValueError:
                continue
        # Try ISO format with just date part
        try:
            return datetime.strptime(val[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None
