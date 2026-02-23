"""Tests for paperless_organizer.knowledge (KnowledgeDB).

Uses a real PostgreSQL connection if KNOWLEDGE_DB_URL is set,
otherwise uses SQLite-backed mock tests for the helper functions.
"""

import json
import os
from datetime import date
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper function tests (no DB needed)
# ---------------------------------------------------------------------------

from paperless_organizer.knowledge import (
    VALID_ENTITY_TYPES,
    VALID_FACT_TYPES,
    VALID_RELATION_TYPES,
    _fuzzy_ratio,
    _normalize_name,
    _parse_date,
)


class TestNormalizeName:
    def test_basic(self):
        assert _normalize_name("  Müller GmbH  ") == "muller gmbh"

    def test_diacritics(self):
        assert _normalize_name("Café") == "cafe"
        # ß is not a combining diacritic, stays as-is
        assert _normalize_name("Straße") == "straße"

    def test_collapse_whitespace(self):
        assert _normalize_name("  A   B   C  ") == "a b c"

    def test_empty(self):
        assert _normalize_name("") == ""

    def test_uppercase(self):
        assert _normalize_name("KRAILLMANN AG") == "kraillmann ag"

    def test_ocr_variants(self):
        # These should normalize to very similar strings
        n1 = _normalize_name("KRAILLMANN AG")
        n2 = _normalize_name("KRAßLMANN AG")
        n3 = _normalize_name("KRAULLMANN AG")
        # n1 and n3 are closest
        assert _fuzzy_ratio(n1, n3) > 0.85


class TestFuzzyRatio:
    def test_identical(self):
        assert _fuzzy_ratio("hello", "hello") == 1.0

    def test_similar(self):
        assert _fuzzy_ratio("KRAILLMANN", "KRAULLMANN") > 0.8

    def test_different(self):
        assert _fuzzy_ratio("apple", "banana") < 0.5

    def test_empty(self):
        assert _fuzzy_ratio("", "hello") == 0.0
        assert _fuzzy_ratio("hello", "") == 0.0
        assert _fuzzy_ratio("", "") == 0.0

    def test_case_insensitive(self):
        assert _fuzzy_ratio("Hello", "hello") == 1.0


class TestParseDate:
    def test_iso_string(self):
        assert _parse_date("2025-01-15") == date(2025, 1, 15)

    def test_date_object(self):
        d = date(2024, 6, 1)
        assert _parse_date(d) is d

    def test_none(self):
        assert _parse_date(None) is None

    def test_empty_string(self):
        assert _parse_date("") is None
        assert _parse_date("null") is None

    def test_iso_with_time(self):
        assert _parse_date("2025-03-20T14:30:00") == date(2025, 3, 20)

    def test_invalid(self):
        assert _parse_date("not-a-date") is None


class TestConstants:
    def test_valid_fact_types_not_empty(self):
        assert len(VALID_FACT_TYPES) > 10

    def test_employment_types_present(self):
        assert "employment_start" in VALID_FACT_TYPES
        assert "employment_end" in VALID_FACT_TYPES
        assert "employment_active" in VALID_FACT_TYPES

    def test_vehicle_types_present(self):
        assert "vehicle_acquired" in VALID_FACT_TYPES
        assert "vehicle_accident" in VALID_FACT_TYPES

    def test_valid_entity_types(self):
        assert "person" in VALID_ENTITY_TYPES
        assert "vehicle" in VALID_ENTITY_TYPES
        assert "company" in VALID_ENTITY_TYPES
        assert "insurance" in VALID_ENTITY_TYPES

    def test_valid_relation_types(self):
        assert "belongs_to" in VALID_RELATION_TYPES
        assert "employs" in VALID_RELATION_TYPES


# ---------------------------------------------------------------------------
# KnowledgeDB tests (require PostgreSQL)
# ---------------------------------------------------------------------------

# Use the test DB URL or skip
_TEST_DB_URL = os.getenv("KNOWLEDGE_TEST_DB_URL", os.getenv("KNOWLEDGE_DB_URL", ""))
_HAS_PG = bool(_TEST_DB_URL)

requires_pg = pytest.mark.skipif(not _HAS_PG, reason="PostgreSQL not available (set KNOWLEDGE_TEST_DB_URL)")


@pytest.fixture
def knowledge_db():
    """Create a KnowledgeDB with clean tables for testing."""
    if not _HAS_PG:
        pytest.skip("PostgreSQL not available")

    from paperless_organizer.knowledge import KnowledgeDB
    db = KnowledgeDB(_TEST_DB_URL)

    # Clean all tables before test
    conn = db._get_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM fact_sources")
        cur.execute("DELETE FROM entity_relations")
        cur.execute("DELETE FROM facts")
        cur.execute("DELETE FROM entities")
    conn.commit()

    yield db
    db.close()


@requires_pg
class TestEntityCRUD:
    def test_create_entity(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "msg systems ag")
        assert eid > 0

    def test_find_existing(self, knowledge_db):
        eid1 = knowledge_db.get_or_create_entity("company", "msg systems ag")
        eid2 = knowledge_db.get_or_create_entity("company", "msg systems ag")
        assert eid1 == eid2

    def test_case_insensitive_dedup(self, knowledge_db):
        eid1 = knowledge_db.get_or_create_entity("company", "MSG Systems AG")
        eid2 = knowledge_db.get_or_create_entity("company", "msg systems ag")
        assert eid1 == eid2

    def test_diacritic_dedup(self, knowledge_db):
        eid1 = knowledge_db.get_or_create_entity("company", "Müller GmbH")
        eid2 = knowledge_db.get_or_create_entity("company", "Muller GmbH")
        assert eid1 == eid2

    def test_different_types(self, knowledge_db):
        eid1 = knowledge_db.get_or_create_entity("company", "AOK PLUS")
        eid2 = knowledge_db.get_or_create_entity("insurance", "AOK PLUS")
        assert eid1 != eid2

    def test_merge_attributes(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("person", "Edgar", {"role": "owner"})
        knowledge_db.get_or_create_entity("person", "Edgar", {"email": "edgar@test.de"})
        entities = knowledge_db.get_all_entities("person")
        edgar = next(e for e in entities if e["id"] == eid)
        assert edgar["attributes"].get("role") == "owner"
        assert edgar["attributes"].get("email") == "edgar@test.de"

    def test_find_entity(self, knowledge_db):
        knowledge_db.get_or_create_entity("company", "Test GmbH")
        assert knowledge_db.find_entity("company", "Test GmbH") is not None
        assert knowledge_db.find_entity("company", "NonExistent") is None

    def test_find_entity_fuzzy(self, knowledge_db):
        knowledge_db.get_or_create_entity("company", "KRAILLMANN AG")
        found = knowledge_db.find_entity_fuzzy("company", "KRAULLMANN AG", threshold=0.80)
        assert found is not None

    def test_get_all_entities(self, knowledge_db):
        knowledge_db.get_or_create_entity("company", "A GmbH")
        knowledge_db.get_or_create_entity("person", "B Person")
        all_e = knowledge_db.get_all_entities()
        assert len(all_e) >= 2
        companies = knowledge_db.get_all_entities("company")
        assert all(e["entity_type"] == "company" for e in companies)


@requires_pg
class TestFactCRUD:
    def test_store_fact(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "msg systems ag")
        fid = knowledge_db.store_fact(
            fact_type="employment_active",
            summary="Edgar arbeitet bei msg systems ag",
            entity_id=eid,
            confidence=0.9,
        )
        assert fid is not None and fid > 0

    def test_duplicate_detection(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "msg systems ag")
        fid1 = knowledge_db.store_fact(
            fact_type="employment_active",
            summary="Edgar arbeitet bei msg systems ag",
            entity_id=eid,
            confidence=0.9,
        )
        fid2 = knowledge_db.store_fact(
            fact_type="employment_active",
            summary="Edgar arbeitet bei msg systems ag",
            entity_id=eid,
            confidence=0.9,
        )
        # Should return same ID (duplicate detected)
        assert fid1 == fid2

    def test_low_confidence_rejected(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "Unsicher GmbH")
        fid = knowledge_db.store_fact(
            fact_type="note",
            summary="Vielleicht ein Kunde",
            entity_id=eid,
            confidence=0.2,  # Below default threshold of 0.4
        )
        assert fid is None

    def test_get_current_facts(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "WBS TRAINING AG")
        knowledge_db.store_fact(
            fact_type="employment_start",
            summary="Edgar begann bei WBS",
            entity_id=eid,
            confidence=0.9,
        )
        facts = knowledge_db.get_current_facts(fact_type="employment_start")
        assert len(facts) >= 1
        assert facts[0]["fact_type"] == "employment_start"

    def test_supersede_fact(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "Alt GmbH")
        fid1 = knowledge_db.store_fact(
            fact_type="employment_active",
            summary="Edgar arbeitet bei Alt GmbH",
            entity_id=eid,
            confidence=0.9,
        )
        fid2 = knowledge_db.store_fact(
            fact_type="employment_end",
            summary="Edgar hat Alt GmbH verlassen",
            entity_id=eid,
            confidence=0.9,
        )
        knowledge_db.supersede_fact(fid1, fid2)
        current = knowledge_db.get_current_facts(entity_id=eid)
        ids = [f["id"] for f in current]
        assert fid1 not in ids
        assert fid2 in ids

    def test_deactivate_fact(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "Test")
        fid = knowledge_db.store_fact(
            fact_type="note",
            summary="Test note",
            entity_id=eid,
            confidence=0.8,
        )
        knowledge_db.deactivate_fact(fid)
        current = knowledge_db.get_current_facts(entity_id=eid)
        assert fid not in [f["id"] for f in current]

    def test_fact_with_dates(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "msg systems ag")
        fid = knowledge_db.store_fact(
            fact_type="employment_start",
            summary="Employment started at msg",
            entity_id=eid,
            valid_from="2020-01-01",
            valid_until="2025-07-31",
            confidence=0.95,
        )
        facts = knowledge_db.get_facts_at_date(date(2023, 6, 15))
        ids = [f["id"] for f in facts]
        assert fid in ids

        # Should not be in future date
        facts_future = knowledge_db.get_facts_at_date(date(2026, 1, 1))
        ids_future = [f["id"] for f in facts_future]
        assert fid not in ids_future

    def test_fact_source_linked(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "Test Corp")
        fid = knowledge_db.store_fact(
            fact_type="note",
            summary="Unique test fact for source check",
            entity_id=eid,
            confidence=0.8,
            doc_id=42,
            doc_title="Test Document",
        )
        conn = knowledge_db._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id, doc_title FROM fact_sources WHERE fact_id = %s", (fid,))
            row = cur.fetchone()
        assert row is not None
        assert row[0] == 42
        assert row[1] == "Test Document"


@requires_pg
class TestRelations:
    def test_store_relation(self, knowledge_db):
        eid1 = knowledge_db.get_or_create_entity("person", "Edgar")
        eid2 = knowledge_db.get_or_create_entity("company", "msg systems ag")
        rid = knowledge_db.store_relation(eid1, "employs", eid2)
        assert rid > 0

    def test_duplicate_relation(self, knowledge_db):
        eid1 = knowledge_db.get_or_create_entity("person", "Edgar")
        eid2 = knowledge_db.get_or_create_entity("company", "msg systems ag")
        rid1 = knowledge_db.store_relation(eid1, "employs", eid2)
        rid2 = knowledge_db.store_relation(eid1, "employs", eid2)
        assert rid1 == rid2

    def test_get_relations(self, knowledge_db):
        eid1 = knowledge_db.get_or_create_entity("person", "Edgar")
        eid2 = knowledge_db.get_or_create_entity("vehicle", "VW Polo")
        knowledge_db.store_relation(eid2, "belongs_to", eid1)
        rels = knowledge_db.get_relations(eid1)
        assert len(rels) >= 1
        assert any(r["relation_type"] == "belongs_to" for r in rels)


@requires_pg
class TestPromptContext:
    def test_empty_db(self, knowledge_db):
        ctx = knowledge_db.build_prompt_context("Edgar Richter")
        assert "Edgar Richter" in ctx
        assert "Keine weiteren Fakten" in ctx

    def test_with_facts(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "msg systems ag")
        knowledge_db.store_fact(
            fact_type="employment_active",
            summary="Edgar arbeitet bei msg systems ag",
            entity_id=eid,
            confidence=0.95,
        )
        ctx = knowledge_db.build_prompt_context("Edgar Richter")
        assert "msg systems ag" in ctx
        assert "BESITZER: Edgar Richter" in ctx

    def test_max_length(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "Test")
        for i in range(20):
            knowledge_db.store_fact(
                fact_type="note",
                summary=f"Very long fact number {i} with lots of text to fill up space quickly",
                entity_id=eid,
                confidence=0.8,
            )
        ctx = knowledge_db.build_prompt_context("Edgar", max_len=200)
        assert len(ctx) <= 200


@requires_pg
class TestMigration:
    def test_migration_basic(self, knowledge_db):
        profile = {
            "owner": "Edgar Richter",
            "jobs": [
                {"company": "msg systems ag", "start": "1990-01-01", "end": "2025-07-31", "source": "seed"},
                {"company": "WBS TRAINING AG", "start": "1990-01-01", "end": "2026-02-12", "source": "seed"},
                {"company": "Autohaus Chemnitz GmbH", "start": "2026-02-13", "end": "", "source": "auto"},
            ],
            "vehicles": {"private": ["VW Polo"], "company": ["Toyota Corolla"]},
            "notes": ["DRK-Mitglied", "AOK PLUS"],
        }
        stats = knowledge_db.migrate_from_learning_profile(profile)
        assert stats["entities"] > 0
        assert stats["facts"] > 0

        # Check real employers have employment facts
        facts = knowledge_db.get_current_facts(fact_type="employment_active")
        summaries = [f["summary"] for f in facts]
        assert any("Autohaus Chemnitz" in s for s in summaries)

    def test_migration_filters_email(self, knowledge_db):
        profile = {
            "owner": "Test Owner",
            "jobs": [
                {"company": "Personalbetreuung@wbstraining.de", "start": "2024-02-29", "end": "", "source": "auto"},
            ],
            "vehicles": {"private": [], "company": []},
            "notes": [],
        }
        stats = knowledge_db.migrate_from_learning_profile(profile)
        assert stats["skipped"] >= 1

    def test_migration_filters_bad_vehicle(self, knowledge_db):
        profile = {
            "owner": "Test Owner",
            "jobs": [],
            "vehicles": {"private": ["Volkswagen Beitragssatz", "VW Polo"], "company": []},
            "notes": [],
        }
        stats = knowledge_db.migrate_from_learning_profile(profile)
        assert stats["skipped"] >= 1
        # VW Polo should be stored
        entities = knowledge_db.get_all_entities("vehicle")
        names = [e["name"] for e in entities]
        assert "VW Polo" in names
        assert "Volkswagen Beitragssatz" not in names

    def test_migration_merges_ocr(self, knowledge_db):
        profile = {
            "owner": "Test Owner",
            "jobs": [
                {"company": "KRAILLMANN AG", "start": "2024-07-24", "end": "", "source": "auto"},
                {"company": "KRAßLMANN AG", "start": "2024-03-08", "end": "", "source": "auto"},
                {"company": "KRAULLMANN AG", "start": "2023-03-08", "end": "", "source": "auto"},
            ],
            "vehicles": {"private": [], "company": []},
            "notes": [],
        }
        stats = knowledge_db.migrate_from_learning_profile(profile)
        assert stats["merged_ocr"] == 3
        # Should result in single entity
        entities = knowledge_db.get_all_entities("company")
        kraillmann = [e for e in entities if "kraillmann" in e["name"].lower()]
        assert len(kraillmann) == 1

    def test_migration_notes(self, knowledge_db):
        profile = {
            "owner": "Test Owner",
            "jobs": [],
            "vehicles": {"private": [], "company": []},
            "notes": ["DRK-Mitglied", "AOK PLUS"],
        }
        stats = knowledge_db.migrate_from_learning_profile(profile)
        facts = knowledge_db.get_current_facts(fact_type="membership_active")
        assert any("DRK" in f["summary"] for f in facts)
        facts_hp = knowledge_db.get_current_facts(fact_type="health_provider")
        assert any("AOK" in f["summary"] for f in facts_hp)

    def test_is_migrated(self, knowledge_db):
        assert not knowledge_db.is_migrated()
        profile = {
            "owner": "Test",
            "jobs": [{"company": "msg systems ag", "start": "2020-01-01", "end": "", "source": "seed"}],
            "vehicles": {"private": [], "company": []},
            "notes": [],
        }
        knowledge_db.migrate_from_learning_profile(profile)
        assert knowledge_db.is_migrated()


@requires_pg
class TestStatistics:
    def test_empty_stats(self, knowledge_db):
        stats = knowledge_db.get_statistics()
        assert stats["entities"] == 0
        assert stats["active_facts"] == 0

    def test_stats_after_insert(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "Test")
        knowledge_db.store_fact(
            fact_type="note",
            summary="A test fact",
            entity_id=eid,
            confidence=0.8,
        )
        stats = knowledge_db.get_statistics()
        assert stats["entities"] == 1
        assert stats["active_facts"] == 1


@requires_pg
class TestTimeline:
    def test_timeline(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "Test")
        knowledge_db.store_fact(
            fact_type="note", summary="Fact 1", entity_id=eid,
            valid_from="2024-01-01", confidence=0.8,
        )
        knowledge_db.store_fact(
            fact_type="note", summary="Fact 2", entity_id=eid,
            valid_from="2025-06-01", confidence=0.8,
        )
        timeline = knowledge_db.get_fact_timeline(limit=10)
        assert len(timeline) == 2
        # Most recent first
        assert timeline[0]["summary"] == "Fact 2"


# ---------------------------------------------------------------------------
# Parse facts response (no DB needed)
# ---------------------------------------------------------------------------

class TestParseFactsResponse:
    """Tests for _parse_facts_response (mocked DB)."""

    def _make_db(self):
        """Create a minimal KnowledgeDB mock for testing parse method."""
        from paperless_organizer.knowledge import KnowledgeDB
        db = MagicMock(spec=KnowledgeDB)
        db._parse_facts_response = KnowledgeDB._parse_facts_response.__get__(db)
        return db

    def test_valid_json_array(self):
        db = self._make_db()
        result = db._parse_facts_response('[{"fact_type": "note", "summary": "test"}]')
        assert len(result) == 1
        assert result[0]["summary"] == "test"

    def test_empty_array(self):
        db = self._make_db()
        result = db._parse_facts_response("[]")
        assert result == []

    def test_markdown_fenced(self):
        db = self._make_db()
        result = db._parse_facts_response('```json\n[{"fact_type": "note", "summary": "test"}]\n```')
        assert len(result) == 1

    def test_surrounding_text(self):
        db = self._make_db()
        result = db._parse_facts_response('Here are the facts:\n[{"fact_type": "note", "summary": "x"}]\nDone.')
        assert len(result) == 1

    def test_invalid_json(self):
        db = self._make_db()
        result = db._parse_facts_response("not json at all")
        assert result == []

    def test_empty_response(self):
        db = self._make_db()
        result = db._parse_facts_response("")
        assert result == []

    def test_filters_non_dicts(self):
        db = self._make_db()
        result = db._parse_facts_response('[{"fact_type": "note"}, "bad", 42]')
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Dynamic Knowledge Guardrails (require PostgreSQL)
# ---------------------------------------------------------------------------

from paperless_organizer.guardrails import _apply_knowledge_guardrails, _find_best_path


class TestFindBestPath:
    """Tests for _find_best_path helper."""

    def test_exact_match(self):
        paths = ["Arbeit", "Auto/Privat", "Versicherungen/KFZ", "Finanzen"]
        assert _find_best_path(paths, ["versicherungen"]) == "Versicherungen/KFZ"

    def test_first_term_priority(self):
        paths = ["Arbeit", "Auto/Privat", "Versicherungen/KFZ"]
        # "kfz" is more specific than "versicherungen" -> matches same path
        assert _find_best_path(paths, ["kfz", "versicherungen"]) == "Versicherungen/KFZ"

    def test_no_match(self):
        paths = ["Arbeit", "Auto/Privat"]
        assert _find_best_path(paths, ["versicherung"]) is None

    def test_empty_terms(self):
        paths = ["Arbeit"]
        assert _find_best_path(paths, []) is None
        assert _find_best_path(paths, [""]) is None


@requires_pg
class TestDynamicKnowledgeGuardrails:
    """Scenario tests for fully dynamic _apply_knowledge_guardrails.

    These test that the guardrails work for ANY category dynamically
    based on what the Knowledge DB has learned - not hardcoded to
    specific companies or categories.
    """

    STORAGE_PATHS = [
        {"name": "Arbeit"},
        {"name": "Arbeit/Ausbildung"},
        {"name": "Auto/Privat"},
        {"name": "Auto/Firmenwagen"},
        {"name": "Versicherungen/KFZ"},
        {"name": "Versicherungen/Haftpflicht"},
        {"name": "Gesundheit"},
        {"name": "Finanzen"},
        {"name": "Finanzen/Bank"},
        {"name": "Freizeit"},
        {"name": "Allgemein/Korrespondenz"},
    ]

    # --- Education scenarios ---

    def test_ihk_routed_to_arbeit_when_education_active(self, knowledge_db):
        """IHK document should go to Arbeit/Ausbildung when education is active."""
        eid = knowledge_db.get_or_create_entity("education", "IHK Chemnitz")
        knowledge_db.store_fact(
            fact_type="education_active",
            summary="Umschulung zum Fachinformatiker bei IHK Chemnitz",
            entity_id=eid,
            detail={"institution": "IHK Chemnitz", "related_to": "Autohaus Chemnitz"},
            confidence=0.9,
        )
        doc = {"content": "IHK Chemnitz Pruefungseinladung ...", "title": "IHK Einladung"}
        sug = {"title": "IHK Einladung", "correspondent": "IHK Chemnitz",
               "storage_path": "Allgemein/Korrespondenz"}
        fixes = _apply_knowledge_guardrails(doc, sug, self.STORAGE_PATHS, knowledge_db)
        assert len(fixes) >= 1
        assert "arbeit" in sug["storage_path"].lower()

    def test_berufsschule_routed_when_education_active(self, knowledge_db):
        """Berufsschule document routed to Arbeit when education is active."""
        eid = knowledge_db.get_or_create_entity("education", "BSZ Chemnitz")
        knowledge_db.store_fact(
            fact_type="education_active",
            summary="Berufsschulausbildung",
            entity_id=eid,
            detail={"institution": "BSZ Chemnitz"},
            confidence=0.9,
        )
        doc = {"content": "Berufsschule Zeugnis BSZ Chemnitz", "title": "Zeugnis"}
        sug = {"title": "Zeugnis", "correspondent": "BSZ Chemnitz",
               "storage_path": "Allgemein/Korrespondenz"}
        fixes = _apply_knowledge_guardrails(doc, sug, self.STORAGE_PATHS, knowledge_db)
        assert len(fixes) >= 1
        assert "arbeit" in sug["storage_path"].lower()

    # --- Insurance scenarios ---

    def test_insurance_rerouted_from_generic(self, knowledge_db):
        """Known insurance company rerouted from generic path to Versicherungen."""
        eid = knowledge_db.get_or_create_entity("insurance", "Ammerlaender Versicherung")
        knowledge_db.store_fact(
            fact_type="insurance_active",
            summary="KFZ-Versicherung bei Ammerlaender",
            entity_id=eid,
            detail={"typ": "KFZ"},
            confidence=0.9,
        )
        doc = {"content": "Ammerlaender Versicherung Beitrag ...", "title": "Beitrag"}
        sug = {"title": "Beitrag", "correspondent": "Ammerlaender Versicherung",
               "storage_path": "Allgemein/Korrespondenz"}
        fixes = _apply_knowledge_guardrails(doc, sug, self.STORAGE_PATHS, knowledge_db)
        assert len(fixes) >= 1
        assert "versicherung" in sug["storage_path"].lower()

    def test_insurance_already_correct_no_change(self, knowledge_db):
        """No correction if path is already correct."""
        eid = knowledge_db.get_or_create_entity("insurance", "HUK24")
        knowledge_db.store_fact(
            fact_type="insurance_active",
            summary="Haftpflichtversicherung bei HUK24",
            entity_id=eid,
            confidence=0.9,
        )
        doc = {"content": "HUK24 Beitrag", "title": "Beitrag"}
        sug = {"title": "Beitrag", "correspondent": "HUK24",
               "storage_path": "Versicherungen/Haftpflicht"}
        fixes = _apply_knowledge_guardrails(doc, sug, self.STORAGE_PATHS, knowledge_db)
        assert len(fixes) == 0
        assert sug["storage_path"] == "Versicherungen/Haftpflicht"

    # --- Health scenarios ---

    def test_health_provider_rerouted(self, knowledge_db):
        """Known health provider rerouted to Gesundheit."""
        eid = knowledge_db.get_or_create_entity("insurance", "AOK PLUS")
        knowledge_db.store_fact(
            fact_type="health_provider",
            summary="AOK PLUS ist Krankenkasse",
            entity_id=eid,
            confidence=0.9,
        )
        doc = {"content": "AOK PLUS Leistungsnachweis", "title": "Leistungsnachweis"}
        sug = {"title": "Leistungsnachweis", "correspondent": "AOK PLUS",
               "storage_path": "Allgemein/Korrespondenz"}
        fixes = _apply_knowledge_guardrails(doc, sug, self.STORAGE_PATHS, knowledge_db)
        assert len(fixes) >= 1
        assert "gesundheit" in sug["storage_path"].lower()

    # --- Vehicle scenarios ---

    def test_vehicle_rerouted_from_freizeit(self, knowledge_db):
        """Vehicle-related document rerouted from Freizeit to Auto."""
        eid = knowledge_db.get_or_create_entity("vehicle", "VW Polo")
        knowledge_db.store_fact(
            fact_type="vehicle_service",
            summary="TUeV fuer VW Polo",
            entity_id=eid,
            confidence=0.9,
        )
        doc = {"content": "VW Polo TUeV Bericht", "title": "TUeV"}
        sug = {"title": "TUeV", "correspondent": "VW Polo",
               "storage_path": "Freizeit"}
        fixes = _apply_knowledge_guardrails(doc, sug, self.STORAGE_PATHS, knowledge_db)
        assert len(fixes) >= 1
        assert "auto" in sug["storage_path"].lower()

    # --- Financial scenarios ---

    def test_bank_rerouted_to_finanzen(self, knowledge_db):
        """Known bank rerouted to Finanzen/Bank."""
        eid = knowledge_db.get_or_create_entity("company", "Sparkasse Chemnitz")
        knowledge_db.store_fact(
            fact_type="bank_account",
            summary="Girokonto bei Sparkasse Chemnitz",
            entity_id=eid,
            confidence=0.9,
        )
        doc = {"content": "Sparkasse Chemnitz Kontoauszug", "title": "Kontoauszug"}
        sug = {"title": "Kontoauszug", "correspondent": "Sparkasse Chemnitz",
               "storage_path": "Allgemein/Korrespondenz"}
        fixes = _apply_knowledge_guardrails(doc, sug, self.STORAGE_PATHS, knowledge_db)
        assert len(fixes) >= 1
        assert "finanzen" in sug["storage_path"].lower()

    # --- Membership scenarios ---

    def test_membership_rerouted_to_freizeit(self, knowledge_db):
        """Known membership rerouted to Freizeit."""
        eid = knowledge_db.get_or_create_entity("membership", "DRK")
        knowledge_db.store_fact(
            fact_type="membership_active",
            summary="DRK Mitgliedschaft aktiv",
            entity_id=eid,
            confidence=0.9,
        )
        doc = {"content": "DRK Beitragsrechnung", "title": "Beitrag DRK"}
        sug = {"title": "Beitrag DRK", "correspondent": "DRK",
               "storage_path": "Allgemein/Korrespondenz"}
        # "freizeit" matches membership path search terms
        fixes = _apply_knowledge_guardrails(doc, sug, self.STORAGE_PATHS, knowledge_db)
        assert len(fixes) >= 1
        assert "freizeit" in sug["storage_path"].lower()

    # --- Employment scenarios ---

    def test_employer_rerouted_to_arbeit(self, knowledge_db):
        """Known employer rerouted from generic to Arbeit."""
        eid = knowledge_db.get_or_create_entity("company", "Autohaus Chemnitz GmbH")
        knowledge_db.store_fact(
            fact_type="employment_active",
            summary="Edgar arbeitet bei Autohaus Chemnitz GmbH",
            entity_id=eid,
            confidence=0.95,
        )
        doc = {"content": "Autohaus Chemnitz GmbH Gehaltsabrechnung",
               "title": "Gehaltsabrechnung"}
        sug = {"title": "Gehaltsabrechnung",
               "correspondent": "Autohaus Chemnitz GmbH",
               "storage_path": "Allgemein/Korrespondenz"}
        fixes = _apply_knowledge_guardrails(doc, sug, self.STORAGE_PATHS, knowledge_db)
        assert len(fixes) >= 1
        assert "arbeit" in sug["storage_path"].lower()

    # --- No KB = no corrections ---

    def test_no_knowledge_db_returns_empty(self):
        """Without knowledge_db, no corrections."""
        doc = {"content": "test", "title": "test"}
        sug = {"title": "test", "storage_path": "Allgemein"}
        fixes = _apply_knowledge_guardrails(doc, sug, self.STORAGE_PATHS, knowledge_db=None)
        assert fixes == []

    # --- Regular service / financial contract ---

    def test_regular_service_rerouted_to_finanzen(self, knowledge_db):
        """Known service provider rerouted to Finanzen."""
        eid = knowledge_db.get_or_create_entity("company", "SachsenEnergie")
        knowledge_db.store_fact(
            fact_type="regular_service",
            summary="Stromvertrag bei SachsenEnergie",
            entity_id=eid,
            confidence=0.9,
        )
        doc = {"content": "SachsenEnergie Jahresabrechnung", "title": "Abrechnung"}
        sug = {"title": "Abrechnung", "correspondent": "SachsenEnergie",
               "storage_path": "Allgemein/Korrespondenz"}
        fixes = _apply_knowledge_guardrails(doc, sug, self.STORAGE_PATHS, knowledge_db)
        assert len(fixes) >= 1
        assert "finanzen" in sug["storage_path"].lower()


@requires_pg
class TestDynamicClassificationHints:
    """Tests for build_classification_hints - fully dynamic hint generation."""

    def test_empty_db_no_hints(self, knowledge_db):
        hints = knowledge_db.build_classification_hints("Edgar")
        assert hints == []

    def test_employer_hint(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("company", "Autohaus Chemnitz")
        knowledge_db.store_fact(
            fact_type="employment_active",
            summary="Edgar arbeitet bei Autohaus Chemnitz",
            entity_id=eid, confidence=0.9,
        )
        hints = knowledge_db.build_classification_hints("Edgar")
        assert any("Autohaus Chemnitz" in h for h in hints)
        assert any("Arbeit" in h for h in hints)

    def test_address_hint(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("address", "Chemnitz")
        knowledge_db.store_fact(
            fact_type="address_current",
            summary="Wohnt in Chemnitz",
            entity_id=eid,
            detail={"stadt": "Chemnitz"},
            confidence=0.9,
        )
        hints = knowledge_db.build_classification_hints("Edgar")
        assert any("Chemnitz" in h for h in hints)

    def test_education_hint(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("education", "IHK Chemnitz")
        knowledge_db.store_fact(
            fact_type="education_active",
            summary="Umschulung bei IHK Chemnitz",
            entity_id=eid,
            detail={"institution": "IHK Chemnitz"},
            confidence=0.9,
        )
        hints = knowledge_db.build_classification_hints("Edgar")
        assert any("Ausbildung" in h for h in hints)
        assert any("IHK" in h for h in hints)

    def test_insurance_hint(self, knowledge_db):
        eid = knowledge_db.get_or_create_entity("insurance", "HUK24")
        knowledge_db.store_fact(
            fact_type="insurance_active",
            summary="Haftpflicht bei HUK24",
            entity_id=eid, confidence=0.9,
        )
        hints = knowledge_db.build_classification_hints("Edgar")
        assert any("HUK24" in h for h in hints)
        assert any("Versicherung" in h for h in hints)

    def test_vehicle_hint_split_private_company(self, knowledge_db):
        eid1 = knowledge_db.get_or_create_entity("vehicle", "VW Polo")
        knowledge_db.store_fact(
            fact_type="vehicle_acquired",
            summary="VW Polo erworben",
            entity_id=eid1,
            detail={"kategorie": "privat"},
            confidence=0.9,
        )
        eid2 = knowledge_db.get_or_create_entity("vehicle", "Toyota Corolla")
        knowledge_db.store_fact(
            fact_type="vehicle_acquired",
            summary="Toyota Corolla Firmenwagen",
            entity_id=eid2,
            detail={"category": "firma"},
            confidence=0.9,
        )
        hints = knowledge_db.build_classification_hints("Edgar")
        assert any("Privatfahrzeug" in h for h in hints)
        assert any("Firmenwagen" in h for h in hints)

    def test_max_hints_capped(self, knowledge_db):
        """Hints are capped at 10 to keep prompt compact."""
        for i in range(15):
            eid = knowledge_db.get_or_create_entity("insurance", f"Versicherung {i}")
            knowledge_db.store_fact(
                fact_type="insurance_active",
                summary=f"Versicherung {i} aktiv",
                entity_id=eid, confidence=0.9,
            )
        hints = knowledge_db.build_classification_hints("Edgar")
        assert len(hints) <= 10
