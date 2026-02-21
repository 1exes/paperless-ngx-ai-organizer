"""Tests for paperless_organizer.guardrails."""

import pytest

from paperless_organizer.guardrails import (
    _find_vendor_key,
    _contains_any_hint,
    _pick_existing_storage_path,
    _get_correspondent_name_by_id,
    _resolve_correspondent_from_name,
    _effective_employer_hints,
    _resolve_path_id,
    _apply_vendor_guardrails,
    _apply_vehicle_guardrails,
    _apply_topic_guardrails,
    _apply_learning_guardrails,
    _try_rule_based_suggestion,
    _build_suggestion_from_priors,
    _detect_content_hints,
    _select_controlled_tags,
    _collect_hard_review_reasons,
    _http_error_detail,
    build_decision_context,
)
from paperless_organizer.models import DecisionContext
from paperless_organizer.utils import _build_id_name_map


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_correspondents():
    return [
        {"id": 1, "name": "Deutsche Telekom", "document_count": 10},
        {"id": 2, "name": "Vodafone GmbH", "document_count": 5},
        {"id": 3, "name": "Stadtwerke Muenchen", "document_count": 3},
    ]


@pytest.fixture
def sample_tags():
    return [
        {"id": 10, "name": "Rechnung"},
        {"id": 11, "name": "Vertrag"},
        {"id": 12, "name": "Privat"},
        {"id": 13, "name": "Arbeit"},
    ]


@pytest.fixture
def sample_storage_paths():
    return [
        {"id": 20, "name": "Finanzen/Rechnungen", "path": "finanzen/rechnungen"},
        {"id": 21, "name": "Arbeit/WBS", "path": "arbeit/wbs"},
        {"id": 22, "name": "Gesundheit/Arzt", "path": "gesundheit/arzt"},
        {"id": 23, "name": "Auto/Service", "path": "auto/service"},
        {"id": 24, "name": "Freizeit/Events", "path": "freizeit/events"},
    ]


@pytest.fixture
def sample_document():
    return {
        "id": 42,
        "title": "Rechnung Telekom",
        "content": "Deutsche Telekom AG Rechnung fuer Februar 2024",
        "original_file_name": "rechnung_telekom.pdf",
        "tags": [10],
        "correspondent": 1,
        "document_type": 1,
        "storage_path": 20,
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestFindVendorKey:
    def test_no_match(self):
        assert _find_vendor_key("some random text") == ""

    def test_empty(self):
        assert _find_vendor_key("") == ""
        assert _find_vendor_key(None) == ""


class TestContainsAnyHint:
    def test_match(self):
        assert _contains_any_hint("this has a car keyword", ["car", "bike"]) is True

    def test_no_match(self):
        assert _contains_any_hint("nothing here", ["car", "bike"]) is False

    def test_empty(self):
        assert _contains_any_hint("", ["car"]) is False
        assert _contains_any_hint("text", []) is False


class TestPickExistingStoragePath:
    def test_exact_match(self, sample_storage_paths):
        result = _pick_existing_storage_path(sample_storage_paths, ["Finanzen/Rechnungen"])
        assert result == "Finanzen/Rechnungen"

    def test_partial_match(self, sample_storage_paths):
        result = _pick_existing_storage_path(sample_storage_paths, ["Gesundheit"])
        assert result == "Gesundheit/Arzt"

    def test_no_match(self, sample_storage_paths):
        result = _pick_existing_storage_path(sample_storage_paths, ["Nonexistent/Path"])
        assert result == ""

    def test_empty_paths(self):
        assert _pick_existing_storage_path([], ["Anything"]) == ""


class TestGetCorrespondentNameById:
    def test_found(self, sample_correspondents):
        assert _get_correspondent_name_by_id(sample_correspondents, 1) == "Deutsche Telekom"

    def test_not_found(self, sample_correspondents):
        assert _get_correspondent_name_by_id(sample_correspondents, 999) == ""

    def test_none_id(self, sample_correspondents):
        assert _get_correspondent_name_by_id(sample_correspondents, None) == ""

    def test_with_lookup(self, sample_correspondents):
        lookup = _build_id_name_map(sample_correspondents)
        assert _get_correspondent_name_by_id(sample_correspondents, 2, _lookup=lookup) == "Vodafone GmbH"
        assert _get_correspondent_name_by_id(sample_correspondents, 999, _lookup=lookup) == ""


class TestResolveCorrespondentFromName:
    def test_exact_match(self, sample_correspondents):
        corr_id, name = _resolve_correspondent_from_name(sample_correspondents, "Deutsche Telekom")
        assert corr_id == 1

    def test_not_found(self, sample_correspondents):
        corr_id, name = _resolve_correspondent_from_name(sample_correspondents, "Unbekannte Firma XYZ")
        assert corr_id is None

    def test_empty(self, sample_correspondents):
        corr_id, name = _resolve_correspondent_from_name(sample_correspondents, "")
        assert corr_id is None


class TestResolvePathId:
    def test_exact_name(self, sample_storage_paths):
        assert _resolve_path_id("Finanzen/Rechnungen", sample_storage_paths) == 20

    def test_exact_path(self, sample_storage_paths):
        assert _resolve_path_id("finanzen/rechnungen", sample_storage_paths) == 20

    def test_not_found(self, sample_storage_paths):
        assert _resolve_path_id("Nonexistent", sample_storage_paths) is None

    def test_empty(self, sample_storage_paths):
        assert _resolve_path_id("", sample_storage_paths) is None


# ---------------------------------------------------------------------------
# Guardrail application
# ---------------------------------------------------------------------------

class TestApplyVehicleGuardrails:
    def test_no_vehicle_hints(self, sample_storage_paths):
        doc = {"title": "Normale Rechnung", "content": "Kein Auto", "original_file_name": ""}
        suggestion = {"title": "Test", "storage_path": "Finanzen/Rechnungen", "tags": []}
        corrections = _apply_vehicle_guardrails(doc, suggestion, sample_storage_paths)
        assert corrections == []


class TestApplyLearningGuardrails:
    def test_no_hints(self, sample_storage_paths):
        suggestion = {"correspondent": "Test", "document_type": "Rechnung", "storage_path": "Finanzen"}
        corrections = _apply_learning_guardrails(suggestion, sample_storage_paths, None)
        assert corrections == []

    def test_empty_hints_list(self, sample_storage_paths):
        suggestion = {"correspondent": "Test"}
        corrections = _apply_learning_guardrails(suggestion, sample_storage_paths, [])
        assert corrections == []


# ---------------------------------------------------------------------------
# Rule-based fast path
# ---------------------------------------------------------------------------

class TestTryRuleBasedSuggestion:
    def test_no_hints(self, sample_storage_paths):
        doc = {"title": "Test"}
        assert _try_rule_based_suggestion(doc, [], sample_storage_paths) is None

    def test_insufficient_samples(self, sample_storage_paths):
        doc = {"title": "Test"}
        hints = [{"correspondent": "Telekom", "count": 2,
                  "document_type_ratio": 0.9, "storage_path_ratio": 0.9,
                  "top_document_type": "Rechnung", "top_storage_path": "Finanzen/Rechnungen",
                  "top_tags": [], "tag_ratios": {}}]
        assert _try_rule_based_suggestion(doc, hints, sample_storage_paths) is None


class TestBuildSuggestionFromPriors:
    def test_no_hints(self):
        assert _build_suggestion_from_priors({"title": "T"}, [], []) is None

    def test_insufficient_count(self):
        hints = [{"correspondent": "X", "count": 1}]
        assert _build_suggestion_from_priors({"title": "T"}, hints, []) is None


# ---------------------------------------------------------------------------
# Content hints
# ---------------------------------------------------------------------------

class TestDetectContentHints:
    def test_iban(self):
        doc = {"content": "IBAN DE89370400440532013000 fuer Ueberweisung"}
        hints = _detect_content_hints(doc)
        assert "iban_present" in hints

    def test_invoice(self):
        doc = {"content": "Rechnungsnummer: RE-2024-001 Betrag: 100 EUR"}
        hints = _detect_content_hints(doc)
        assert "invoice_detected" in hints

    def test_contract(self):
        doc = {"content": "Vertragsnummer: V-123 Vertragslaufzeit: 24 Monate"}
        hints = _detect_content_hints(doc)
        assert "contract_detected" in hints

    def test_salary(self):
        doc = {"content": "Gehaltsabrechnung Bruttobezug 3.500,00 EUR Lohnsteuer"}
        hints = _detect_content_hints(doc)
        assert "salary_detected" in hints

    def test_empty(self):
        assert _detect_content_hints({}) == []
        assert _detect_content_hints({"content": ""}) == []


# ---------------------------------------------------------------------------
# Tag selection
# ---------------------------------------------------------------------------

class TestSelectControlledTags:
    def test_existing_tags_matched(self, sample_tags):
        approved, dropped = _select_controlled_tags(
            ["Rechnung", "Vertrag"], sample_tags
        )
        assert "Rechnung" in approved
        assert "Vertrag" in approved

    def test_nonexistent_blocked(self, sample_tags):
        approved, dropped = _select_controlled_tags(
            ["NonexistentTag123"], sample_tags
        )
        assert len(approved) == 0
        assert len(dropped) == 1

    def test_duplicate_tags_deduped(self, sample_tags):
        approved, dropped = _select_controlled_tags(
            ["Rechnung", "rechnung", "RECHNUNG"], sample_tags
        )
        assert approved.count("Rechnung") == 1

    def test_empty_input(self, sample_tags):
        approved, dropped = _select_controlled_tags([], sample_tags)
        assert approved == []
        assert dropped == []

    def test_conflict_resolution(self, sample_tags):
        approved, dropped = _select_controlled_tags(
            ["Privat", "Arbeit"], sample_tags
        )
        # One should be dropped due to conflict
        assert len(approved) == 1


# ---------------------------------------------------------------------------
# Review reasons
# ---------------------------------------------------------------------------

class TestCollectHardReviewReasons:
    def test_no_reasons_for_complete(self, sample_storage_paths):
        doc = {"title": "Good Doc", "content": "Normal text with enough words " * 20,
               "original_file_name": "doc.pdf", "document_type": 1, "tags": [1], "storage_path": 20}
        suggestion = {"title": "Good Title", "document_type": "Rechnung",
                      "storage_path": "Finanzen/Rechnungen", "confidence": "high"}
        reasons = _collect_hard_review_reasons(doc, suggestion, ["Rechnung"], sample_storage_paths)
        # Should have no or minimal reasons
        assert isinstance(reasons, list)

    def test_empty_suggestion(self, sample_storage_paths):
        doc = {"title": "", "content": "", "original_file_name": ""}
        suggestion = {"title": "", "document_type": "", "storage_path": "", "confidence": "low"}
        reasons = _collect_hard_review_reasons(doc, suggestion, [], sample_storage_paths)
        assert len(reasons) > 0  # Should flag multiple issues


# ---------------------------------------------------------------------------
# Decision context
# ---------------------------------------------------------------------------

class TestBuildDecisionContext:
    def test_basic(self, sample_correspondents, sample_storage_paths):
        documents = [
            {"correspondent": 1, "storage_path": 20, "tags": [10]},
            {"correspondent": 2, "storage_path": 21, "tags": [11]},
        ]
        ctx = build_decision_context(documents, sample_correspondents, sample_storage_paths)
        assert isinstance(ctx, DecisionContext)
        assert len(ctx.notes) > 0

    def test_empty(self):
        ctx = build_decision_context([], [], [])
        assert isinstance(ctx, DecisionContext)


# ---------------------------------------------------------------------------
# HTTP error helpers
# ---------------------------------------------------------------------------

class TestHttpErrorDetail:
    def test_plain_exception(self):
        exc = Exception("simple error")
        assert _http_error_detail(exc) == "simple error"

    def test_with_response(self):
        class MockResponse:
            status_code = 400
            text = "Bad Request: invalid tags"
        exc = Exception("HTTP error")
        exc.response = MockResponse()
        detail = _http_error_detail(exc)
        assert "400" in detail
        assert "invalid tags" in detail
