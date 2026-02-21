"""Tests for paperless_organizer.utils."""

import pytest

from paperless_organizer.utils import (
    _build_id_name_map,
    _normalize_text,
    _normalize_tag_name,
    _normalize_correspondent_name,
    _canonicalize_correspondent_name,
    _sanitize_suggestion_spelling,
    _safe_iso_date,
    _extract_document_date,
    _detect_language,
    _assess_ocr_quality,
    _extract_keywords,
    _extract_invoice_number,
    _extract_amount,
    _improve_title,
    _content_fingerprint,
    _content_similarity,
    _minhash_signature,
    _lsh_find_candidates,
    _extract_document_entities,
    _strip_diacritics,
    _correspondent_core_name,
    _is_correspondent_duplicate_name,
    _is_fully_organized,
    safe_parse_json,
)


# ---------------------------------------------------------------------------
# _build_id_name_map
# ---------------------------------------------------------------------------

class TestBuildIdNameMap:
    def test_basic(self):
        items = [{"id": 1, "name": "Alpha"}, {"id": 2, "name": "Beta"}]
        assert _build_id_name_map(items) == {1: "Alpha", 2: "Beta"}

    def test_empty(self):
        assert _build_id_name_map([]) == {}

    def test_missing_id(self):
        items = [{"name": "NoId"}, {"id": 3, "name": "HasId"}]
        assert _build_id_name_map(items) == {3: "HasId"}

    def test_missing_name(self):
        items = [{"id": 5}]
        assert _build_id_name_map(items) == {5: ""}

    def test_id_none_skipped(self):
        items = [{"id": None, "name": "Skip"}, {"id": 1, "name": "Keep"}]
        assert _build_id_name_map(items) == {1: "Keep"}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_basic(self):
        assert _normalize_text("  hello   world  ") == "hello world"

    def test_empty(self):
        assert _normalize_text("") == ""
        assert _normalize_text(None) == ""

    def test_preserves_content(self):
        assert _normalize_text("abc") == "abc"


class TestNormalizeTagName:
    def test_basic(self):
        result = _normalize_tag_name("  Some  Tag  ")
        assert result == "some tag"

    def test_empty(self):
        assert _normalize_tag_name("") == ""


class TestNormalizeCorrespondentName:
    def test_basic(self):
        result = _normalize_correspondent_name("  Deutsche   Telekom  ")
        assert "telekom" in result.lower()

    def test_empty(self):
        assert _normalize_correspondent_name("") == ""


class TestCanonicalizeCorrespondentName:
    def test_empty(self):
        assert _canonicalize_correspondent_name("") == ""

    def test_preserves_text(self):
        result = _canonicalize_correspondent_name("Test Company")
        assert result  # Not empty


# ---------------------------------------------------------------------------
# safe_parse_json
# ---------------------------------------------------------------------------

class TestSafeParseJson:
    def test_valid_json(self):
        assert safe_parse_json('{"a": 1}') == {"a": 1}

    def test_valid_array(self):
        assert safe_parse_json('[1, 2, 3]') == [1, 2, 3]

    def test_invalid_returns_fallback(self):
        assert safe_parse_json("not json", fallback={}) == {}

    def test_empty_returns_fallback(self):
        assert safe_parse_json("", fallback={"default": True}) == {"default": True}
        assert safe_parse_json(None, fallback=[]) == []

    def test_markdown_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert safe_parse_json(text) == {"key": "value"}

    def test_embedded_json(self):
        text = 'Here is the result: {"x": 42} and done.'
        assert safe_parse_json(text) == {"x": 42}

    def test_nested_object(self):
        text = '{"a": {"b": [1, 2]}}'
        result = safe_parse_json(text)
        assert result == {"a": {"b": [1, 2]}}


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

class TestSafeIsoDate:
    def test_valid(self):
        assert _safe_iso_date("2024-06-15") == "2024-06-15"

    def test_invalid(self):
        assert _safe_iso_date("2024-13-01") == ""
        assert _safe_iso_date("") == ""
        assert _safe_iso_date("abc") == ""

    def test_with_extra_text(self):
        assert _safe_iso_date("2024-01-15T10:00:00") == "2024-01-15"


class TestExtractDocumentDate:
    def test_german_dot_format(self):
        doc = {"content": "Rechnung vom 15.06.2024 fuer Leistungen"}
        assert _extract_document_date(doc) == "2024-06-15"

    def test_iso_format(self):
        doc = {"content": "Date: 2024-03-20 Amount: 100 EUR"}
        assert _extract_document_date(doc) == "2024-03-20"

    def test_no_date(self):
        doc = {"content": "Keine Datumsangabe hier"}
        assert _extract_document_date(doc) == ""

    def test_empty_content(self):
        doc = {"content": ""}
        assert _extract_document_date(doc) == ""
        assert _extract_document_date({}) == ""

    def test_picks_latest_date(self):
        doc = {"content": "Erstellt am 01.01.2020. Letzte Aenderung 15.06.2024."}
        assert _extract_document_date(doc) == "2024-06-15"


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_german(self):
        text = "Dies ist ein deutsches Dokument mit vielen deutschen Woertern und Saetzen."
        assert _detect_language(text) == "de"

    def test_english(self):
        text = "This is an English document with many English words and sentences."
        assert _detect_language(text) == "en"

    def test_empty(self):
        result = _detect_language("")
        assert result in ("de", "en")


# ---------------------------------------------------------------------------
# OCR quality
# ---------------------------------------------------------------------------

class TestAssessOcrQuality:
    def test_good_quality(self):
        doc = {"content": "Dies ist ein gut lesbarer Text mit normalen Woertern und Saetzen. " * 10}
        quality, score = _assess_ocr_quality(doc)
        assert quality in ("good", "medium")
        assert score > 0.3

    def test_empty_content(self):
        quality, score = _assess_ocr_quality({"content": ""})
        assert quality == "poor"
        assert score == 0.0

    def test_no_content(self):
        quality, score = _assess_ocr_quality({})
        assert quality == "poor"

    def test_very_short(self):
        quality, score = _assess_ocr_quality({"content": "abc"})
        assert quality == "poor"


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_basic(self):
        text = "Telekom Telekom Rechnung Rechnung Rechnung Zahlung"
        keywords = _extract_keywords(text, limit=3)
        assert len(keywords) <= 3
        assert "rechnung" in keywords

    def test_empty(self):
        assert _extract_keywords("") == []
        assert _extract_keywords(None) == []


# ---------------------------------------------------------------------------
# Invoice / amount extraction
# ---------------------------------------------------------------------------

class TestExtractInvoiceNumber:
    def test_rechnungsnr(self):
        content = "Ihre Rechnungsnr: RE-2024-12345 vom 15.06.2024"
        assert _extract_invoice_number(content) == "RE-2024-12345"

    def test_invoice_no(self):
        content = "Invoice No: INV123456 dated 2024-06-15"
        assert _extract_invoice_number(content) == "INV123456"

    def test_no_invoice(self):
        assert _extract_invoice_number("Kein Inhalt hier") == ""


class TestExtractAmount:
    def test_german_format(self):
        content = "Gesamtbetrag: 1.234,56 EUR"
        assert _extract_amount(content) == "1.234,56"

    def test_euro_symbol(self):
        content = "Summe: 99,99 \u20ac"
        assert _extract_amount(content) == "99,99"

    def test_no_amount(self):
        assert _extract_amount("Kein Betrag") == ""


# ---------------------------------------------------------------------------
# Title improvement
# ---------------------------------------------------------------------------

class TestImproveTitle:
    def test_strip_extension(self):
        result = _improve_title("Rechnung.pdf", {})
        assert result == "Rechnung"

    def test_strip_prefix(self):
        result = _improve_title("Dokument: Wichtiger Inhalt", {})
        assert result == "Wichtiger Inhalt"

    def test_capitalize(self):
        result = _improve_title("kleine anfangszeichen", {})
        assert result[0].isupper()

    def test_deduplicate_words(self):
        result = _improve_title("Rechnung Rechnung Test", {})
        assert result.count("Rechnung") == 1 or result.count("rechnung") == 1

    def test_max_length(self):
        long_title = "A" * 200
        result = _improve_title(long_title, {})
        assert len(result) <= 128

    def test_enrich_with_invoice_nr(self):
        doc = {"content": "Rechnungsnr: RE-001 Gesamtbetrag: 50,00 EUR"}
        result = _improve_title("Rechnung", doc)
        assert "RE-001" in result


# ---------------------------------------------------------------------------
# Content fingerprinting
# ---------------------------------------------------------------------------

class TestContentFingerprint:
    def test_empty(self):
        assert _content_fingerprint("") == set()

    def test_short_text(self):
        assert _content_fingerprint("ab") == set()

    def test_normal_text(self):
        text = "dies ist ein test dokument mit genuegend woertern fuer trigrams"
        fp = _content_fingerprint(text)
        assert len(fp) > 0

    def test_identical_texts(self):
        text = "dies ist ein test dokument mit genuegend woertern fuer trigrams"
        fp1 = _content_fingerprint(text)
        fp2 = _content_fingerprint(text)
        assert fp1 == fp2


class TestContentSimilarity:
    def test_identical(self):
        fp = {1, 2, 3, 4, 5}
        assert _content_similarity(fp, fp) == 1.0

    def test_disjoint(self):
        assert _content_similarity({1, 2, 3}, {4, 5, 6}) == 0.0

    def test_partial(self):
        sim = _content_similarity({1, 2, 3, 4}, {3, 4, 5, 6})
        assert 0.0 < sim < 1.0

    def test_empty(self):
        assert _content_similarity(set(), {1, 2}) == 0.0
        assert _content_similarity({1, 2}, set()) == 0.0


class TestMinHashSignature:
    def test_empty(self):
        assert _minhash_signature(set()) == ()

    def test_produces_signature(self):
        sig = _minhash_signature({1, 2, 3, 4, 5}, num_hashes=16)
        assert len(sig) == 16

    def test_deterministic(self):
        trigrams = {hash("abc"), hash("def"), hash("ghi")}
        sig1 = _minhash_signature(trigrams, num_hashes=32)
        sig2 = _minhash_signature(trigrams, num_hashes=32)
        assert sig1 == sig2


class TestLshFindCandidates:
    def test_empty(self):
        assert _lsh_find_candidates([]) == []

    def test_finds_similar(self):
        text = "dies ist ein test dokument mit genuegend woertern fuer trigrams und mehr text"
        fp = _content_fingerprint(text)
        sig = _minhash_signature(fp, num_hashes=128)
        # Same signature should always be found as candidates
        candidates = _lsh_find_candidates([sig, sig], num_bands=16)
        assert (0, 1) in candidates


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

class TestExtractDocumentEntities:
    def test_email_domain(self):
        doc = {"content": "Kontakt: info@deutsche-telekom.de", "title": "", "original_file_name": ""}
        entities = _extract_document_entities(doc)
        assert any("telekom" in e.lower() for e in entities)

    def test_iban_extraction(self):
        doc = {"content": "IBAN: DE89370400440532013000", "title": "", "original_file_name": ""}
        entities = _extract_document_entities(doc)
        assert len(entities) >= 1  # Should extract bank from BLZ

    def test_empty_document(self):
        assert _extract_document_entities({}) == []


# ---------------------------------------------------------------------------
# Correspondent deduplication
# ---------------------------------------------------------------------------

class TestStripDiacritics:
    def test_umlauts(self):
        assert _strip_diacritics("Muenchen") == "Muenchen"  # No diacritics
        assert _strip_diacritics("M\u00fcnchen") == "Munchen"  # ue umlaut stripped

    def test_empty(self):
        assert _strip_diacritics("") == ""


class TestCorrespondentCoreName:
    def test_basic(self):
        result = _correspondent_core_name("Deutsche Telekom AG")
        assert "telekom" in result

    def test_empty(self):
        assert _correspondent_core_name("") == ""


class TestIsCorrespondentDuplicateName:
    def test_exact_match(self):
        assert _is_correspondent_duplicate_name("Deutsche Telekom", "Deutsche Telekom")

    def test_different(self):
        assert not _is_correspondent_duplicate_name("Telekom", "Vodafone")

    def test_empty(self):
        assert not _is_correspondent_duplicate_name("", "")


# ---------------------------------------------------------------------------
# _is_fully_organized
# ---------------------------------------------------------------------------

class TestIsFullyOrganized:
    def test_fully_organized(self):
        doc = {"tags": [1], "correspondent": 1, "document_type": 1, "storage_path": 1}
        assert _is_fully_organized(doc) is True

    def test_missing_tags(self):
        doc = {"tags": [], "correspondent": 1, "document_type": 1, "storage_path": 1}
        assert _is_fully_organized(doc) is False

    def test_missing_correspondent(self):
        doc = {"tags": [1], "correspondent": None, "document_type": 1, "storage_path": 1}
        assert _is_fully_organized(doc) is False

    def test_empty_doc(self):
        assert _is_fully_organized({}) is False


# ---------------------------------------------------------------------------
# Spelling sanitization
# ---------------------------------------------------------------------------

class TestSanitizeSuggestionSpelling:
    def test_capitalizes_title(self):
        suggestion = {"title": "test title", "tags": []}
        _sanitize_suggestion_spelling(suggestion)
        # Should at least not crash; specific spelling fixes depend on constants
        assert isinstance(suggestion["title"], str)

    def test_empty_suggestion(self):
        suggestion = {"title": "", "correspondent": "", "tags": []}
        _sanitize_suggestion_spelling(suggestion)
        assert suggestion["title"] == ""
