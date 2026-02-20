"""Tests for paperless_organizer.db (LocalStateDB)."""

import os
import sqlite3
import tempfile

import pytest

from paperless_organizer.db import LocalStateDB


@pytest.fixture
def db(tmp_path):
    """Create a fresh LocalStateDB in a temp directory."""
    db_path = str(tmp_path / "test_state.db")
    return LocalStateDB(db_path)


# ---------------------------------------------------------------------------
# Schema & indexes
# ---------------------------------------------------------------------------

class TestSchema:
    def test_tables_created(self, db):
        conn = sqlite3.connect(db.db_path)
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "runs" in tables
        assert "documents" in tables
        assert "tag_events" in tables
        assert "review_queue" in tables
        assert "confidence_log" in tables

    def test_indexes_created(self, db):
        conn = sqlite3.connect(db.db_path)
        indexes = {row[1] for row in conn.execute(
            "SELECT * FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        conn.close()
        expected = {
            "idx_documents_doc_id", "idx_documents_run_id",
            "idx_documents_status", "idx_documents_created_at",
            "idx_tag_events_run_id",
            "idx_review_queue_status", "idx_review_queue_doc_id",
            "idx_confidence_log_doc_id",
        }
        assert expected.issubset(indexes)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

class TestRuns:
    def test_start_and_finish(self, db):
        run_id = db.start_run("test_action", dry_run=True, llm_model="test", llm_url="http://test")
        assert isinstance(run_id, int)
        assert run_id > 0

        db.finish_run(run_id, {"docs_processed": 5})
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        conn.close()
        assert row["action"] == "test_action"
        assert row["dry_run"] == 1
        assert row["ended_at"] is not None
        assert "docs_processed" in row["summary_json"]

    def test_multiple_runs(self, db):
        id1 = db.start_run("action1", False, "m1", "u1")
        id2 = db.start_run("action2", True, "m2", "u2")
        assert id1 != id2
        assert id2 > id1


# ---------------------------------------------------------------------------
# Document recording
# ---------------------------------------------------------------------------

class TestDocuments:
    def test_record_document(self, db):
        run_id = db.start_run("test", False, "model", "url")
        doc = {"title": "Test Doc", "tags": [1, 2], "correspondent": 5}
        suggestion = {"title": "Better Title", "tags": ["Tag1"], "correspondent": "Firma"}

        db.record_document(run_id, doc_id=42, status="ok", document=doc, suggestion=suggestion)

        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM documents WHERE doc_id = 42").fetchone()
        conn.close()
        assert row is not None
        assert row["status"] == "ok"
        assert row["title_before"] == "Test Doc"
        assert row["title_after"] == "Better Title"

    def test_record_error_document(self, db):
        run_id = db.start_run("test", False, "m", "u")
        db.record_document(run_id, doc_id=99, status="error",
                          document={"title": "Broken"}, error="API timeout")

        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM documents WHERE doc_id = 99").fetchone()
        conn.close()
        assert row["status"] == "error"
        assert row["error_text"] == "API timeout"


# ---------------------------------------------------------------------------
# Tag events
# ---------------------------------------------------------------------------

class TestTagEvents:
    def test_record_tag_event(self, db):
        run_id = db.start_run("test", False, "m", "u")
        db.record_tag_event(run_id, doc_id=10, action="blocked",
                           tag_name="BadTag", detail="not in taxonomy")

        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM tag_events WHERE doc_id = 10").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0]["action"] == "blocked"
        assert rows[0]["tag_name"] == "BadTag"


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------

class TestReviewQueue:
    def test_enqueue_and_list(self, db):
        run_id = db.start_run("test", False, "m", "u")
        db.enqueue_review(run_id, doc_id=100, reason="low confidence",
                         suggestion={"title": "Test"})

        reviews = db.list_open_reviews()
        assert len(reviews) == 1
        assert reviews[0]["doc_id"] == 100
        assert reviews[0]["reason"] == "low confidence"

    def test_close_review(self, db):
        run_id = db.start_run("test", False, "m", "u")
        db.enqueue_review(run_id, doc_id=200, reason="no tags", suggestion=None)

        reviews = db.list_open_reviews()
        review_id = reviews[0]["id"]

        result = db.close_review(review_id)
        assert result is True

        open_reviews = db.list_open_reviews()
        assert len(open_reviews) == 0

    def test_close_already_closed(self, db):
        run_id = db.start_run("test", False, "m", "u")
        db.enqueue_review(run_id, doc_id=300, reason="test", suggestion=None)
        reviews = db.list_open_reviews()
        review_id = reviews[0]["id"]
        db.close_review(review_id)

        # Closing again should return False
        result = db.close_review(review_id)
        assert result is False

    def test_get_review_with_suggestion(self, db):
        run_id = db.start_run("test", False, "m", "u")
        db.enqueue_review(run_id, doc_id=400, reason="test",
                         suggestion={"title": "My Title", "tags": ["A", "B"]})

        reviews = db.list_open_reviews()
        review = db.get_review_with_suggestion(reviews[0]["id"])
        assert review is not None
        assert review["doc_id"] == 400
        assert "suggestion_json" in review

    def test_get_nonexistent_review(self, db):
        assert db.get_review_with_suggestion(99999) is None

    def test_duplicate_enqueue_updates(self, db):
        run_id = db.start_run("test", False, "m", "u")
        db.enqueue_review(run_id, doc_id=500, reason="reason1", suggestion=None)
        db.enqueue_review(run_id, doc_id=500, reason="reason2", suggestion=None)

        reviews = db.list_open_reviews()
        doc_500 = [r for r in reviews if r["doc_id"] == 500]
        assert len(doc_500) == 1
        assert doc_500[0]["reason"] == "reason2"


# ---------------------------------------------------------------------------
# Confidence logging
# ---------------------------------------------------------------------------

class TestConfidenceLog:
    def test_record_and_calibrate(self, db):
        db.record_confidence(doc_id=1, confidence="high", source="llm")
        db.record_confidence(doc_id=2, confidence="high", source="llm")
        db.record_confidence(doc_id=3, confidence="low", source="llm")

        cal = db.get_confidence_calibration()
        assert "high" in cal
        assert cal["high"]["total"] == 2
        assert cal["high"]["corrected"] == 0

    def test_mark_corrected(self, db):
        db.record_confidence(doc_id=10, confidence="medium")
        db.mark_confidence_corrected(doc_id=10)

        cal = db.get_confidence_calibration()
        assert cal["medium"]["corrected"] == 1


# ---------------------------------------------------------------------------
# Recent document status counting
# ---------------------------------------------------------------------------

class TestCountRecentDocStatuses:
    def test_count(self, db):
        run_id = db.start_run("test", False, "m", "u")
        db.record_document(run_id, doc_id=50, status="error",
                          document={"title": "T"}, error="fail")
        db.record_document(run_id, doc_id=50, status="error",
                          document={"title": "T"}, error="fail2")
        db.record_document(run_id, doc_id=50, status="ok",
                          document={"title": "T"})

        count = db.count_recent_document_statuses(50, ["error"], within_minutes=60)
        assert count == 2

    def test_empty(self, db):
        assert db.count_recent_document_statuses(999, ["error"], within_minutes=60) == 0

    def test_zero_minutes(self, db):
        assert db.count_recent_document_statuses(1, ["ok"], within_minutes=0) == 0


# ---------------------------------------------------------------------------
# Purge old runs
# ---------------------------------------------------------------------------

class TestPurge:
    def test_purge_nothing(self, db):
        result = db.purge_old_runs(keep_days=90)
        assert result["runs"] == 0

    def test_purge_keeps_recent(self, db):
        run_id = db.start_run("test", False, "m", "u")
        db.record_document(run_id, doc_id=1, status="ok", document={"title": "T"})

        result = db.purge_old_runs(keep_days=1)
        assert result["runs"] == 0  # Just created, should be kept


# ---------------------------------------------------------------------------
# Processing stats
# ---------------------------------------------------------------------------

class TestProcessingStats:
    def test_empty(self, db):
        stats = db.get_processing_stats(days=30)
        assert stats["total_docs"] == 0
        assert stats["total_runs"] == 0

    def test_with_data(self, db):
        run_id = db.start_run("organize", False, "m", "u")
        db.record_document(run_id, 1, "ok", {"title": "A"})
        db.record_document(run_id, 2, "error", {"title": "B"}, error="fail")
        db.record_document(run_id, 3, "updated", {"title": "C"})

        stats = db.get_processing_stats(days=30)
        assert stats["total_docs"] == 3
        assert stats["total_runs"] == 1


# ---------------------------------------------------------------------------
# Monthly report
# ---------------------------------------------------------------------------

class TestMonthlyReport:
    def test_empty_month(self, db):
        report = db.generate_monthly_report(2025, 1)
        assert report["total_runs"] == 0
        assert report["total_docs"] == 0
