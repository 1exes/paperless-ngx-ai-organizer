"""Local SQLite storage for run/document history."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta


class LocalStateDB:
    """Simple local SQLite storage for run/document history."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        self._harden_permissions()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _harden_permissions(self):
        try:
            if os.path.exists(self.db_path):
                os.chmod(self.db_path, 0o600)
        except Exception:
            pass

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    action TEXT NOT NULL,
                    dry_run INTEGER NOT NULL,
                    llm_model TEXT,
                    llm_url TEXT,
                    summary_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    doc_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    title_before TEXT,
                    title_after TEXT,
                    tags_before TEXT,
                    tags_after TEXT,
                    correspondent_before TEXT,
                    correspondent_after TEXT,
                    error_text TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tag_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    doc_id INTEGER,
                    action TEXT NOT NULL,
                    tag_name TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    doc_id INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    suggestion_json TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS confidence_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    predicted_confidence TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'llm',
                    was_corrected INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            # Performance indexes for frequently queried columns
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents (doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_run_id ON documents (run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tag_events_run_id ON tag_events (run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_review_queue_doc_id ON review_queue (doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence_log_doc_id ON confidence_log (doc_id)")

    def record_confidence(self, doc_id: int, confidence: str, source: str = "llm"):
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO confidence_log (doc_id, predicted_confidence, source, created_at) VALUES (?, ?, ?, ?)",
                (doc_id, confidence.lower(), source, now),
            )

    def mark_confidence_corrected(self, doc_id: int):
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM confidence_log WHERE doc_id = ? AND was_corrected = 0 "
                "ORDER BY id DESC LIMIT 1",
                (doc_id,),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE confidence_log SET was_corrected = 1 WHERE id = ?",
                    (row[0],),
                )

    def get_confidence_calibration(self) -> dict:
        """Returns calibration stats: for each confidence level, how often was it corrected."""
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT predicted_confidence, COUNT(*) AS total, "
                "SUM(was_corrected) AS corrected FROM confidence_log "
                "GROUP BY predicted_confidence"
            ).fetchall()
            result = {}
            for row in rows:
                total = row["total"]
                corrected = row["corrected"] or 0
                accuracy = ((total - corrected) / total * 100) if total > 0 else 0.0
                result[row["predicted_confidence"]] = {
                    "total": total, "corrected": corrected, "accuracy": round(accuracy, 1),
                }
            return result

    def start_run(self, action: str, dry_run: bool, llm_model: str, llm_url: str) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (started_at, action, dry_run, llm_model, llm_url)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now, action, int(dry_run), llm_model, llm_url),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, summary: dict):
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET ended_at = ?, summary_json = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), json.dumps(summary, ensure_ascii=False), run_id),
            )

    def record_document(self, run_id: int, doc_id: int, status: str,
                        document: dict, suggestion: dict | None = None, error: str = ""):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    run_id, doc_id, status, title_before, title_after,
                    tags_before, tags_after, correspondent_before, correspondent_after,
                    error_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    doc_id,
                    status,
                    document.get("title", ""),
                    (suggestion or {}).get("title", ""),
                    json.dumps(document.get("tags") or []),
                    json.dumps((suggestion or {}).get("tags") or []),
                    str(document.get("correspondent") or ""),
                    str((suggestion or {}).get("correspondent") or ""),
                    error[:500],
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def record_tag_event(self, run_id: int | None, doc_id: int | None, action: str, tag_name: str, detail: str):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tag_events (run_id, doc_id, action, tag_name, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    doc_id,
                    action,
                    tag_name,
                    detail[:500],
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def enqueue_review(self, run_id: int | None, doc_id: int, reason: str, suggestion: dict | None):
        now = datetime.now().isoformat(timespec="seconds")
        payload = json.dumps(suggestion or {}, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            recent_cutoff = (datetime.now() - timedelta(hours=24)).isoformat(timespec="seconds")
            recently_resolved = conn.execute(
                "SELECT id FROM review_queue WHERE doc_id = ? AND status = 'resolved' AND updated_at > ? LIMIT 1",
                (doc_id, recent_cutoff),
            ).fetchone()
            if recently_resolved:
                return

            existing = conn.execute(
                "SELECT id FROM review_queue WHERE doc_id = ? AND status = 'open' ORDER BY id DESC LIMIT 1",
                (doc_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE review_queue
                    SET run_id = ?, reason = ?, suggestion_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (run_id, reason[:800], payload, now, int(existing[0])),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO review_queue (run_id, doc_id, reason, suggestion_json, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'open', ?, ?)
                    """,
                    (run_id, doc_id, reason[:800], payload, now, now),
                )

    def list_open_reviews(self, limit: int = 50) -> list[dict]:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, doc_id, reason, created_at, updated_at
                FROM review_queue
                WHERE status = 'open'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_review_with_suggestion(self, review_id: int) -> dict | None:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, doc_id, suggestion_json FROM review_queue WHERE id = ?",
                (review_id,),
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def close_review(self, review_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE review_queue SET status = 'resolved', updated_at = ? WHERE id = ? AND status = 'open'",
                (datetime.now().isoformat(timespec="seconds"), review_id),
            )
            return cur.rowcount > 0

    def count_recent_document_statuses(self, doc_id: int, statuses: list[str], within_minutes: int) -> int:
        if not statuses or within_minutes <= 0:
            return 0
        since = (datetime.now() - timedelta(minutes=within_minutes)).isoformat(timespec="seconds")
        placeholders = ",".join("?" for _ in statuses)
        params = [doc_id, *statuses, since]
        query = (
            f"SELECT COUNT(*) FROM documents "
            f"WHERE doc_id = ? AND status IN ({placeholders}) AND created_at >= ?"
        )
        with self._lock, self._connect() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
            return int((row or [0])[0])

    def purge_old_runs(self, keep_days: int = 90) -> dict:
        """Delete runs and documents older than keep_days. Returns counts of deleted rows."""
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            old_run_ids = [
                row[0] for row in
                conn.execute("SELECT id FROM runs WHERE started_at < ?", (cutoff,)).fetchall()
            ]
            if not old_run_ids:
                return {"runs": 0, "documents": 0, "tag_events": 0, "reviews": 0}
            placeholders = ",".join("?" for _ in old_run_ids)
            docs_deleted = conn.execute(
                f"DELETE FROM documents WHERE run_id IN ({placeholders})", old_run_ids
            ).rowcount
            tags_deleted = conn.execute(
                f"DELETE FROM tag_events WHERE run_id IN ({placeholders})", old_run_ids
            ).rowcount
            reviews_deleted = conn.execute(
                "DELETE FROM review_queue WHERE status = 'resolved' AND updated_at < ?", (cutoff,)
            ).rowcount
            runs_deleted = conn.execute(
                f"DELETE FROM runs WHERE id IN ({placeholders})", old_run_ids
            ).rowcount
            return {"runs": runs_deleted, "documents": docs_deleted,
                    "tag_events": tags_deleted, "reviews": reviews_deleted}

    def generate_monthly_report(self, year: int, month: int) -> dict:
        """Monatlichen Report aus SQLite-Daten generieren."""
        start = f"{year:04d}-{month:02d}-01T00:00:00"
        if month == 12:
            end = f"{year + 1:04d}-01-01T00:00:00"
        else:
            end = f"{year:04d}-{month + 1:02d}-01T00:00:00"

        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM runs WHERE started_at >= ? AND started_at < ?",
                (start, end),
            ).fetchone()
            total_runs = row["cnt"] if row else 0

            rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM documents "
                "WHERE created_at >= ? AND created_at < ? GROUP BY status",
                (start, end),
            ).fetchall()
            status_counts = {r["status"]: r["cnt"] for r in rows}
            total_docs = sum(status_counts.values())
            success_docs = status_counts.get("ok", 0) + status_counts.get("applied", 0)
            success_rate = (success_docs / total_docs * 100) if total_docs > 0 else 0.0

            new_corrs = conn.execute(
                "SELECT DISTINCT correspondent_after FROM documents "
                "WHERE created_at >= ? AND created_at < ? "
                "AND (correspondent_before IS NULL OR correspondent_before = '') "
                "AND correspondent_after IS NOT NULL AND correspondent_after != ''",
                (start, end),
            ).fetchall()
            new_correspondents = [r["correspondent_after"] for r in new_corrs]

            deleted_tags = conn.execute(
                "SELECT tag_name, COUNT(*) AS cnt FROM tag_events "
                "WHERE action = 'delete' AND created_at >= ? AND created_at < ? "
                "GROUP BY tag_name ORDER BY cnt DESC",
                (start, end),
            ).fetchall()
            deleted_tag_list = [{"name": r["tag_name"], "count": r["cnt"]} for r in deleted_tags]

            open_reviews = conn.execute(
                "SELECT id, doc_id, reason, updated_at FROM review_queue "
                "WHERE status = 'open' ORDER BY updated_at DESC LIMIT 20",
            ).fetchall()
            open_review_list = [dict(r) for r in open_reviews]

            errors = conn.execute(
                "SELECT error_text, COUNT(*) AS cnt FROM documents "
                "WHERE status = 'error' AND created_at >= ? AND created_at < ? "
                "AND error_text != '' "
                "GROUP BY error_text ORDER BY cnt DESC LIMIT 10",
                (start, end),
            ).fetchall()
            error_list = [{"error": r["error_text"], "count": r["cnt"]} for r in errors]

        return {
            "year": year,
            "month": month,
            "total_runs": total_runs,
            "total_docs": total_docs,
            "success_docs": success_docs,
            "success_rate": success_rate,
            "status_counts": status_counts,
            "new_correspondents": new_correspondents,
            "deleted_tags": deleted_tag_list,
            "open_reviews": open_review_list,
            "errors": error_list,
        }

    def get_processing_stats(self, days: int = 30) -> dict:
        """Aggregate processing statistics over the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM documents WHERE created_at >= ? GROUP BY status",
                (cutoff,),
            ).fetchall()
            status_counts = {r["status"]: r["cnt"] for r in rows}
            total = sum(status_counts.values())
            updated = sum(status_counts.get(s, 0) for s in ("updated", "ok", "applied"))
            errors = sum(v for k, v in status_counts.items() if k.startswith("error"))
            reviews = sum(status_counts.get(s, 0) for s in ("queued_review", "queued_review_update_error"))
            rule_based = status_counts.get("rule_based", 0)
            prior_fb = status_counts.get("prior_fallback", 0)
            run_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM runs WHERE started_at >= ?", (cutoff,),
            ).fetchone()
            total_runs = run_row["cnt"] if run_row else 0
            rev_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM review_queue WHERE status = 'open'",
            ).fetchone()
            open_reviews = rev_row["cnt"] if rev_row else 0
            top_errors = conn.execute(
                "SELECT error_text, COUNT(*) AS cnt FROM documents "
                "WHERE created_at >= ? AND error_text IS NOT NULL AND error_text != '' "
                "GROUP BY error_text ORDER BY cnt DESC LIMIT 5",
                (cutoff,),
            ).fetchall()
            top_error_list = [{"error": r["error_text"][:80], "count": r["cnt"]} for r in top_errors]
        return {
            "days": days,
            "total_docs": total,
            "updated": updated,
            "errors": errors,
            "reviews": reviews,
            "rule_based": rule_based,
            "prior_fallback": prior_fb,
            "success_rate": (updated / total * 100) if total > 0 else 0.0,
            "total_runs": total_runs,
            "open_reviews": open_reviews,
            "top_errors": top_error_list,
            "status_breakdown": dict(status_counts),
        }
