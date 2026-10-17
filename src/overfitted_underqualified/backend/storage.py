"""SQLite job storage with atomic admission, claims, leases, and retention."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .schemas import AnalysisResult


@dataclass(frozen=True)
class Job:
    job_id: str
    file_path: Path
    attempts: int
    deadline_at: float


class JobStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'done', 'failed')),
                    file_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    deadline_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    lease_expires_at REAL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    verdict TEXT,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    analysis_provider TEXT NOT NULL,
                    expires_at REAL
                );
                CREATE INDEX IF NOT EXISTS jobs_status_created_idx
                    ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS jobs_lease_idx
                    ON jobs(status, lease_expires_at);
                CREATE INDEX IF NOT EXISTS jobs_expiry_idx
                    ON jobs(expires_at);
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def admit_job(
        self,
        *,
        job_id: str,
        file_path: Path,
        now: float,
        max_active_jobs: int,
        job_deadline_seconds: int,
        analysis_provider: str,
    ) -> bool:
        with self._transaction() as connection:
            active_count = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'processing')"
            ).fetchone()[0]
            if active_count >= max_active_jobs:
                return False
            connection.execute(
                """INSERT INTO jobs (
                    job_id, status, file_path, created_at, updated_at, deadline_at,
                    analysis_provider
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    str(file_path),
                    now,
                    now,
                    now + job_deadline_seconds,
                    analysis_provider,
                ),
            )
        return True

    def claim_next(
        self, *, now: float, lease_seconds: int, retention_seconds: int
    ) -> Job | None:
        with self._transaction() as connection:
            overdue = connection.execute(
                "SELECT job_id FROM jobs WHERE status = 'queued' AND deadline_at <= ?",
                (now,),
            ).fetchall()
            for row in overdue:
                connection.execute(
                    """UPDATE jobs SET status = 'failed', updated_at = ?, completed_at = ?,
                       error_code = 'processing_deadline_exceeded',
                       error_message = 'The analysis could not finish before its deadline.',
                       expires_at = ? WHERE job_id = ? AND status = 'queued'""",
                    (now, now, now + retention_seconds, row["job_id"]),
                )
            row = connection.execute(
                """SELECT job_id, file_path, attempts, deadline_at FROM jobs
                   WHERE status = 'queued' ORDER BY created_at, job_id LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            attempts = row["attempts"] + 1
            connection.execute(
                """UPDATE jobs SET status = 'processing', attempts = ?, started_at = ?,
                   updated_at = ?, lease_expires_at = ?
                   WHERE job_id = ? AND status = 'queued'""",
                (attempts, now, now, now + lease_seconds, row["job_id"]),
            )
            return Job(
                job_id=row["job_id"],
                file_path=Path(row["file_path"]),
                attempts=attempts,
                deadline_at=row["deadline_at"],
            )

    def extend_lease(self, job_id: str, attempts: int, *, now: float, lease_seconds: int) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET lease_expires_at = ?, updated_at = ?
                   WHERE job_id = ? AND status = 'processing' AND attempts = ?""",
                (now + lease_seconds, now, job_id, attempts),
            )
            return cursor.rowcount == 1

    def recover_expired_leases(
        self, *, now: float, retention_seconds: int, max_attempts: int = 2
    ) -> None:
        with self._transaction() as connection:
            expired = connection.execute(
                """SELECT job_id, attempts, deadline_at FROM jobs
                   WHERE status = 'processing' AND lease_expires_at <= ?""",
                (now,),
            ).fetchall()
            for row in expired:
                can_retry = row["attempts"] < max_attempts and row["deadline_at"] > now
                if can_retry:
                    connection.execute(
                        """UPDATE jobs SET status = 'queued', updated_at = ?,
                           lease_expires_at = NULL WHERE job_id = ? AND status = 'processing'""",
                        (now, row["job_id"]),
                    )
                else:
                    connection.execute(
                        """UPDATE jobs SET status = 'failed', updated_at = ?, completed_at = ?,
                           lease_expires_at = NULL, error_code = 'worker_interrupted',
                           error_message = 'The analysis worker stopped before completing the job.',
                           expires_at = ? WHERE job_id = ? AND status = 'processing'""",
                        (now, now, now + retention_seconds, row["job_id"]),
                    )

    def complete(
        self,
        job_id: str,
        attempts: int,
        result: AnalysisResult,
        *,
        now: float,
        retention_seconds: int,
    ) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET status = 'done', verdict = ?, result_json = ?,
                   updated_at = ?, completed_at = ?, lease_expires_at = NULL, expires_at = ?
                   WHERE job_id = ? AND status = 'processing' AND attempts = ?""",
                (
                    result.verdict,
                    result.model_dump_json(),
                    now,
                    now,
                    now + retention_seconds,
                    job_id,
                    attempts,
                ),
            )
            return cursor.rowcount == 1

    def fail(
        self,
        job_id: str,
        attempts: int,
        error_code: str,
        error_message: str,
        *,
        now: float,
        retention_seconds: int,
    ) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET status = 'failed', updated_at = ?, completed_at = ?,
                   lease_expires_at = NULL, error_code = ?, error_message = ?, expires_at = ?
                   WHERE job_id = ? AND status = 'processing' AND attempts = ?""",
                (now, now, error_code, error_message, now + retention_seconds, job_id, attempts),
            )
            return cursor.rowcount == 1

    def get_job(self, job_id: str, *, now: float) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None or (row["expires_at"] is not None and row["expires_at"] <= now):
            return None
        result: dict[str, object] = dict(row)
        if result["result_json"]:
            result["result"] = json.loads(result["result_json"])
        else:
            result["result"] = None
        return result

    def delete_expired(self, *, now: float) -> list[Path]:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT file_path FROM jobs WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            ).fetchall()
            connection.execute(
                "DELETE FROM jobs WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,)
            )
        return [Path(row["file_path"]) for row in rows]


def current_time() -> float:
    return time.time()
