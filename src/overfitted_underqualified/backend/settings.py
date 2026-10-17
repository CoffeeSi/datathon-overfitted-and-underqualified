"""Environment-backed configuration for the local service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _boolean(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    data_dir: Path
    database_path: Path
    upload_dir: Path
    max_upload_bytes: int
    max_active_jobs: int
    analysis_timeout_seconds: int
    job_deadline_seconds: int
    lease_seconds: int
    retention_seconds: int
    frontend_origin: str | None
    mock_fixture_path: Path | None
    analyzer_factory: str | None
    allow_mock_results: bool

    def __post_init__(self) -> None:
        if self.analysis_timeout_seconds * self.max_active_jobs > self.job_deadline_seconds:
            raise ValueError(
                "analysis timeout multiplied by active job capacity must fit the job deadline"
            )

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("FORTEBANK_DATA_DIR", "data/service")).expanduser()
        configured_fixture = os.getenv("FORTEBANK_MOCK_FIXTURE_PATH")
        frontend_origin = os.getenv("FORTEBANK_FRONTEND_ORIGIN") or None
        return cls(
            api_key=os.getenv("FORTEBANK_API_KEY") or None,
            data_dir=data_dir,
            database_path=Path(
                os.getenv("FORTEBANK_DATABASE_PATH", str(data_dir / "jobs.sqlite3"))
            ).expanduser(),
            upload_dir=Path(
                os.getenv("FORTEBANK_UPLOAD_DIR", str(data_dir / "uploads"))
            ).expanduser(),
            max_upload_bytes=_positive_int("FORTEBANK_MAX_UPLOAD_MB", 100) * 1024 * 1024,
            # One processing job plus one queued job fits the initial 120 second estimate.
            # Replace this cap after collecting local model runtimes.
            max_active_jobs=_positive_int("FORTEBANK_MAX_ACTIVE_JOBS", 2),
            analysis_timeout_seconds=_positive_int("FORTEBANK_ANALYSIS_TIMEOUT_SECONDS", 120),
            job_deadline_seconds=_positive_int("FORTEBANK_JOB_DEADLINE_SECONDS", 300),
            lease_seconds=_positive_int("FORTEBANK_JOB_LEASE_SECONDS", 45),
            retention_seconds=_positive_int("FORTEBANK_RETENTION_HOURS", 24) * 60 * 60,
            frontend_origin=frontend_origin,
            mock_fixture_path=Path(configured_fixture).expanduser() if configured_fixture else None,
            analyzer_factory=os.getenv("FORTEBANK_ANALYZER_FACTORY") or None,
            allow_mock_results=_boolean("FORTEBANK_ENABLE_MOCK_RESULTS"),
        )
