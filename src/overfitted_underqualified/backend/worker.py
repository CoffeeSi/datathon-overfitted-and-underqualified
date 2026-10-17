"""Background job worker with SQLite leases and restart recovery."""

from __future__ import annotations

import asyncio
import logging

from .analyzer import Analyzer
from .settings import Settings
from .storage import Job, JobStore, current_time

logger = logging.getLogger(__name__)


async def _heartbeat(store: JobStore, job: Job, settings: Settings, stop: asyncio.Event) -> None:
    interval = max(1, settings.lease_seconds // 3)
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            renewed = store.extend_lease(
                job.job_id,
                job.attempts,
                now=current_time(),
                lease_seconds=settings.lease_seconds,
            )
            if not renewed:
                return


async def run_worker(
    store: JobStore,
    analyzer: Analyzer,
    settings: Settings,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        now = current_time()
        store.recover_expired_leases(now=now, retention_seconds=settings.retention_seconds)
        job = store.claim_next(
            now=now,
            lease_seconds=settings.lease_seconds,
            retention_seconds=settings.retention_seconds,
        )
        if job is None:
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.5)
            except TimeoutError:
                pass
            continue

        if analyzer.name == "mock" and not settings.allow_mock_results:
            store.fail(
                job.job_id,
                job.attempts,
                "analyzer_not_configured",
                "A real local video analysis provider is not configured.",
                now=current_time(),
                retention_seconds=settings.retention_seconds,
            )
            continue

        heartbeat_stop = asyncio.Event()
        heartbeat_task = asyncio.create_task(_heartbeat(store, job, settings, heartbeat_stop))
        try:
            remaining = max(0.01, job.deadline_at - current_time())
            timeout = min(settings.analysis_timeout_seconds, remaining)
            result = await asyncio.wait_for(analyzer.analyze(job.file_path), timeout=timeout)
            store.complete(
                job.job_id,
                job.attempts,
                result,
                now=current_time(),
                retention_seconds=settings.retention_seconds,
            )
        except TimeoutError:
            store.fail(
                job.job_id,
                job.attempts,
                "analysis_timeout",
                "The analysis exceeded its configured time limit.",
                now=current_time(),
                retention_seconds=settings.retention_seconds,
            )
        except Exception:
            logger.exception("Analysis failed for job %s", job.job_id)
            store.fail(
                job.job_id,
                job.attempts,
                "analysis_failed",
                "The video could not be analyzed.",
                now=current_time(),
                retention_seconds=settings.retention_seconds,
            )
        finally:
            heartbeat_stop.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass


async def run_retention_cleanup(
    store: JobStore,
    retention_seconds: int,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        for path in store.delete_expired(now=current_time()):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Could not remove expired upload %s", path)
        try:
            await asyncio.wait_for(stop.wait(), timeout=min(60, retention_seconds))
        except TimeoutError:
            pass
