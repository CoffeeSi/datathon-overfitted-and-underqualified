"""Authenticated asynchronous HTTP API for local video review jobs."""

from __future__ import annotations

import asyncio
import hmac
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile

from .analyzer import Analyzer
from .middleware import UploadLimitMiddleware
from .settings import Settings
from .schemas import AnalysisResult
from .storage import JobStore, current_time
from .video import has_valid_mov_container
from .worker import run_retention_cleanup, run_worker

logger = logging.getLogger(__name__)


def _require_api_key(settings: Settings):
    async def authenticate(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key is None:
            raise HTTPException(status_code=503, detail="API key is not configured")
        if x_api_key is None or not hmac.compare_digest(
            x_api_key.encode("utf-8"), settings.api_key.encode("utf-8")
        ):
            raise HTTPException(status_code=401, detail="Invalid API key")

    return authenticate


def _jury_job_response(job: dict[str, object]) -> dict[str, object]:
    response: dict[str, object] = {
        "job_id": job["job_id"],
        "status": _public_job_status(job),
    }
    if job["status"] == "done":
        result = job["result"]
        assert isinstance(result, dict)
        response.update(AnalysisResult.model_validate(result).model_dump())
    return response


def _public_job_status(job: dict[str, object]) -> str:
    # Keep the internal queue state out of the API contract. Waiting and actively
    # running jobs are both reported as processing to clients.
    status = job["status"]
    return "processing" if status == "queued" else str(status)


def create_app(
    settings: Settings | None = None, analyzer: Analyzer | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    store = JobStore(settings.database_path)
    if analyzer is None:
        from .analyzer import configured_analyzer

        analyzer = configured_analyzer(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        store.initialize()
        store.recover_expired_leases(
            now=current_time(), retention_seconds=settings.retention_seconds
        )
        stop = asyncio.Event()
        worker_task = asyncio.create_task(run_worker(store, analyzer, settings, stop))
        cleanup_task = asyncio.create_task(
            run_retention_cleanup(store, settings.retention_seconds, stop)
        )
        try:
            yield
        finally:
            stop.set()
            worker_task.cancel()
            cleanup_task.cancel()
            for task in (worker_task, cleanup_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(
        title="ForteBank Video Review API", version="0.1.0", lifespan=lifespan
    )
    app.state.settings = settings
    app.state.store = store
    app.state.analyzer = analyzer
    analysis_ready = analyzer.name != "unconfigured" and not (
        analyzer.name == "mock" and not settings.allow_mock_results
    )
    app.state.analysis_ready = analysis_ready
    authenticate = _require_api_key(settings)
    app.add_middleware(
        UploadLimitMiddleware, max_upload_bytes=settings.max_upload_bytes
    )

    if settings.frontend_origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[settings.frontend_origin],
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readiness() -> dict[str, str]:
        if not analysis_ready:
            raise HTTPException(
                status_code=503,
                detail="A local video analysis provider is not configured",
            )
        return {"status": "ready", "analysis_provider": analyzer.name}

    @app.post("/api/v1/analyze", status_code=202)
    async def analyze(
        request: Request,
        _: None = Depends(authenticate),
    ) -> dict[str, str]:
        if not analysis_ready:
            raise HTTPException(
                status_code=503,
                detail="A local video analysis provider is not configured",
            )
        # Parse multipart only after the API-key dependency has run.
        form = await request.form(max_files=1, max_fields=1)
        file = form.get("file")
        if (
            set(form.keys()) != {"file"}
            or len(form.getlist("file")) != 1
            or not isinstance(file, StarletteUploadFile)
        ):
            await form.close()
            raise HTTPException(
                status_code=422, detail="A video file is required in the 'file' field"
            )
        if not file.filename or Path(file.filename).suffix.lower() != ".mov":
            await form.close()
            raise HTTPException(
                status_code=415, detail="Only .mov uploads are supported"
            )

        job_id = str(uuid.uuid4())
        saved_path = settings.upload_dir / f"{job_id}.mov"
        header = bytearray()
        total_bytes = 0
        accepted = False
        try:
            with saved_path.open("xb") as destination:
                while chunk := await file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Upload exceeds the configured size limit",
                        )
                    if len(header) < 12:
                        header.extend(chunk[: 12 - len(header)])
                    destination.write(chunk)

            if total_bytes < 12 or bytes(header[4:8]) != b"ftyp":
                raise HTTPException(
                    status_code=422, detail="The uploaded .mov file is unreadable"
                )
            if not has_valid_mov_container(saved_path):
                raise HTTPException(
                    status_code=422, detail="The uploaded .mov file is unreadable"
                )

            accepted = store.admit_job(
                job_id=job_id,
                file_path=saved_path,
                now=current_time(),
                max_active_jobs=settings.max_active_jobs,
                job_deadline_seconds=settings.job_deadline_seconds,
                analysis_provider=analyzer.name,
            )
            if not accepted:
                raise HTTPException(
                    status_code=503, detail="The analysis queue is full; retry shortly"
                )
            return {"job_id": job_id}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Could not accept video upload")
            raise HTTPException(
                status_code=500, detail="The upload could not be accepted"
            ) from exc
        finally:
            await form.close()
            if not accepted:
                saved_path.unlink(missing_ok=True)

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(authenticate)])
    async def get_job(job_id: str) -> dict[str, object]:
        job = store.get_job(job_id, now=current_time())
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _jury_job_response(job)

    @app.get("/api/v1/ui/jobs/{job_id}", dependencies=[Depends(authenticate)])
    async def get_ui_job(job_id: str) -> dict[str, object]:
        job = store.get_job(job_id, now=current_time())
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        response: dict[str, object] = {
            "job_id": job["job_id"],
            "status": _public_job_status(job),
            "analysis_provider": job["analysis_provider"],
        }
        if job["status"] == "done":
            result = job["result"]
            assert isinstance(result, dict)
            response.update(result)
        elif job["status"] == "failed":
            response["error"] = {
                "code": job["error_code"],
                "message": job["error_message"],
            }
        return response

    return app


app = create_app()
