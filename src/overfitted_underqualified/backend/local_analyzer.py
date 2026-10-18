"""Local typed provider. Enable explicitly only after a reviewed action experiment."""
import asyncio
import logging
import os
from pathlib import Path
import threading

import yaml

from .analyzer import UnconfiguredAnalyzer
from .schemas import TypedAnalysisResult, TypedFinding, UnresolvedCandidate

logger = logging.getLogger(__name__)


class LocalActionAnalyzer:
    name = "local-r3d18-actions"

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._lock = asyncio.Lock()

    def _analyze_sync(self, video_path, cancelled):
        from ..ml.inference import check_cancelled, evidence_frames
        result = self.pipeline.scan(video_path, cancel_event=cancelled)
        findings = []
        for event in result["typed_events"]:
            check_cancelled(cancelled)
            action = event["action"]
            findings.append(TypedFinding(
                type=event["event_type"], timecode_sec=event["timestamp_sec"],
                description=action["visible_description"], binary_score=event["binary_score"],
                type_confidence=event["type_confidence"], source_links=action["source_links"],
                applicability=action["applicability"], video_cannot_establish=action["video_cannot_establish"],
                evidence_frames=evidence_frames(video_path, event["timestamp_sec"], cancel_event=cancelled)))
        return TypedAnalysisResult(verdict=result["verdict"], violations=findings,
                                   unresolved_candidates=[UnresolvedCandidate(**c) for c in result["unresolved_candidates"]],
                                   recognition_scope=sorted(self.pipeline.recognizer.enabled_types))

    async def analyze(self, video_path):
        async with self._lock:
            cancelled = threading.Event()
            def outcome():
                # Carry exceptions as data until the owner resumes. In particular a
                # cancelled shield must not report an unhandled thread exception.
                try:
                    return self._analyze_sync(video_path, cancelled), None
                except Exception as error:
                    return None, error
            task = asyncio.create_task(asyncio.to_thread(outcome))
            try:
                result, error = await asyncio.shield(task)
                if error is not None:
                    raise error
                return result
            except asyncio.CancelledError:
                cancelled.set()
                # wait_for cancellation must drain the actual computation. Keep the
                # lock, job lease heartbeat and admission slot until the thread exits.
                # Repeated shutdown cancellations must not release them prematurely.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if task.done() and not task.cancelled():
                    task.exception()  # retrieve a cooperative InterruptedError
                raise


def create_analyzer(_settings):
    """Missing/invalid weights leave readiness false; never substitute a mock."""
    try:
        from ..ml.action_inference import prepare_typed_pipeline
        from ..ml.inference import ScanConfig
        from ..ml.train import ROOT
        config_path = Path(os.environ.get("FORTEBANK_ACTION_CONFIG", ROOT / "configs/action_service.yaml"))
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        pipeline = prepare_typed_pipeline(ROOT / config["binary_checkpoint"], ROOT / config["action_checkpoint"],
                                          enabled_types=config["enabled_types"], device=config["device"],
                                          scan=ScanConfig(**config["scan"]))
        return LocalActionAnalyzer(pipeline)
    except Exception:
        # Includes malformed YAML, corrupt weights and unsupported runtime/device.
        # Preserve diagnostics server-side while allowing /readyz to report 503.
        logger.exception("Typed local analyzer unavailable; readiness remains false")
        return UnconfiguredAnalyzer()
