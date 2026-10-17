"""Annotation-free binary full-video scanning. Candidates have no action/legal type."""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Protocol

import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights

from .dataset import VideoInfo, decode_frames, sha256
from .train import load_checkpoint, seed_everything
from ..rules.catalog import event_by_type, source_links_for, supported_event_types


@dataclass(frozen=True)
class ScanConfig:
    window_sec: float = 4.0
    step_sec: float = 2.0
    batch_size: int = 2
    threshold: float = 0.5
    merge_gap_sec: float = 0.0
    frames: int = 16

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (self.window_sec, self.step_sec)):
            raise ValueError("Window and step must be positive and finite")
        if self.step_sec > self.window_sec:
            raise ValueError("step_sec must not exceed window_sec: no gaps in coverage")
        if type(self.batch_size) is not int or self.batch_size < 1 or self.frames != 16:
            raise ValueError("Use a positive integer batch_size and 16 frames")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if not math.isfinite(self.merge_gap_sec) or self.merge_gap_sec < 0:
            raise ValueError("merge_gap_sec must be finite and nonnegative")


def inspect_video(path: Path) -> VideoInfo:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if (not cap.isOpened() or not math.isfinite(fps) or fps <= 0
                or not math.isfinite(count) or count < 1):
            raise ValueError(f"Unreadable video metadata: {path}")
        return VideoInfo(path.stem, path, fps, int(count))
    finally:
        cap.release()


def sliding_windows(last_time: float, config: ScanConfig) -> list[dict]:
    """Cover [0, last frame] including a final inward-shifted window.

    Short videos get one window over their available span. Frame timestamps use
    the baseline's nominal FPS/frame-index convention, not a variable-FPS model.
    """
    if not math.isfinite(last_time) or last_time < 0:
        raise ValueError("Invalid last frame time")
    span = min(config.window_sec, last_time)
    final_start = last_time - span
    starts = [i * config.step_sec for i in range(math.floor(final_start / config.step_sec) + 1)]
    if final_start - starts[-1] > 1e-9:
        starts.append(final_start)
    return [dict(window_index=i, start_sec=start, end_sec=start + span,
                 center_sec=start + span / 2) for i, start in enumerate(starts)]


def merge_candidates(windows: list[dict], config: ScanConfig) -> list[dict]:
    """Union positive supports across <= gap seconds; peak ties choose earliest window.

    Overlapping positive supports may bridge a negative window. A long continuous
    alarm produces one candidate; it does not imply one true event in that span.
    """
    ordered = sorted(windows, key=lambda w: (w["start_sec"], w["window_index"]))
    groups = []
    for window in ordered:
        probability = window["probability"]
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid probability")
        if probability < config.threshold:
            continue
        if groups and window["start_sec"] <= groups[-1]["end_sec"] + config.merge_gap_sec + 1e-9:
            groups[-1]["windows"].append(window)
            groups[-1]["end_sec"] = max(groups[-1]["end_sec"], window["end_sec"])
        else:
            groups.append(dict(start_sec=window["start_sec"], end_sec=window["end_sec"], windows=[window]))
    candidates = []
    for i, group in enumerate(groups):
        peak = max(group["windows"], key=lambda w: w["probability"])
        candidates.append(dict(candidate_id=i, start_sec=group["start_sec"], end_sec=group["end_sec"],
                               timestamp_sec=peak["center_sec"], score=peak["probability"],
                               peak_window_index=peak["window_index"],
                               window_indices=[w["window_index"] for w in group["windows"]],
                               event_type=None, type_confidence=None, source_links=[]))
    return candidates


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class FullVideoScanner:
    """Bounded frame memory: one raw window plus at most batch_size transformed windows.

    Output metadata grows with the number of windows. This object never reads labels,
    annotations, split assignments or clip IDs to choose windows or predict types.
    """

    def __init__(self, model, transform, config: ScanConfig, device="cpu", *, provenance=None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.transform = transform
        self.config = config
        self.provenance = provenance

    def scan(self, video_path: Path) -> dict:
        synchronize(self.device)
        start = time.perf_counter()
        video = inspect_video(video_path)
        digest = sha256(video.path)
        windows = sliding_windows(video.last_time, self.config)
        self.model.eval()
        with torch.inference_mode():
            for offset in range(0, len(windows), self.config.batch_size):
                batch_windows = windows[offset:offset + self.config.batch_size]
                batch = []
                for window in batch_windows:
                    times = np.linspace(window["start_sec"], window["end_sec"], self.config.frames)
                    tensor = self.transform(decode_frames(video, times))
                    if tuple(tensor.shape) != (3, 16, 112, 112) or not torch.isfinite(tensor).all():
                        raise ValueError(f"Invalid preprocessed window: {video.path}, {window['window_index']}")
                    batch.append(tensor)
                logits = self.model(torch.stack(batch).to(self.device))
                if tuple(logits.shape) != (len(batch), 2) or not torch.isfinite(logits).all():
                    raise ValueError("Expected finite binary model logits")
                probabilities = logits.softmax(dim=1)[:, 1].cpu().tolist()
                for window, probability in zip(batch_windows, probabilities):
                    window.update(probability=probability, prediction=int(probability >= self.config.threshold))
        candidates = merge_candidates(windows, self.config)
        if sha256(video.path) != digest:
            raise ValueError("Video changed during scanning")
        synchronize(self.device)
        return dict(video_path=str(video.path), sha256=digest, fps=video.fps, frame_count=video.frame_count,
                    duration_sec=video.frame_count / video.fps, last_frame_sec=video.last_time,
                    scan_config=asdict(self.config), windows=windows, candidates=candidates,
                    prediction=int(bool(candidates)), probability=max(w["probability"] for w in windows),
                    processing_sec=time.perf_counter() - start)


def prepare_scanner(checkpoint_path: Path, config: ScanConfig, device="cpu") -> FullVideoScanner:
    path = Path(checkpoint_path).resolve()
    digest = sha256(path)
    model, checkpoint = load_checkpoint(path, device)
    saved = checkpoint["config"]
    if saved["weights"] != "KINETICS400_V1" or saved["frames"] != 16:
        raise ValueError("Unsupported checkpoint preprocessing")
    seed_everything(saved["seed"], saved["cpu_threads"])
    if sha256(path) != digest:
        raise ValueError("Checkpoint changed during loading")
    provenance = dict(checkpoint_path=str(path), checkpoint_sha256=digest, epoch=checkpoint["epoch"],
                      model_config=saved, training_manifest=checkpoint["data_manifest"])
    return FullVideoScanner(model, R3D_18_Weights[saved["weights"]].transforms(), config, device,
                            provenance=provenance)


def evidence_frames(video_path: Path, timestamp_sec: float, *, count=3, offset_sec=0.5) -> list[dict]:
    """Local-only JPEG context, with sequentially decoded indices and decoder PTS.

    Return backend-compatible fields plus sampling provenance. Frame indices are
    selected using nominal FPS; timestamp_sec records actual OpenCV POS_MSEC.
    Never silently substitute a requested timestamp when decoder timing is invalid.
    """
    video = inspect_video(video_path)
    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError("Use 1-3 evidence frames")
    if not math.isfinite(timestamp_sec) or not 0 <= timestamp_sec <= video.last_time:
        raise ValueError("Evidence timestamp outside video")
    if not math.isfinite(offset_sec) or offset_sec < 0:
        raise ValueError("Invalid evidence offset")
    offsets = [0] if count == 1 else np.linspace(-offset_sec, offset_sec, count)
    indices = sorted(set(int(round(min(max(timestamp_sec + delta, 0), video.last_time) * video.fps))
                         for delta in offsets))
    frames = []
    cap = cv2.VideoCapture(str(video.path))
    try:
        for index in range(max(indices) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode evidence frame {index}: {video.path}")
            if index not in indices:
                continue
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(pts) or pts < 0 or (index > 0 and pts <= 0):
                raise ValueError(f"Invalid evidence decoder PTS at frame {index}")
            if frames and pts <= frames[-1]["timestamp_sec"]:
                raise ValueError("Non-increasing evidence decoder PTS")
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("JPEG encoding failed")
            frames.append(dict(timestamp_sec=pts, image_base64=base64.b64encode(encoded).decode("ascii"),
                               frame_index=index, nominal_timestamp_sec=index / video.fps,
                               timestamp_basis="opencv_decoder_pts"))
    finally:
        cap.release()
    return frames


@dataclass(frozen=True)
class ActionObservation:
    """Output of a separate, evaluated action recognizer; not a binary scan score."""
    event_type: str
    confidence: float
    component: str
    visible_description: str


class ActionRecognizer(Protocol):
    def recognize(self, video_path: Path, candidate: dict) -> ActionObservation | None: ...


def link_action(observation: ActionObservation) -> dict:
    """Enrich an externally determined catalog type, retaining unobservable conditions."""
    if observation.event_type not in supported_event_types():
        raise ValueError("Unsupported catalog action type")
    if not math.isfinite(observation.confidence) or not 0 <= observation.confidence <= 1:
        raise ValueError("Invalid type confidence")
    if not observation.component.strip() or not observation.visible_description.strip():
        raise ValueError("Action recognizer provenance and visible description are required")
    event = event_by_type(observation.event_type)
    return dict(**asdict(observation), source_links=source_links_for(observation.event_type),
                applicability=list(event["applicability"]),
                video_cannot_establish=list(event.get("video_cannot_establish", [])),
                legal_conclusion=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    import yaml
    config = ScanConfig(**yaml.safe_load(args.config.read_text(encoding="utf-8"))["scan"]) if args.config else ScanConfig()
    if args.output.exists():
        raise FileExistsError(args.output)
    scanner = prepare_scanner(args.checkpoint, config, args.device)
    result = scanner.scan(args.video)
    result["provenance"] = scanner.provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(args.output)


if __name__ == "__main__":
    main()
