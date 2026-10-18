"""Point annotations -> video windows, with split/hash validation before decoding."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LABELS = {"no_violation": 0, "violation": 1}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_fields_digest(rows, fields):
    """Canonical digest of explicitly preserved source columns, including row order."""
    payload = json.dumps(dict(fields=fields, rows=[{k: r[k] for k in fields} for r in rows]),
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_annotation_revision(path, expected_sha256, *, registry_path=None):
    """Binary protocol: accept exact input or an explicitly registered metadata edit.

    Checkpoint hashes remain historical. A registry entry explicitly approves a
    particular pair of full-file hashes, plus all protected source values.
    """
    path = Path(path)
    actual = sha256(path)
    if actual == expected_sha256:
        return None
    registry = Path(registry_path) if registry_path else Path(__file__).resolve().parents[3]/"data/annotation_revisions.json"
    entries = json.loads(registry.read_text(encoding="utf-8")) if registry.is_file() else []
    for entry in entries:
        if entry.get("before_sha256") != expected_sha256 or entry.get("after_sha256") != actual:
            continue
        allowed = entry.get("allowed_changed_fields")
        legal_only = entry.get("schema") == "legal_basis_only_revision_v1" and allowed == ["legal_basis"]
        description_only = (entry.get("schema") == "binary_metadata_revision_v1"
                            and allowed in (["observed_event"], ["observed_event", "legal_basis"]))
        action_review_only = (entry.get("schema") == "binary_action_review_revision_v1"
                              and isinstance(allowed, list) and "action_review" in allowed
                              and set(allowed) <= {"action_review", "observed_event", "legal_basis"})
        primitive_review_only = (entry.get("schema") == "binary_primitive_review_revision_v1"
                                 and isinstance(allowed, list) and "primitive_review" in allowed
                                 and set(allowed) <= {"primitive_review", "action_review", "observed_event", "legal_basis"})
        if not (legal_only or description_only or action_review_only or primitive_review_only):
            raise ValueError("Invalid annotation revision scope")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        protected = [k for k in fields if k not in allowed]
        if (protected != entry["preserved_fields"] or annotation_fields_digest(rows, protected) != entry["preserved_fields_sha256"]):
            raise ValueError("Annotation revision changed protected fields")
        return entry
    raise ValueError("events hash differs from checkpoint provenance; no approved metadata revision")


def parse_time(value: str) -> float | None:
    if not value.strip() or value.strip() in {"—", "-"}:
        return None
    parts = value.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid event time: {value}") from exc
    if len(numbers) > 3 or any(not math.isfinite(x) or x < 0 for x in numbers):
        raise ValueError(f"Invalid event time: {value}")
    if len(numbers) > 1 and any(x >= 60 for x in numbers[1:]):
        raise ValueError(f"Invalid event time: {value}")
    return sum(x * 60 ** i for i, x in enumerate(reversed(numbers)))


@dataclass(frozen=True)
class VideoInfo:
    """Decode metadata only; no supervision is required for an unseen video."""

    clip_id: str
    path: Path
    fps: float
    frame_count: int

    @property
    def last_time(self) -> float:
        return (self.frame_count - 1) / self.fps


@dataclass(frozen=True)
class VideoRecord:
    clip_id: str
    path: Path
    split: str
    label: int
    sha256: str
    aliases: tuple[str, ...]
    event_times: tuple[float, ...]
    fps: float
    frame_count: int

    @property
    def last_time(self) -> float:
        return (self.frame_count - 1) / self.fps


def load_records(events_path: Path, split_path: Path, video_dir: Path) -> list[VideoRecord]:
    """Use labels/times from events.csv, checking the split snapshot and real hashes."""
    with events_path.open(encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    events = {row["clip_id"]: row for row in source}
    if not source or len(events) != len(source):
        raise ValueError("events.csv must have unique, nonempty source records")
    with split_path.open(encoding="utf-8-sig", newline="") as stream:
        assignments = list(csv.DictReader(stream))
    if not assignments or len({r["clip_id"] for r in assignments}) != len(assignments):
        raise ValueError("Split must contain unique canonical video IDs")
    paths: dict[str, Path] = {}
    for path in sorted(video_dir.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        match = re.fullmatch(r"[XХ]-(\d+)", path.stem, re.IGNORECASE)
        if match:
            clip_id = f"X-{int(match[1])}"
            if clip_id in paths:
                raise ValueError(f"Ambiguous video ID: {clip_id}")
            paths[clip_id] = path
    covered: set[str] = set()
    hashes: set[str] = set()
    records = []
    for row in assignments:
        clip_id = row["clip_id"]
        aliases = tuple(json.loads(row["source_clip_ids"]))
        if (not aliases or clip_id not in aliases or len(set(aliases)) != len(aliases)
                or covered.intersection(aliases) or not set(aliases) <= events.keys()):
            raise ValueError(f"Invalid or overlapping source aliases: {clip_id}")
        covered.update(aliases)
        if row["split"] not in {"train", "validation"}:
            raise ValueError(f"Unknown split: {row['split']}")
        labels = {events[alias]["source_label"] for alias in aliases}
        if len(labels) != 1 or row["source_label"] not in labels or not labels <= LABELS.keys():
            raise ValueError(f"Conflicting/stale labels for {clip_id}")
        label = LABELS[row["source_label"]]
        times = []
        for alias in aliases:
            stamp = parse_time(events[alias]["timecode"])
            if (label == 1 and stamp is None) or (label == 0 and stamp is not None):
                raise ValueError(f"Label/time mismatch: {alias}")
            if stamp is not None:
                times.append(stamp)
        times = tuple(sorted(set(times)))
        if tuple(sorted(json.loads(row["event_times_sec"]))) != times:
            raise ValueError(f"Stale event times in split for {clip_id}; rerun EDA")
        expected_hash = row["sha256"]
        if expected_hash in hashes or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError("Duplicate/invalid SHA-256 in canonical split; possible leakage")
        hashes.add(expected_hash)
        if clip_id not in paths:
            raise FileNotFoundError(f"Missing canonical video: {clip_id} in {video_dir}")
        for alias in aliases:
            if alias in paths and sha256(paths[alias]) != expected_hash:
                raise ValueError(f"Video hash differs from split: {alias}")
        cap = cv2.VideoCapture(str(paths[clip_id]))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count < 1:
                raise ValueError(f"Unreadable video metadata: {clip_id}")
        finally:
            cap.release()
        if any(t >= count / fps for t in times):
            raise ValueError(f"Event time outside video: {clip_id}")
        records.append(VideoRecord(clip_id, paths[clip_id], row["split"], label,
                                   expected_hash, aliases, times, fps, count))
    if covered != events.keys():
        raise ValueError("Split does not cover all events.csv source IDs")
    for subset in ("train", "validation"):
        if {r.label for r in records if r.split == subset} != {0, 1}:
            raise ValueError(f"Both classes are required in {subset}")
    return sorted(records, key=lambda r: int(r.clip_id.split("-")[1]))


def sample_times(center: float, last_time: float, window_sec: float, frames: int = 16) -> np.ndarray:
    if frames != 16 or window_sec <= 0 or not math.isfinite(window_sec):
        raise ValueError("R3D baseline requires 16 frames and a positive finite window")
    if not math.isfinite(center) or not math.isfinite(last_time) or last_time < 0:
        raise ValueError("Invalid center or video length")
    span = min(window_sec, last_time)
    start = min(max(center - span / 2, 0), last_time - span)
    return np.linspace(start, start + span, frames)


def decode_frames(record: VideoRecord | VideoInfo, times: np.ndarray) -> torch.Tensor:
    """Return uint8 RGB (T,C,H,W); recover broken MOV seeks by sequential decoding."""
    indices = np.clip(np.rint(times * record.fps).astype(int), 0, record.frame_count - 1)
    wanted = set(indices.tolist())
    decoded = {}
    cap = cv2.VideoCapture(str(record.path))
    try:
        for index in sorted(wanted):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                break
            ok, frame = cap.read()
            if not ok:
                break
            decoded[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    if len(decoded) != len(wanted):
        decoded = {}
        cap = cv2.VideoCapture(str(record.path))
        try:
            for index in range(max(wanted) + 1):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode frame {index}: {record.clip_id}")
                if index in wanted:
                    decoded[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
    return torch.from_numpy(np.stack([decoded[i] for i in indices])).permute(0, 3, 1, 2)


class VideoWindowDataset(Dataset):
    """Fixed annotation-centred positives and evenly spaced negative windows.

    Validation here measures annotation-centred window classification, not blind
    event localization. No unlabeled portions of positive videos become negatives.
    """

    def __init__(self, records: list[VideoRecord], split: str, transform,
                 window_sec: float = 4.0, frames: int = 16, negative_windows: int = 1,
                 *, negative_sampling="interior"):
        if split not in {"train", "validation"} or negative_windows < 1:
            raise ValueError("Invalid subset or negative window count")
        sample_times(0, 1, window_sec, frames)
        self.transform = transform
        self.window_sec = window_sec
        self.frames = frames
        self.samples = []
        if negative_sampling not in {"interior", "coverage"}:
            raise ValueError("Unknown negative sampling policy")
        for record in records:
            if record.split != split:
                continue
            centers = record.event_times if record.label else tuple(
                record.last_time * (i + 1) / (negative_windows + 1) for i in range(negative_windows))
            if not record.label and negative_sampling == "coverage" and negative_windows > 1:
                span = min(window_sec, record.last_time)
                centers = tuple(float(t) for t in np.unique(np.linspace(span / 2, record.last_time - span / 2, negative_windows)))
            for center in centers:
                self.samples.append((record, center))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record, center = self.samples[index]
        stamps = sample_times(center, record.last_time, self.window_sec, self.frames)
        video = self.transform(decode_frames(record, stamps))
        if video.shape != (3, 16, 112, 112) or not torch.isfinite(video).all():
            raise ValueError(f"Invalid preprocessed input for {record.clip_id}: {video.shape}")
        return {"video": video, "label": torch.tensor(record.label, dtype=torch.long),
                "clip_id": record.clip_id, "center_sec": center}
