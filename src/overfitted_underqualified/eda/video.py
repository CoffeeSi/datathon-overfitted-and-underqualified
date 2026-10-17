"""Container checks and timestamp-based decoding with sequential PTS fallback."""
import hashlib
import re
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}

# Сопоставление имён и поиск видео.

def clip_id_from_name(name):
    stem = Path(name).stem.translate(str.maketrans({"Х": "X", "х": "x"}))
    match = re.search(r"(?<![A-Za-z0-9])X-(\d+)(?!\d)", stem, re.IGNORECASE)
    return f"X-{int(match[1])}" if match else stem


def clip_sort_key(value):
    match = re.fullmatch(r"X-(\d+)", str(value))
    return (0, int(match[1])) if match else (1, str(value))


def collect_videos(folder):
    return sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
                  key=lambda p: (clip_sort_key(clip_id_from_name(p.name)), p.as_posix()))

# Чтение кадров с резервным последовательным декодированием; используется и при визуальном просмотре.

def read_frames(path, timestamps):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        for stamp in timestamps:
            if not cap.isOpened():
                frames.append(None)
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, float(stamp) * 1000)
            ok, frame = cap.read()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok and frame is not None else None)
    finally:
        cap.release()
    # Some MOV decoders seek inaccurately near EOF despite a complete stream.
    # Retry failed requests by decoding sequentially and matching actual PTS.
    missing = [i for i, frame in enumerate(frames) if frame is None]
    if missing:
        stream = cv2.VideoCapture(str(path))
        fps = float(stream.get(cv2.CAP_PROP_FPS))
        tolerance = 0.5 / fps + 1e-6 if np.isfinite(fps) and fps > 0 else 0.02
        best = {i: (float("inf"), None) for i in missing}
        try:
            while stream.isOpened():
                ok, frame = stream.read()
                if not ok or frame is None:
                    break
                pts = float(stream.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                for i in missing:
                    distance = abs(pts - float(timestamps[i]))
                    if np.isfinite(pts) and distance < best[i][0]:
                        best[i] = (distance, frame.copy())
        finally:
            stream.release()
        for i, (distance, frame) in best.items():
            if frame is not None and distance <= tolerance:
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frames

# Хеш файла, метаданные и проверка чтения кадров.

def inspect_video(path, directory):
    record = {"video": path.name, "video_rel": path.relative_to(directory).as_posix(),
                  "clip_id": clip_id_from_name(path.name), "fps": np.nan, "frames": 0,
                  "duration_sec": np.nan, "width": 0, "height": 0, "readable": False,
                  "sampled_frames_ok": 0, "sampled_frames_expected": 0, "metadata_valid": False,
                  "sha256": "", "error": "", "brightness": np.nan, "blur_score": np.nan,
                  "tail_read_ok": False}
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        record["sha256"] = digest.hexdigest()
    except OSError as exc:
        record["error"] = str(exc)
        return record, []
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            record["error"] = "container_not_opened"
            return record, []
        fps, count, width, height = [float(cap.get(prop)) for prop in (
            cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
        valid = all(np.isfinite(v) and v > 0 for v in (fps, count, width, height))
        record.update(fps=fps, frames=int(count) if np.isfinite(count) else 0,
                      width=int(width) if np.isfinite(width) else 0,
                      height=int(height) if np.isfinite(height) else 0, metadata_valid=valid)
        if not valid:
            record["error"] = "invalid_metadata"
            return record, []
        record["duration_sec"] = count / fps
    finally:
        cap.release()
    # Check body samples and the last frame; read_frames retries failed seeks
    # by sequential decoding using actual presentation timestamps.
    stamps = np.linspace(0, max(0, min((count - 1) / fps, count / fps * 0.98)), 5)
    frames = read_frames(path, stamps)
    record["tail_read_ok"] = read_frames(path, [max(0, (count - 1) / fps)])[0] is not None
    good = [f for f in frames if f is not None]
    record.update(sampled_frames_ok=len(good), sampled_frames_expected=len(stamps),
                  readable=len(good) == len(stamps))
    if len(good) != len(stamps):
        record["error"] = "sample_decode_failed"
    if good:
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in good]
        record["brightness"] = float(np.mean([f.mean() for f in gray]))
        record["blur_score"] = float(np.mean([cv2.Laplacian(f, cv2.CV_64F).var() for f in gray]))
    thumbnails = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (32, 32)) for f in good]
    return record, thumbnails
