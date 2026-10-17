"""Validate the sole annotation source without modifying it."""
import re
import numpy as np
import pandas as pd
from .video import clip_sort_key
LABELS = ("violation", "no_violation")

# Преобразование и проверка таймкодов.

def parse_timecode(value):
    if pd.isna(value) or str(value).strip() in {"", "—", "-"}:
        return np.nan
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        result = float(text)
        return result if np.isfinite(result) else np.nan
    parts = text.split(":")
    if len(parts) not in (2, 3):
        return np.nan
    if not all(re.fullmatch(r"\d+", part) for part in parts[:-1]):
        return np.nan
    if not re.fullmatch(r"\d+(?:\.\d+)?", parts[-1]):
        return np.nan
    if any(float(part) >= 60 for part in parts[1:]):
        return np.nan
    result = 0.0
    for part in parts:
        result = result * 60 + float(part)
    return result if np.isfinite(result) else np.nan

# Проверка соответствия файлов и исходной разметки.

def validate_dataset(events, inventory):
    """Validate source labels/files/times, without treating provisional notes as facts."""
    issues = []

    def add(clip_id, code, detail, severity="error"):
        issues.append({"clip_id": str(clip_id), "severity": severity, "code": code, "detail": str(detail)})

    for clip_id in events.loc[events["clip_id"].duplicated(False), "clip_id"].unique():
        add(clip_id, "duplicate_annotation_id", "Repeated clip_id in events.csv")
    for clip_id in inventory.loc[inventory["clip_id"].duplicated(False), "clip_id"].unique():
        add(clip_id, "duplicate_video_id", "Multiple video files match this clip_id")
    counts = inventory["clip_id"].value_counts()
    unique = inventory.drop_duplicates("clip_id").set_index("clip_id")
    for row in events.itertuples(index=False):
        clip_id = row.clip_id
        if pd.isna(clip_id) or not re.fullmatch(r"X-\d+", str(clip_id)):
            add(clip_id, "invalid_clip_id", "Expected normalized X-N identifier")
        label = None if pd.isna(row.source_label) else row.source_label
        if label not in LABELS:
            add(clip_id, "invalid_label", row.source_label)
        if counts.get(clip_id, 0) == 0:
            add(clip_id, "missing_video", "No matching video file")
        elif counts.get(clip_id, 0) == 1 and not unique.loc[clip_id, "readable"]:
            add(clip_id, "unreadable_video", unique.loc[clip_id, "error"])
        stamp = parse_timecode(row.timecode)
        supplied = pd.notna(row.timecode) and str(row.timecode).strip() not in {"", "—", "-"}
        if supplied and pd.isna(stamp):
            add(clip_id, "invalid_timecode", row.timecode)
        if label == "violation" and pd.isna(stamp):
            add(clip_id, "missing_event_time", "Violation requires a valid event time")
        if label == "no_violation" and supplied:
            add(clip_id, "normal_with_event_time", row.timecode, "warning")
        if counts.get(clip_id, 0) == 1 and pd.notna(stamp):
            duration = unique.loc[clip_id, "duration_sec"]
            if pd.notna(duration) and not 0 <= stamp < duration:
                add(clip_id, "time_out_of_bounds", f"{stamp}s / duration {duration}s")
        source_ru = getattr(row, "source_label_ru", None)
        expected = {"violation": "Нарушение", "no_violation": "Без нарушения"}.get(label)
        if pd.notna(source_ru) and source_ru != expected:
            add(clip_id, "inconsistent_label_translation", source_ru)
    for row in inventory.itertuples(index=False):
        if row.clip_id not in set(events["clip_id"]):
            add(row.clip_id, "unlabelled_video", row.video_rel, "warning")
        if pd.notna(row.duration_sec) and not 10 <= row.duration_sec <= 145:
            add(row.clip_id, "unexpected_duration", row.duration_sec, "warning")
        if getattr(row, "readable", False) and not getattr(row, "tail_read_ok", True):
            add(row.clip_id, "tail_decode_failed", "End frame cannot be decoded, including sequential fallback. Check the video.", "warning")
    for _, group in inventory.loc[inventory["sha256"].ne("")].groupby("sha256"):
        if len(group) > 1:
            ids = group["clip_id"].tolist()
            for clip_id in ids:
                add(clip_id, "exact_duplicate_grouped", f"Canonical video: {min(ids, key=clip_sort_key)}; aliases: {', '.join(ids)}", "info")
            labels = events.loc[events["clip_id"].isin(ids), "source_label"].dropna().unique()
            if len(labels) > 1:
                for clip_id in ids:
                    add(clip_id, "duplicate_label_conflict", ", ".join(ids))
    return pd.DataFrame(issues, columns=["clip_id", "severity", "code", "detail"])
