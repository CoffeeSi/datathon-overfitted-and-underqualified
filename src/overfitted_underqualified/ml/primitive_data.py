"""Sparse visible facts in events.csv; unknown targets are never inferred from violations."""
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import decode_frames, load_records, sha256
from .preprocessing import video_transform

SCHEMA = "visible_primitives_v1"
CLASSES = ("client_at_counter", "banknotes_on_counter", "cash_handover_to_cashier")
DEFINITIONS = {
    "client_at_counter": "At endpoint: visitor at the service side of this counter, seated or standing; people behind glass do not count.",
    "banknotes_on_counter": "At endpoint: visible banknote(s) resting on the counter; ID cards, papers, notes only in hands/machine do not count. Occlusion is unknown.",
    "cash_handover_to_cashier": "Within trailing window: visible transfer of banknotes from visitor to cashier, directly or via counter. Document transfer and return to visitor do not count.",
}


class PrimitiveTransform:
    """One service view, or service + desk views, using identical pretrained normalization."""
    def __init__(self, config):
        rois = config.get("view_rois", [config["roi"]])
        if not isinstance(rois, list) or not 1 <= len(rois) <= 2:
            raise ValueError("Use one or two explicit primitive views")
        self.transforms = [video_transform(dict(config, roi=roi)) for roi in rois]

    def __call__(self, frames):
        views = [transform(frames) for transform in self.transforms]
        return views[0] if len(views) == 1 else torch.stack(views)


def causal_times(end_sec, last_time, window_sec=2., frames=16):
    if (frames != 16 or not all(math.isfinite(v) for v in (end_sec, last_time, window_sec))
            or window_sec <= 0 or not 0 <= end_sec <= last_time):
        raise ValueError("Invalid causal primitive window")
    # At clip start repeat available frames, never look into the future.
    return np.linspace(max(0., end_sec - window_sec), end_sec, frames)


def validate_review(value, record, window_sec):
    if (value.get("schema") != SCHEMA or value.get("video_sha256") != record.sha256
            or value.get("window_sec") != window_sec or value.get("method") != "frame_sequence_review"
            or not value.get("reviewer") or not value.get("reviewed_at") or not value.get("points")):
        raise ValueError("Invalid primitive review provenance")
    seen = set()
    for point in value["points"]:
        stamp, labels = point["time_sec"], point["labels"]
        causal_times(stamp, record.last_time, window_sec)
        if stamp in seen or not labels or not set(labels) <= set(CLASSES):
            raise ValueError("Duplicate timestamp or unknown primitive")
        if any(type(v) is not int or v not in (0, 1) for v in labels.values()):
            raise ValueError("Primitive labels must be explicit 0/1; omit unknowns")
        if not point.get("note") or not point.get("evidence_seconds"):
            raise ValueError("Primitive point needs traceable visual evidence")
        evidence = point["evidence_seconds"]
        if any(not math.isfinite(t) or not 0 <= t <= record.last_time for t in evidence):
            raise ValueError("Invalid evidence timestamp")
        if stamp not in evidence:
            raise ValueError("Endpoint must be inspected")
        # Handover is a motion label, not a single-frame appearance guess.
        if "cash_handover_to_cashier" in labels:
            start = max(0., stamp - window_sec)
            if min(evidence) > start or max(evidence) < stamp or len(set(evidence)) < 3:
                raise ValueError("Handover requires a sequence covering the trailing window")
        seen.add(stamp)


@dataclass
class PrimitiveSample:
    record: object
    center_sec: float  # common batch API; semantically the END of the causal window
    positives: tuple
    negatives: tuple
    complete: bool = False


def prepare_primitive_data(config):
    events, split = Path(config["events_path"]), Path(config["split_path"])
    if sha256(events) != config["events_sha256"] or sha256(split) != config["split_sha256"]:
        raise ValueError("Primitive source/split hash differs from fixed experiment")
    records = load_records(events, split, Path(config["video_dir"]))
    with events.open(encoding="utf-8-sig", newline="") as stream:
        rows = {r["clip_id"]: r for r in csv.DictReader(stream)}
    samples, reviews = [], []
    for record in records:
        points = {}
        for alias in record.aliases:
            raw = rows[alias].get("primitive_review", "")
            if not raw:
                continue
            value = json.loads(raw)
            validate_review(value, record, config["window_sec"])
            reviews.append(dict(clip_id=record.clip_id, source_id=alias, review=value))
            for point in value["points"]:
                labels = points.setdefault(point["time_sec"], {})
                for name, target in point["labels"].items():
                    if name in labels and labels[name] != target:
                        raise ValueError("Conflicting primitive labels across duplicate aliases")
                    labels[name] = target
        for stamp, labels in sorted(points.items()):
            samples.append(PrimitiveSample(record, stamp, tuple(k for k, v in labels.items() if v == 1),
                                            tuple(k for k, v in labels.items() if v == 0)))
    coverage = []
    for name in CLASSES:
        item = dict(primitive=name)
        for subset in ("train", "validation"):
            selected = [s for s in samples if s.record.split == subset]
            item[subset] = dict(positive=sum(name in s.positives for s in selected),
                                negative=sum(name in s.negatives for s in selected),
                                unknown=sum(name not in s.positives + s.negatives for s in selected),
                                positive_videos=sorted({s.record.clip_id for s in selected if name in s.positives}),
                                negative_videos=sorted({s.record.clip_id for s in selected if name in s.negatives}))
        coverage.append(item)
    manifest = dict(schema=SCHEMA, events_sha256=sha256(events), split_sha256=sha256(split),
                    definitions=DEFINITIONS, reviews=reviews,
                    videos=[dict(clip_id=r.clip_id, sha256=r.sha256, split=r.split,
                                 source_ids=list(r.aliases)) for r in records])
    return records, samples, dict(coverage=coverage, manifest=manifest,
                                training_ready=all(c["train"]["positive"] and c["train"]["negative"] for c in coverage))


class PrimitiveDataset(Dataset):
    def __init__(self, samples, transform, window_sec):
        self.samples, self.transform, self.window_sec = samples, transform, window_sec

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        times = causal_times(s.center_sec, s.record.last_time, self.window_sec)
        video = self.transform(decode_frames(s.record, times))
        if tuple(video.shape) not in ((3, 16, 112, 112), (2, 3, 16, 112, 112)) or not torch.isfinite(video).all():
            raise ValueError("Invalid primitive preprocessing")
        return dict(video=video, targets=torch.tensor([n in s.positives for n in CLASSES], dtype=torch.float32),
                    mask=torch.tensor([n in s.positives + s.negatives for n in CLASSES], dtype=torch.float32),
                    clip_id=s.record.clip_id, center_sec=s.center_sec)
