"""Reviewed point-action targets, independent of binary labels and catalog applicability."""
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset import decode_frames, load_records, parse_time, sample_times, sha256
from .action_review import SEMANTIC_GAPS, read_review, validate_review
from ..rules.catalog import CATALOG_PATH, supported_events
from ..rules.evidence import proposed_mapping

SCHEMA = "catalog_actions_multilabel_v1"


def mapped_types(tag):
    return proposed_mapping(tag)['candidate_types']


@dataclass
class ActionSample:
    record: object
    center_sec: float
    positives: tuple[str, ...]
    complete: bool
    source_ids: tuple[str, ...]
    negatives: tuple[str, ...] = ()
    whole_video_absent_types: tuple[str, ...] = ()
    reviews: tuple[dict, ...] = ()


def reviewed_samples(records, rows, window_sec):
    """Only explicit per-type reviews supervise this mode; omitted types stay unknown."""
    samples = []
    for record in records:
        points = {}
        for alias in record.aliases:
            review = read_review(rows[alias], record, window_sec)
            if review is None:
                continue
            point = points.setdefault(review["center_sec"], dict(positive=set(), negative=set(), whole=set(), ids=[], reviews=[]))
            for key, field in (("positive", "present_types"), ("negative", "absent_types"), ("whole", "whole_video_absent_types")):
                point[key].update(review[field])
            point["ids"].append(alias)
            point["reviews"].append(review)
        whole = set().union(*(p["whole"] for p in points.values()))
        for center, point in points.items():
            if point["positive"] & (point["negative"] | whole):
                raise ValueError(f"Conflicting action reviews for canonical video {record.clip_id}")
            times = sample_times(center, record.last_time, window_sec, 16)
            for other_center, other in points.items():
                if times[0] <= other_center <= times[-1] and point["negative"] & other["positive"]:
                    raise ValueError(f"Absent action overlaps a reviewed positive point: {record.clip_id}")
            samples.append(ActionSample(record, center, tuple(sorted(point["positive"])), False,
                                        tuple(point["ids"]), tuple(sorted(point["negative"])),
                                        tuple(sorted(point["whole"])), tuple(point["reviews"])))
    return samples


def validate_review_manifest(manifest, classes, window_sec):
    """Revalidate partial-review provenance without reopening the training videos."""
    videos = {v["clip_id"]: v for v in manifest["videos"]}
    support = {name: [0, 0] for name in classes}
    for point in manifest["points"]:
        video = videos[point["clip_id"]]
        if point["split"] != video["split"] or not set(point["source_ids"]) <= set(video["source_ids"]):
            raise ValueError("Action checkpoint point conflicts with video provenance")
        reviews = point.get("reviews", [])
        if not reviews or len(reviews) != len(point["source_ids"]):
            raise ValueError("Action checkpoint point lacks per-type review")
        for value in reviews:
            validate_review(value, center=point["center_sec"], window_sec=window_sec, video_sha256=video["sha256"])
        positive = set().union(*(set(r["present_types"]) for r in reviews))
        negative = set().union(*(set(r["absent_types"]) for r in reviews))
        whole = set().union(*(set(r["whole_video_absent_types"]) for r in reviews))
        if (positive & negative or positive != set(point["types"]) or negative != set(point["negative_types"])
                or whole != set(point["whole_video_absent_types"]) or point["complete"] is not False):
            raise ValueError("Action checkpoint targets conflict with per-type review")
        if point["split"] == "train":
            for name in classes:
                support[name][0] += name in positive
                support[name][1] += name in negative
    if not classes or any(not p or not n for p, n in support.values()):
        raise ValueError("Action checkpoint lacks reviewed positive/negative training support")
    for clip_id in videos:
        points = [p for p in manifest["points"] if p["clip_id"] == clip_id]
        positive = set().union(*(set(p["types"]) for p in points))
        whole = set().union(*(set(p["whole_video_absent_types"]) for p in points))
        if positive & whole:
            raise ValueError("Action checkpoint whole-video absence conflicts with a positive point")


def prepare_action_data(config):
    """Fail closed on stale review; reuse binary data integrity checks, never its targets.

    Review attestations pin the exact CSV digest, not a second editable label file.
    Completeness is separate from confirmation of listed observations. Without it,
    absent types remain unknown, including on binary 'no_violation' videos.
    """
    events = Path(config["events_path"])
    digest = sha256(events)
    review = config["review"]
    if review.get("events_sha256") != digest or review.get("listed_actions_verified") is not True:
        raise ValueError("Review must confirm the listed actions and pin the current events.csv SHA-256")
    mode = review.get("label_mode", "listed_actions")
    if mode not in {"listed_actions", "per_type_v1"}:
        raise ValueError("Unknown action label review mode")
    complete = review.get("complete_action_lists") is True
    records = load_records(events, Path(config["split_path"]), Path(config["video_dir"]))
    with events.open(encoding="utf-8-sig", newline="") as stream:
        rows = {r["clip_id"]: r for r in csv.DictReader(stream)}
    samples, unresolved = [], []
    for record in records:
        points = {}
        for alias in record.aliases:
            row = rows[alias]
            tag = row.get("observed_event_type", "")
            types = mapped_types(tag)
            center = parse_time(row["timecode"])
            if center is None:
                continue
            if not types:
                unresolved.append(dict(clip_id=alias, canonical_id=record.clip_id,
                                       observed_event_type=tag, split=record.split))
            point = points.setdefault(center, dict(types=set(), ids=[], mapped=True))
            point["types"].update(types)
            point["ids"].append(alias)
            point["mapped"] &= bool(types)
        for center, point in points.items():
            # Preserve distinct canonical alias timestamps. Include co-occurring known
            # point actions inside the input support, rather than marking them absent.
            times = sample_times(center, record.last_time, config["window_sec"], 16)
            labels = set(point["types"])
            fully_mapped = point["mapped"]
            for t, other in points.items():
                if times[0] <= t <= times[-1]:
                    labels.update(other["types"])
                    fully_mapped &= other["mapped"]
            samples.append(ActionSample(record, center, tuple(sorted(labels)),
                                        complete and fully_mapped, tuple(point["ids"])))
        if not points:
            # Sampling background is permitted only after an explicit all-types review.
            samples.append(ActionSample(record, record.last_time / 2, (),
                                        complete and all(rows[a].get("observed_event_type") == "not_reported"
                                                         for a in record.aliases), record.aliases))
    candidates = samples
    if mode == "per_type_v1":
        samples = reviewed_samples(records, rows, config["window_sec"])
    classes = sorted({t for s in samples if s.record.split == "train" for t in s.positives})
    requested = config.get("target_types")
    if requested is not None:
        if (not isinstance(requested, list) or not requested or len(set(requested)) != len(requested)
                or not set(requested) <= {e["event_type"] for e in supported_events()}):
            raise ValueError("Invalid target_types")
        if not set(requested) <= set(classes):
            raise ValueError("Requested action types lack reviewed train positives")
        classes = sorted(requested)
    all_types = sorted(e["event_type"] for e in supported_events())
    coverage = []
    for name in all_types:
        item = dict(event_type=name,
                    has_train_positives=any(s.record.split == "train" and name in s.positives for s in samples),
                    selected_for_training=name in classes)
        for subset in ("train", "validation"):
            selected = [s for s in samples if s.record.split == subset]
            positives = [s for s in selected if name in s.positives]
            item[subset] = dict(positive_points=len(positives),
                                positive_videos=len({s.record.clip_id for s in positives}),
                                candidate_positive_points=sum(s.record.split == subset and name in s.positives for s in candidates),
                                known_negative_windows=sum(name in s.negatives or (s.complete and name not in s.positives) for s in selected))
        item["trainable"] = (name in classes and item["train"]["known_negative_windows"] > 0
                             and (mode == "per_type_v1" or review.get("catalog_mapping_verified") is True))
        item["validation_evaluable"] = bool(item["validation"]["positive_points"] and item["validation"]["known_negative_windows"])
        coverage.append(item)
    manifest = dict(schema=SCHEMA, events_sha256=digest,
                    split_sha256=sha256(Path(config["split_path"])), catalog_sha256=sha256(CATALOG_PATH),
                    review=review, independent_evaluation=False,
                    videos=[dict(clip_id=r.clip_id, sha256=r.sha256, split=r.split,
                                 source_ids=list(r.aliases)) for r in records],
                    points=[dict(clip_id=s.record.clip_id, center_sec=s.center_sec,
                                 types=list(s.positives), complete=s.complete,
                                 negative_types=list(s.negatives), whole_video_absent_types=list(s.whole_video_absent_types),
                                 reviews=list(s.reviews),
                                 source_ids=list(s.source_ids), split=s.record.split) for s in samples])
    blockers = [r["event_type"] for r in coverage if r["selected_for_training"] and
                not r["train"]["known_negative_windows"]]
    audit = dict(schema=SCHEMA, class_names=classes, coverage=coverage, unmapped=unresolved,
                 training_blockers=blockers, manifest=manifest,
                 catalog_mapping_verified=(mode == "per_type_v1" and bool(samples)) or review.get("catalog_mapping_verified") is True,
                 label_mode=mode, reviewed_windows=len(samples) if mode == "per_type_v1" else 0,
                 note="Positive-only labels do not support a discriminative action classifier. "
                      "Binary negatives are not action negatives without an explicit completeness review.")
    audit["training_ready"] = bool(classes) and not blockers and audit["catalog_mapping_verified"]
    audit["readiness_reasons"] = (
        (["No reviewed positive action classes in train"] if not classes else [])
        + (["No per-type reviewed windows"] if mode == "per_type_v1" and not samples else [])
        + (["Catalog mapping has not been confirmed"] if not audit["catalog_mapping_verified"] else [])
        + [f"Missing confirmed train negatives: {name}" for name in blockers]
    )
    return records, samples, audit


class ActionDataset(Dataset):
    def __init__(self, samples, classes, transform, window_sec):
        self.samples = samples
        self.classes = classes
        self.transform = transform
        self.window_sec = window_sec

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        times = sample_times(sample.center_sec, sample.record.last_time, self.window_sec, 16)
        video = self.transform(decode_frames(sample.record, times))
        if tuple(video.shape) != (3, 16, 112, 112) or not torch.isfinite(video).all():
            raise ValueError("Invalid action preprocessing")
        targets = torch.tensor([name in sample.positives for name in self.classes], dtype=torch.float32)
        mask = torch.ones_like(targets) if sample.complete else torch.tensor(
            [name in sample.positives or name in sample.negatives for name in self.classes], dtype=torch.float32)
        return dict(video=video, targets=targets, mask=mask,
                    clip_id=sample.record.clip_id, center_sec=sample.center_sec)
