"""Audit catalog coverage; descriptive tags never become verified training targets."""
import csv
import json
from pathlib import Path

from .catalog import CATALOG_PATH, event_type_for_observed, supported_events
from .evidence import proposed_mapping
from ..ml.dataset import parse_time, sha256


def catalog_coverage(events_path: Path, split_path: Path) -> dict:
    with Path(events_path).open(encoding="utf-8-sig", newline="") as stream:
        events = list(csv.DictReader(stream))
    with Path(split_path).open(encoding="utf-8-sig", newline="") as stream:
        split = list(csv.DictReader(stream))
    aliases = {}
    hashes = set()
    for record in split:
        if record["sha256"] in hashes:
            raise ValueError("Duplicate canonical hash in split")
        hashes.add(record["sha256"])
        for alias in json.loads(record["source_clip_ids"]):
            if alias in aliases:
                raise ValueError("Overlapping aliases")
            aliases[alias] = record
    if len({r["clip_id"] for r in events}) != len(events) or set(aliases) != {r["clip_id"] for r in events}:
        raise ValueError("Source coverage differs from split")
    rows = []
    for event in supported_events():
        # Retain raw catalog hypotheses for historical counts. Expose the guarded
        # proposals separately: neither group constitutes reviewed action targets.
        selected = [r for r in events if r['observed_event_type'] in event.get('maps_from_observed_event_type', [])]
        usable = [r for r in selected if event['event_type'] in proposed_mapping(r['observed_event_type'])['candidate_types']]
        blocked = [r['clip_id'] for r in selected if r not in usable]
        by_subset = {subset: {aliases[r["clip_id"]]["clip_id"] for r in selected
                              if aliases[r["clip_id"]]["split"] == subset} for subset in ("train", "validation")}
        points = {(aliases[r["clip_id"]]["clip_id"], parse_time(r["timecode"])) for r in selected}
        points = {point for point in points if point[1] is not None}
        has_both = bool(by_subset["train"] and by_subset["validation"])
        rows.append(dict(event_type=event["event_type"], source_records=len(selected),
                         canonical_videos=len(by_subset["train"] | by_subset["validation"]),
                         train_videos=len(by_subset["train"]), validation_videos=len(by_subset["validation"]),
                         distinct_markers=len(points), source_ids=[r["clip_id"] for r in selected],
                         raw_catalog_hypotheses_only=True,
                         usable_proposal_source_ids=[r['clip_id'] for r in usable],
                         usable_proposal_canonical_videos=len({aliases[r['clip_id']]['clip_id'] for r in usable}),
                         needs_context_review_ids=blocked,
                         example_ids_without_tag_match=sorted(set(event.get("example_clip_ids", [])) - {r["clip_id"] for r in selected}),
                         present_in_both_subsets=has_both, verified_multiclass_targets=False,
                         assessment="Exploratory only; too few independent videos and unverified action tags" if has_both
                                    else "Cannot evaluate held-out action recognition with this split",
                         applicability=event["applicability"], video_cannot_establish=event.get("video_cannot_establish", [])))
    return dict(events_sha256=sha256(Path(events_path)), split_sha256=sha256(Path(split_path)),
                catalog_sha256=sha256(CATALOG_PATH), rows=rows,
                unmapped_positive_ids=[r["clip_id"] for r in events if r["source_label"] == "violation"
                                       and event_type_for_observed(r["observed_event_type"]) is None])
