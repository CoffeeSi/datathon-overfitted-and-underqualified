"""Evaluate a frozen primitive backbone with a counter-banknote detector.

Sampled visual-sequence ground truth is deliberately separate from legal-event
ground truth. Neither a banknote detection nor a complete one-second review grid
proves ownership, completion of accounting, or a legal violation.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from torch.utils.data import DataLoader

from .action_train import action_epoch, action_metrics, write_json
from .banknote_model import prepare_banknote_data
from .dataset import decode_frames, sha256
from .full_video_evaluation import match_events
from .primitive_data import CLASSES, PrimitiveDataset, prepare_primitive_data
from .primitive_inference import PrimitiveScanner
from .train import ROOT

TARGET = "receipt_then_cash_after_departure"
EVENT_TYPE = "unsecured_accepted_cash"
GRID_SCHEMA = "cash_sequence_grid_v1"


def verify_primitive_manifest(saved, current):
    """Only new box/grid metadata may differ from the trained primitive source.

    Existing points, labels, masks, evidence and split membership must be exact.
    This is a checked source revision, not an unrestricted hash override.
    """
    projected = copy.deepcopy(current)
    projected["events_sha256"] = saved["events_sha256"]
    for entry in projected.get("reviews", []):
        review = entry["review"]
        review.pop("sequence_grid_review", None)
        for point in review["points"]:
            for field in ("banknote_boxes", "banknote_boxes_complete", "banknote_box_review"):
                point.pop(field, None)
    baseline = copy.deepcopy(saved)
    # Also support a primitive checkpoint trained after the same metadata addition.
    for entry in baseline.get("reviews", []):
        entry["review"].pop("sequence_grid_review", None)
        for point in entry["review"]["points"]:
            for field in ("banknote_boxes", "banknote_boxes_complete", "banknote_box_review"):
                point.pop(field, None)
    if projected != baseline:
        raise ValueError("Current primitive points/labels/evidence or split differ from checkpoint")


def _grid(last_time, step=1.):
    if not math.isfinite(last_time) or last_time < 0 or not math.isfinite(step) or step <= 0:
        raise ValueError("Invalid video grid bounds")
    stamps = [i * step for i in range(math.floor(last_time / step) + 1)]
    if last_time > stamps[-1] + 1e-8:
        stamps.append(last_time)
    return stamps


def validate_sequence_grid(review, record):
    """Incomplete reviews remain unknown; complete claims require full grid proof."""
    if (not isinstance(review, dict) or review.get("schema") != GRID_SCHEMA
            or review.get("visual_target") != TARGET
            or review.get("mapped_candidate_type") != EVENT_TYPE
            or review.get("video_sha256") != record.sha256
            or review.get("continuous_legal_review") is not False):
        raise ValueError("Invalid sampled sequence review identity/scope")
    if review.get("complete_for_sampled_grid") is not True:
        return None
    if (review.get("grid_step_sec") != 1.
            or review.get("method") != "full_duration_sampled_frame_sequence_review"
            or any(not isinstance(review.get(k), str) or not review[k].strip()
                   for k in ("reviewer", "reviewed_at", "note", "evidence_manifest"))):
        raise ValueError("Complete sampled grid needs explicit evidence provenance")
    expected = _grid(record.last_time)
    observed = review.get("sampled_seconds")
    if (not isinstance(observed, list) or len(observed) != len(expected)
            or any(type(t) not in (int, float) or not math.isfinite(t)
                   or abs(t - reference) > 1e-8 for t, reference in zip(observed, expected))):
        raise ValueError("Sampled sequence review does not cover the complete one-second grid")
    refs = review.get("reference_times_sec")
    if (not isinstance(refs, list)
            or any(type(t) not in (int, float) or not math.isfinite(t)
                   or not 0 <= t <= record.last_time for t in refs)
            or len(set(refs)) != len(refs)
            or any(not any(abs(t - s) <= 1e-8 for s in expected) for t in refs)):
        raise ValueError("Invalid sampled sequence reference times")
    return review


def _record_grid_review(record, source):
    reviews = []
    for alias in record.aliases:
        raw = source[alias].get("primitive_review")
        value = json.loads(raw) if raw else {}
        review = value.get("sequence_grid_review")
        if review is not None:
            # An incomplete alias is unknown, not a contradictory negative label.
            complete = validate_sequence_grid(review, record)
            if complete is not None:
                reviews.append(complete)
    if not reviews:
        return None
    signature = lambda r: (tuple(r["sampled_seconds"]), tuple(sorted(r["reference_times_sec"])))
    if any(signature(r) != signature(reviews[0]) for r in reviews[1:]):
        raise ValueError("Conflicting sampled sequence review across canonical aliases")
    return reviews[0]


def _binary_metrics(rows):
    tp = sum(r["target"] == 1 and r["prediction"] == 1 for r in rows)
    tn = sum(r["target"] == 0 and r["prediction"] == 0 for r in rows)
    fp = sum(r["target"] == 0 and r["prediction"] == 1 for r in rows)
    fn = sum(r["target"] == 1 and r["prediction"] == 0 for r in rows)
    evaluable = bool(tp + fn) and bool(tn + fp)
    return dict(videos=len(rows), positives=tp + fn, negatives=tn + fp,
                precision=tp / (tp + fp) if evaluable and tp + fp else 0. if evaluable else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if evaluable else None,
                accuracy=(tp + tn) / len(rows) if rows else None,
                confusion_matrix=[[tn, fp], [fn, tp]], evaluable=evaluable)


def summarize_grid_sequences(records, scans, source, tolerance_sec):
    """Video and event metrics only on explicitly complete sampled visual reviews."""
    if len(scans) != len(records) or [s["clip_id"] for s in scans] != [r.clip_id for r in records]:
        raise ValueError("Incomplete or reordered validation scan coverage")
    videos, matching, excluded = [], [], []
    for record, scan in zip(records, scans):
        if scan["video_sha256"] != record.sha256:
            raise ValueError("Scan/review video hash mismatch")
        review = _record_grid_review(record, source)
        if review is None:
            excluded.append(record.clip_id)
            continue
        truth = review["reference_times_sec"]
        predicted = [e["timestamp_sec"] for e in scan["events"] if e["event_type"] == EVENT_TYPE]
        target, prediction = int(bool(truth)), int(bool(predicted))
        videos.append(dict(clip_id=record.clip_id, target=target, prediction=prediction,
                           outcome="TP" if target == prediction == 1 else "TN" if target == prediction == 0
                           else "FP" if prediction else "FN", candidates=len(predicted),
                           reference_times_sec=truth, predicted_times_sec=predicted, review=review))
        matching.extend(dict(clip_id=record.clip_id, **m) for m in match_events(predicted, truth, tolerance_sec))
    matched = [r for r in matching if r["status"] == "matched"]
    missed = [r for r in matching if r["status"] == "unmatched_annotation"]
    extras = [r for r in matching if r["status"] == "unmatched_prediction"]
    hits = len(matched)
    video_metrics = _binary_metrics(videos)
    return dict(schema=GRID_SCHEMA, visual_target=TARGET, mapped_candidate_type=EVENT_TYPE,
                scope="Complete reviewed one-second sampled grids only; not continuous legal ground truth",
                continuous_legal_review=False, independent_evaluation=False,
                video=video_metrics, videos=videos, excluded_unknown_videos=excluded,
                temporal=dict(matched=hits, missed=len(missed), unmatched_predictions=len(extras),
                              precision=hits / (hits + len(extras)) if hits + len(extras) else 0. if missed else None,
                              recall=hits / (hits + len(missed)) if hits + len(missed) else None,
                              f1=2 * hits / (2 * hits + len(extras) + len(missed))
                              if 2 * hits + len(extras) + len(missed) else None,
                              timestamp_mae_sec=sum(r["absolute_error_sec"] for r in matched) / hits if hits else None,
                              tolerance_sec=tolerance_sec), matching=matching)


def _known_action_matching(records, scans, source, tolerance_sec):
    matching = []
    for record, scan in zip(records, scans):
        references = set()
        for alias in record.aliases:
            raw = source[alias].get("action_review")
            review = json.loads(raw) if raw else {}
            if EVENT_TYPE in review.get("present_types", []):
                stamp = review.get("center_sec")
                if (review.get("video_sha256") != record.sha256 or type(stamp) not in (int, float)
                        or not math.isfinite(stamp) or not 0 <= stamp <= record.last_time):
                    raise ValueError("Action marker provenance differs from validated video")
                references.add(stamp)
        predicted = [e["timestamp_sec"] for e in scan["events"] if e["event_type"] == EVENT_TYPE]
        matching.extend(dict(clip_id=record.clip_id, **m) for m in match_events(predicted, sorted(references), tolerance_sec))
    matched = [r for r in matching if r["status"] == "matched"]
    missed = [r for r in matching if r["status"] == "unmatched_annotation"]
    return dict(matched=len(matched), missed=len(missed),
                recall=len(matched) / (len(matched) + len(missed)) if matched or missed else None,
                precision=None, f1=None,
                reason="No complete type-specific continuous legal-event ground truth; extra candidates remain unresolved",
                timestamp_mae_sec=sum(r["absolute_error_sec"] for r in matched) / len(matched) if matched else None,
                tolerance_sec=tolerance_sec), matching


def _verify_scan(record, scan, scanner):
    if (scan["video_sha256"] != record.sha256
            or scan.get("checkpoint_sha256") != scanner.checkpoint_sha256
            or scan.get("banknote_checkpoint_sha256") != scanner.banknote_detector.checkpoint_sha256
            or scan.get("fps") != record.fps or scan.get("frame_count") != record.frame_count
            or not math.isfinite(scan["processing_sec"]) or scan["processing_sec"] <= 0):
        raise ValueError("Invalid hybrid scan provenance/timing")
    if scan.get("banknote_tracking_config") != getattr(scanner, "banknote_tracking_config", None):
        raise ValueError("Banknote tracking protocol differs from the fixed scanner configuration")
    expected = _grid(record.last_time, scanner.config["scan_step_sec"])
    if ([r["time_sec"] for r in scan["observations"]] != expected
            or any(e.get("event_type") != EVENT_TYPE for e in scan["events"])):
        raise ValueError("Incomplete scan grid or unsupported event type")


def evaluate_banknote_pipeline(primitive_checkpoint, banknote_checkpoint, events_path, split_path,
                               video_dir, output_dir, device="cpu", *, track_banknotes=False):
    """Evaluate existing weights, fixed thresholds and fixed split; never train."""
    output = Path(output_dir)
    archive_path = output.with_suffix(".zip")
    if output.exists() or archive_path.exists():
        raise FileExistsError("Use a fresh hybrid report directory AND archive path")
    if type(track_banknotes) is not bool:
        raise ValueError("track_banknotes must be an explicit boolean")
    started = time.perf_counter()
    events_path, split_path = Path(events_path).resolve(), Path(split_path).resolve()
    primitive_checkpoint, banknote_checkpoint = Path(primitive_checkpoint).resolve(), Path(banknote_checkpoint).resolve()
    scanner = PrimitiveScanner(primitive_checkpoint, device, banknote_checkpoint=banknote_checkpoint,
                               banknote_tracking=track_banknotes)
    detector = scanner.banknote_detector
    tracking_config = copy.deepcopy(getattr(scanner, "banknote_tracking_config", None))
    if ((track_banknotes and (not isinstance(tracking_config, dict) or not tracking_config))
            or (not track_banknotes and tracking_config is not None)):
        raise ValueError("Scanner does not expose the requested fixed banknote tracking protocol")
    config = dict(scanner.config, events_path=str(events_path), split_path=str(split_path),
                  video_dir=str(Path(video_dir).resolve()), events_sha256=sha256(events_path), device=str(device),
                  banknote_tracking_config=tracking_config)
    if sha256(split_path) != config["split_sha256"] or detector.config["split_sha256"] != config["split_sha256"]:
        raise ValueError("Hybrid evaluation must preserve both checkpoints' split")
    records, samples, audit = prepare_primitive_data(config)
    verify_primitive_manifest(scanner.checkpoint["data_manifest"], audit["manifest"])
    detector_config = dict(detector.config, events_path=str(events_path), split_path=str(split_path),
                           video_dir=config["video_dir"], device=str(device))
    _, banknote_manifest = prepare_banknote_data(detector_config)
    original_banknote_manifest = detector.checkpoint["data_manifest"]
    if banknote_manifest != {k: v for k, v in original_banknote_manifest.items() if k != "supplemental"}:
        raise ValueError("Banknote evaluation source differs from trained box manifest")
    with events_path.open(encoding="utf-8-sig", newline="") as stream:
        source = {r["clip_id"]: r for r in csv.DictReader(stream)}
    validation_records = [r for r in records if r.split == "validation"]
    validation = [s for s in samples if s.record.split == "validation"]
    # Validate scope before any expensive inference, including unknown/incomplete reviews.
    for record in validation_records:
        _record_grid_review(record, source)
    preparation_sec = time.perf_counter() - started
    window_start = time.perf_counter()
    loader = DataLoader(PrimitiveDataset(validation, scanner.transform, config["window_sec"]),
                        batch_size=2, num_workers=0)
    original_metrics, predictions = action_epoch(scanner.model, loader, scanner.device, CLASSES, config["threshold"])
    if [(r["clip_id"], r["center_sec"]) for r in predictions] != [(s.record.clip_id, s.center_sec) for s in validation]:
        raise ValueError("Primitive prediction/sample coverage differs")
    for row, sample in zip(predictions, validation):
        rgb = decode_frames(sample.record, np.asarray([sample.center_sec]))[0].permute(1, 2, 0).numpy()
        detected = detector.predict(rgb)
        if not math.isfinite(detected["score"]) or not 0 <= detected["score"] <= 1:
            raise ValueError("Invalid detector score")
        row["r3d_probabilities"] = list(row["probabilities"])
        row["probabilities"][1] = detected["score"]
        row["banknotes"] = detected
    metrics = action_metrics(predictions, CLASSES, config["threshold"])
    errors = [dict(clip_id=r["clip_id"], time_sec=r["center_sec"], primitive=name,
                   target=int(r["targets"][i]), score=r["probabilities"][i],
                   error="FN" if r["targets"][i] else "FP")
              for r in predictions for i, name in enumerate(CLASSES)
              if r["mask"][i] and int(r["probabilities"][i] >= config["threshold"]) != r["targets"][i]]
    window_sec = time.perf_counter() - window_start
    scans = []
    for record in validation_records:
        # The scanner receives only a path. Ground truth is used after prediction.
        scan = scanner.scan(record.path)
        _verify_scan(record, scan, scanner)
        scan["clip_id"] = record.clip_id
        scans.append(scan)
        print(record.clip_id, "candidates", len(scan["events"]), "seconds", round(scan["processing_sec"], 2), flush=True)
    if tracking_config != getattr(scanner, "banknote_tracking_config", None):
        raise ValueError("Banknote tracking configuration changed during evaluation")
    tolerance = config["match_tolerance_sec"]
    grid_result = summarize_grid_sequences(validation_records, scans, source, tolerance)
    known_markers, marker_matching = _known_action_matching(validation_records, scans, source, tolerance)
    for path, expected in ((events_path, config["events_sha256"]), (split_path, config["split_sha256"]),
                           (primitive_checkpoint, scanner.checkpoint_sha256),
                           (banknote_checkpoint, detector.checkpoint_sha256)):
        if sha256(path) != expected:
            raise ValueError("Evaluation input changed during inference")
    for record in validation_records:
        if sha256(record.path) != record.sha256:
            raise ValueError("Validation video changed during evaluation")
    summary = dict(schema="banknote_pipeline_evaluation_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                   primitive_checkpoint=str(primitive_checkpoint), primitive_checkpoint_sha256=scanner.checkpoint_sha256,
                   banknote_checkpoint=str(banknote_checkpoint), banknote_checkpoint_sha256=detector.checkpoint_sha256,
                   events_sha256=config["events_sha256"], split_sha256=config["split_sha256"],
                   primitive_training_events_sha256=scanner.config["events_sha256"],
                   banknote_training_events_sha256=detector.config["events_sha256"],
                   banknote_tracking_enabled=track_banknotes, banknote_tracking_config=tracking_config,
                   primitive_point_protocol="Standalone endpoints: original R3D client/handover and banknote detector; tracking is never applied to these metrics",
                   full_scan_protocol="Causal primitive sequence with image-evidence banknote tracking" if track_banknotes
                   else "Causal primitive sequence with standalone banknote detections",
                   coverage=audit["coverage"], reviewed_validation_points=len(validation),
                   primitive_metrics=metrics, original_r3d_metrics=original_metrics,
                   sampled_grid_proxy=grid_result, known_action_markers=known_markers,
                   total_candidates=sum(len(s["events"]) for s in scans),
                   validation_videos=len(scans),
                   videos=[dict(clip_id=s["clip_id"], sha256=s["video_sha256"], candidates=len(s["events"]),
                                processing_sec=s["processing_sec"], uncertain_observations=s["uncertain_observations"])
                           for s in scans],
                   timing=dict(preparation_sec=preparation_sec, validation_point_processing_sec=window_sec,
                               full_scan_processing_sec=sum(s["processing_sec"] for s in scans),
                               total_before_export_sec=time.perf_counter() - started,
                               scope="Sequential full-video decode, transforms, both models, state machine and input hashes; no report export"),
                   independent_evaluation=False, deployment_enabled=False,
                   limitations=["Same room/people, unknown sessions and repeatedly inspected validation; exploratory results.",
                                "Video-level proxy metrics cover only complete sampled visual grids, not all validation videos or continuous legal events.",
                                "Bounding rectangles are weak segmentation labels; uncertain/unreviewed states remain unknown.",
                                "A receipt/departure/cash sequence does not prove cash identity, completed accounting or legal applicability.",
                                "No candidate does not certify absence; legal precision/F1 remain unavailable.",
                                "Nominal FPS sampling can differ from variable-frame-rate presentation timestamps."])
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "summary.json": summary, "evaluation_config.json": config,
        "primitive_checkpoint_config.json": scanner.checkpoint["config"],
        "banknote_checkpoint_config.json": detector.checkpoint["config"],
        "primitive_training_manifest.json": scanner.checkpoint["data_manifest"],
        "banknote_training_manifest.json": original_banknote_manifest,
        "evaluation_manifest.json": audit["manifest"], "primitive_predictions.json": predictions,
        "primitive_errors.json": errors, "known_action_matching.json": marker_matching,
        "sampled_grid_matching.json": grid_result["matching"],
    }
    artifacts.update({f"{s['clip_id']}_scan.json": s for s in scans})
    for name, value in artifacts.items():
        write_json(output / name, value)
    # Explicit generated JSON allowlist only: no videos, evidence images, weights or env files.
    with ZipFile(archive_path, "x", ZIP_DEFLATED) as archive:
        for name in sorted(artifacts):
            archive.write(output / name, name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitive-checkpoint", type=Path, required=True)
    parser.add_argument("--banknote-checkpoint", type=Path, required=True)
    parser.add_argument("--events-path", type=Path, default=ROOT / "data/annotations/events.csv")
    parser.add_argument("--split-path", type=Path, default=ROOT / "data/processed/split_proposal.csv")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data/raw/videos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--track-banknotes", action="store_true",
                        help="Opt in to fixed image-evidence tracking for full scans; endpoint metrics remain untracked")
    args = parser.parse_args()
    result = evaluate_banknote_pipeline(args.primitive_checkpoint, args.banknote_checkpoint, args.events_path,
                                        args.split_path, args.video_dir, args.output_dir, args.device,
                                        track_banknotes=args.track_banknotes)
    print(json.dumps(dict(sampled_grid_proxy=result["sampled_grid_proxy"]["video"],
                         known_action_markers=result["known_action_markers"]), indent=2))


if __name__ == "__main__":
    main()
