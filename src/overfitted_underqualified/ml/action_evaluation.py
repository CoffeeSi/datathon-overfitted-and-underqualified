"""Fixed-threshold action validation and full-chain point matching; no training."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

from torch.utils.data import DataLoader

from .action_data import ActionDataset, prepare_action_data
from .action_inference import prepare_typed_pipeline
from .action_train import action_epoch, read_config, write_json
from .full_video_evaluation import match_events
from .inference import synchronize
from .train import ROOT


def temporal_summary(samples, scans, classes, tolerance_sec=5.0):
    """One-to-one matching within each video/type. Unknown extra alarms stay unresolved."""
    rows, counts = [], []
    for name in classes:
        tp = fn = negative_video_fp = unresolved = gate_hits = 0
        distances = []
        for clip_id, scan in scans.items():
            clip_samples = [s for s in samples if s.record.clip_id == clip_id]
            truths = sorted({s.center_sec for s in clip_samples if name in s.positives})
            predictions = [e["timestamp_sec"] for e in scan["typed_events"] if e["event_type"] == name]
            confirmed_background = bool(clip_samples) and (
                any(name in s.whole_video_absent_types for s in clip_samples)
                or all(s.complete and not s.positives for s in clip_samples))
            gate_hits += sum(any(w["probability"] >= scan["scan_config"]["threshold"]
                                 and w["start_sec"] <= t <= w["end_sec"] for w in scan["windows"]) for t in truths)
            for match in match_events(predictions, truths, tolerance_sec):
                rows.append(dict(clip_id=clip_id, event_type=name, **match,
                                 unmatched_prediction_is_confirmed_fp=confirmed_background))
                if match["status"] == "matched":
                    tp += 1
                    distances.append(match["absolute_error_sec"])
                elif match["status"] == "unmatched_annotation":
                    fn += 1
                elif confirmed_background:
                    negative_video_fp += 1
                else:
                    unresolved += 1
        counts.append(dict(event_type=name, matched=tp, missed=fn, support=tp + fn,
                           recall=tp / (tp + fn) if tp + fn else None,
                           precision=None, f1=None, confirmed_background_false_positives=negative_video_fp,
                           unresolved_predictions=unresolved, binary_gate_hits=gate_hits,
                           binary_gate_coverage=gate_hits / (tp + fn) if tp + fn else None,
                           matched_timestamp_mae_sec=sum(distances) / len(distances) if distances else None))
    return dict(per_class=counts, errors=rows, tolerance_sec=tolerance_sec,
                precision_unavailable_reason="Point-action lists do not confirm full temporal annotation coverage on positive videos.")


def evaluate_actions(config, binary_checkpoint, action_checkpoint, enabled_types, *, tolerance_sec=5.0, output_parent=None):
    records, samples, audit = prepare_action_data(config)
    pipeline = prepare_typed_pipeline(binary_checkpoint, action_checkpoint,
                                      enabled_types=enabled_types, device=config["device"])
    recognizer = pipeline.recognizer
    if audit["class_names"] != recognizer.classes:
        raise ValueError("Evaluation target types differ from checkpoint")
    for key,default in (("preprocessing", "kinetics400_v1"), ("roi", None)):
        if config.get(key,default) != recognizer.checkpoint["config"].get(key,default):
            raise ValueError(f"Evaluation geometry differs from checkpoint: {key}")
    for key in ("window_sec", "frames", "weights", "threshold", "background_ceiling"):
        if config[key] != recognizer.checkpoint["config"][key]:
            raise ValueError(f"Evaluation settings differ from fixed action experiment: {key}")
    saved = recognizer.checkpoint["data_manifest"]
    for key in ("events_sha256", "split_sha256", "catalog_sha256", "review"):
        if saved[key] != audit["manifest"][key]:
            raise ValueError(f"Evaluation differs from checkpoint training provenance: {key}")
    validation = [s for s in samples if s.record.split == "validation"]
    loader = DataLoader(ActionDataset(validation, recognizer.classes, recognizer.transform, recognizer.window_sec),
                        batch_size=config["batch_size"], shuffle=False, num_workers=0)
    synchronize(recognizer.device)
    start = time.perf_counter()
    metrics, rows = action_epoch(recognizer.model, loader, recognizer.device, recognizer.classes, recognizer.threshold)
    synchronize(recognizer.device)
    window_sec = time.perf_counter() - start
    scans = {r.clip_id: pipeline.scan(r.path) for r in records if r.split == "validation"}
    # Include held-out-only types in the temporal denominator: unsupported types are misses.
    all_types = sorted({r["event_type"] for r in audit["coverage"]})
    temporal = temporal_summary(validation, scans, all_types, tolerance_sec)
    known = [w for scan in scans.values() for w in scan["windows"] if w["action_status"] != "binary_negative"]
    report = dict(schema="action_evaluation_v1", provenance=dict(binary=pipeline.scanner.provenance,
                  action=recognizer.provenance, data_manifest=saved), device=str(recognizer.device),
                  window_metrics=metrics, window_processing_sec=window_sec, temporal=temporal,
                  coverage=audit["coverage"],
                  full_video_processing_sec=sum(scan["processing_sec"] for scan in scans.values()),
                  candidate_windows=len(known), typed_windows=sum(bool(w["actions"]) for w in known),
                  abstentions=sum(not w["actions"] for w in known),
                  unresolved_windows=sum(len(scan["unresolved_candidates"]) for scan in scans.values()),
                  recognition_coverage=sum(bool(w["actions"]) for w in known) / len(known) if known else None,
                  limitations=["Shared room, people and unknown sessions; no independent evaluation.",
                               "Only reviewed known labels enter per-window metrics; missing validation classes have null metrics.",
                               "Binary gating limits action recall; misses and gate coverage are reported per type.",
                               "Thresholds and 5-second default tolerance were fixed before evaluation.",
                               "Open-set rejection by confidence is heuristic, not a guarantee.",
                               "Point annotations are not complete temporal action annotation. Unmatched alarms require review."])
    parent = Path(output_parent or ROOT / "reports")
    output = parent / ("evaluation_actions_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    write_json(output / "window_predictions.json", rows)
    # Scans contain numeric decisions and rule metadata only, never extracted frames.
    write_json(output / "scans.json", scans)
    with ZipFile(output.with_suffix(".zip"), "x", ZIP_DEFLATED) as archive:
        for name in ("report.json", "window_predictions.json", "scans.json"):
            archive.write(output / name, name)
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/actions.yaml")
    parser.add_argument("--binary-checkpoint", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--enabled-type", action="append", required=True)
    args = parser.parse_args()
    output, _ = evaluate_actions(read_config(args.config), args.binary_checkpoint, args.action_checkpoint, args.enabled_type)
    print(output)


if __name__ == "__main__":
    main()
