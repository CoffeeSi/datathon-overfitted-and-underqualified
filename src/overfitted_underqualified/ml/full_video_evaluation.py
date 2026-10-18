"""Full-video validation protocol, separate from annotation-centred classification."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import csv
import json
import math
from pathlib import Path
import platform
import time
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .dataset import sha256
from .evaluate import prepare_evaluation, verify_evaluation_inputs
from .inference import FullVideoScanner, ScanConfig, merge_candidates, sliding_windows, synchronize
from .train import ROOT

LIMITATIONS = [
    "Exploratory within_room_unique_video validation; shared room/people and unknown sessions; no independent test.",
    "Checkpoint was selected using this validation set. Scan parameters and tolerance are fixed, not tuned here.",
    "Binary scores do not identify catalog actions or establish legal violations.",
    "Known point annotations may be incomplete. Unmatched candidates on positive videos are unresolved, not confirmed FP.",
    "Negative-video false alarms are relative to the team's accepted binary labels.",
    "One candidate per merged support; sustained alarms can merge multiple true events and cause temporal misses.",
    "Candidate time is the peak window centre, not a learned event boundary. Matching MAE excludes misses; counts are reported.",
    "Nominal FPS/frame indices are inherited from baseline; variable frame rate can shift timestamps. Evidence records decoder PTS separately.",
    "Full-video metrics must not be directly equated with annotation-centred window F1.",
]


def match_events(predictions, annotations, tolerance_sec=5.0):
    """Maximum-cardinality, then minimum-total-error ordered one-to-one matching.

    Scalar times allow an optimal noncrossing matching. Dynamic programming keeps
    original indices, including equal times. Ties prefer a match, then skip prediction.
    Extra predictions remain unmatched even if near an already matched annotation.
    """
    if not math.isfinite(tolerance_sec) or tolerance_sec < 0:
        raise ValueError("Invalid matching tolerance")
    if any(not math.isfinite(t) or t < 0 for t in [*predictions, *annotations]):
        raise ValueError("Invalid event time")
    ps = sorted(enumerate(predictions), key=lambda pair: (pair[1], pair[0]))
    gs = sorted(enumerate(annotations), key=lambda pair: (pair[1], pair[0]))
    n, m = len(ps), len(gs)
    scores = [[(0, 0.0) for _ in range(m + 1)] for _ in range(n + 1)]
    choices = {}
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            options = []
            distance = abs(ps[i][1] - gs[j][1])
            if distance <= tolerance_sec:
                count, cost = scores[i + 1][j + 1]
                options.append(((count + 1, cost + distance), "match"))
            options.extend([(scores[i + 1][j], "prediction"), (scores[i][j + 1], "annotation")])
            score, choice = max(options, key=lambda option: (option[0][0], -option[0][1]))
            scores[i][j], choices[i, j] = score, choice
    matched = []
    used_p, used_g = set(), set()
    i = j = 0
    while i < n and j < m:
        choice = choices[i, j]
        if choice == "match":
            pi, predicted = ps[i]
            gi, reference = gs[j]
            matched.append(dict(status="matched", prediction_index=pi, annotation_index=gi,
                                predicted_sec=predicted, annotation_sec=reference,
                                signed_error_sec=predicted-reference, absolute_error_sec=abs(predicted-reference)))
            used_p.add(pi)
            used_g.add(gi)
            i, j = i + 1, j + 1
        elif choice == "prediction":
            i += 1
        else:
            j += 1
    for pi, predicted in ps:
        if pi not in used_p:
            matched.append(dict(status="unmatched_prediction", prediction_index=pi, annotation_index=None,
                                predicted_sec=predicted, annotation_sec=None,
                                signed_error_sec=None, absolute_error_sec=None))
    for gi, reference in gs:
        if gi not in used_g:
            matched.append(dict(status="unmatched_annotation", prediction_index=None, annotation_index=gi,
                                predicted_sec=None, annotation_sec=reference,
                                signed_error_sec=None, absolute_error_sec=None))
    return matched


def prepare_full_evaluation(checkpoint_path, scan_config: ScanConfig, **kwargs):
    """Reuse the existing strict data/checkpoint validation; pass only paths to scanner."""
    start = time.perf_counter()
    prepared = prepare_evaluation(checkpoint_path, **kwargs)
    config = prepared["checkpoint"]["config"]
    if config["frames"] != 16 or config["weights"] != "KINETICS400_V1":
        raise ValueError("Unsupported checkpoint preprocessing")
    scanner = FullVideoScanner(prepared["model"], prepared["dataset"].transform,
                               scan_config, prepared["device"])
    records = list({r.clip_id: r for r, _ in prepared["dataset"].samples}.values())
    synchronize(prepared["device"])
    return dict(evaluation=prepared, scanner=scanner, records=records,
                preparation_sec=time.perf_counter() - start)


def summarize_scans(records, scans, tolerance_sec):
    if not records or [r.clip_id for r in records] != [s["clip_id"] for s in scans]:
        raise ValueError("Incomplete or reordered canonical video coverage")
    videos, matching, windows, candidates = [], [], [], []
    for record, scan in zip(records, scans):
        if scan["sha256"] != record.sha256:
            raise ValueError("Scanned video differs from validated video hash")
        if (scan["fps"] != record.fps or scan["frame_count"] != record.frame_count
                or scan["duration_sec"] != record.frame_count / record.fps
                or not math.isfinite(scan["processing_sec"]) or scan["processing_sec"] <= 0):
            raise ValueError("Invalid scan metadata or processing time")
        config = ScanConfig(**scan["scan_config"])
        expected = sliding_windows(record.last_time, config)
        if len(expected) != len(scan["windows"]) or any(
                any(window[k] != reference[k] for k in reference)
                for reference, window in zip(expected, scan["windows"])):
            raise ValueError("Incomplete full-video window coverage")
        grouped = merge_candidates(scan["windows"], config)
        if grouped != scan["candidates"]:
            raise ValueError("Candidates differ from fixed postprocessing")
        probability = max(w["probability"] for w in scan["windows"])
        prediction = int(bool(grouped))
        outcome = "TP" if record.label == prediction == 1 else "TN" if record.label == prediction == 0 else "FP" if prediction else "FN"
        videos.append(dict(clip_id=record.clip_id, label=record.label, prediction=prediction,
                           probability=probability, outcome=outcome, candidates=len(grouped),
                           windows=len(scan["windows"]), duration_sec=scan["duration_sec"],
                           processing_sec=scan["processing_sec"], sha256=scan["sha256"]))
        for window in scan["windows"]:
            if window["prediction"] != int(window["probability"] >= config.threshold):
                raise ValueError("Window prediction differs from configured threshold")
            windows.append(dict(clip_id=record.clip_id, **{k: window[k] for k in
                                ("window_index", "start_sec", "end_sec", "center_sec", "probability", "prediction")}))
        candidates.extend(dict(clip_id=record.clip_id, **candidate) for candidate in grouped)
        for row in match_events([c["timestamp_sec"] for c in grouped], record.event_times, tolerance_sec):
            interpretation = row["status"]
            if row["status"] == "unmatched_prediction":
                interpretation = "unresolved_positive_video_candidate" if record.label else "negative_video_false_alarm"
            matching.append(dict(clip_id=record.clip_id, label=record.label, interpretation=interpretation, **row))
    truth, predicted = [r["label"] for r in videos], [r["prediction"] for r in videos]
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
    negatives = [r for r in videos if r["label"] == 0]
    negative_minutes = sum(r["duration_sec"] for r in negatives) / 60
    alarms = sum(r["candidates"] for r in negatives)
    hits = [r for r in matching if r["status"] == "matched"]
    annotation_count = sum(len(r.event_times) for r in records)
    candidate_count = len(candidates)
    metrics = dict(
        video=dict(videos=len(videos), accuracy=float(accuracy_score(truth, predicted)),
                   precision=float(precision), recall=float(recall), f1=float(f1),
                   confusion_matrix=confusion_matrix(truth, predicted, labels=[0, 1]).tolist()),
        false_alarms=dict(negative_videos=len(negatives), negative_minutes=negative_minutes, candidates=alarms,
                          per_negative_video=alarms / len(negatives) if negatives else None,
                          per_negative_minute=alarms / negative_minutes if negative_minutes else None),
        temporal=dict(annotations=annotation_count, candidates=candidate_count, matched=len(hits),
                      unmatched_annotations=annotation_count-len(hits), unmatched_predictions=candidate_count-len(hits),
                      unresolved_positive_candidates=sum(r["interpretation"] == "unresolved_positive_video_candidate" for r in matching),
                      known_marker_recall=len(hits)/annotation_count if annotation_count else None,
                      candidate_match_fraction=len(hits)/candidate_count if candidate_count else None,
                      matched_mae_sec=sum(r["absolute_error_sec"] for r in hits)/len(hits) if hits else None,
                      matched_max_error_sec=max((r["absolute_error_sec"] for r in hits), default=None)),
    )
    return dict(metrics=metrics, videos=videos, windows=windows, candidates=candidates, matching=matching)


def evaluate_full_prepared(prepared, *, tolerance_sec=5.0):
    if not math.isfinite(tolerance_sec) or tolerance_sec < 0:
        raise ValueError("Invalid matching tolerance")
    start = time.perf_counter()
    scans = []
    for record in prepared["records"]:
        scan = prepared["scanner"].scan(record.path)
        scans.append(dict(clip_id=record.clip_id, **scan))
        print(f"{record.clip_id}: {len(scan['windows'])} windows, {len(scan['candidates'])} candidates", flush=True)
    summary = summarize_scans(prepared["records"], scans, tolerance_sec)
    return dict(scans=scans, summary=summary, tolerance_sec=tolerance_sec,
                processing_sec=time.perf_counter()-start)


def _write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (list, dict))
                             else row[key] for key in fields})


def save_full_report(prepared, result, output_dir):
    """Save cached predictions after checking provenance; never rerun inference."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Use a new report directory: {output}")
    p = prepared["evaluation"]
    config = p["config"]
    verify_evaluation_inputs(p)
    if sha256(p["checkpoint_path"]) != p["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    for record in prepared["records"]:
        if sha256(record.path) != record.sha256:
            raise ValueError("Video changed during evaluation")
    if any(scan["scan_config"] != asdict(prepared["scanner"].config) for scan in result["scans"]):
        raise ValueError("Scan configuration drift")
    summary = summarize_scans(prepared["records"], result["scans"], result["tolerance_sec"])
    if not math.isfinite(result["processing_sec"]) or result["processing_sec"] <= 0:
        raise ValueError("Invalid processing time")
    report = dict(protocol="full_video_sliding_windows_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint_sha256=p["checkpoint_sha256"], checkpoint=str(p["checkpoint_path"]),
                  checkpoint_epoch=p["checkpoint"]["epoch"], model_config=p["checkpoint"]["config"],
                  scan_config=asdict(prepared["scanner"].config), matching_tolerance_sec=result["tolerance_sec"],
                  matching_rule="maximum cardinality then minimum total absolute error, one-to-one",
                  data_manifest=p["checkpoint"]["data_manifest"],
                  evaluation_events_sha256=p.get("evaluation_events_sha256", p["checkpoint"]["data_manifest"]["events_sha256"]),
                  annotation_revision=p.get("annotation_revision"),
                  evaluated_videos=[dict(clip_id=r.clip_id, sha256=r.sha256, source_clip_ids=list(r.aliases),
                                         event_times=list(r.event_times)) for r in prepared["records"]],
                  metrics=summary["metrics"],
                  timing=dict(preparation_sec=prepared["preparation_sec"], processing_sec=result["processing_sec"],
                              mean_video_sec=result["processing_sec"]/len(prepared["records"]),
                              scope="one full pass: metadata, hashing, decoding, preprocessing, transfer, inference, merge, matching; report/evidence excluded",
                              warmup="no dedicated warmup; earlier notebook work may have warmed the runtime"),
                  environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                                   torchvision=str(torchvision.__version__), opencv=cv2.__version__, numpy=np.__version__,
                                   device=str(p["device"]), cuda_runtime=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(p["device"]) if p["device"].type == "cuda" else None),
                  implementation_sha256={path.name: sha256(path) for path in
                                         (Path(__file__), Path(__file__).with_name("inference.py"),
                                          Path(__file__).with_name("dataset.py"), Path(__file__).with_name("train.py"))},
                  limitations=LIMITATIONS)
    output.mkdir(parents=True)
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(output / "video_predictions.csv", summary["videos"], list(summary["videos"][0]))
    _write_csv(output / "window_predictions.csv", summary["windows"], list(summary["windows"][0]))
    _write_csv(output / "candidates.csv", summary["candidates"],
               ["clip_id", "candidate_id", "start_sec", "end_sec", "timestamp_sec", "score", "peak_window_index",
                "window_indices", "event_type", "type_confidence", "source_links"])
    _write_csv(output / "matching.csv", summary["matching"],
               ["clip_id", "label", "interpretation", "status", "prediction_index", "annotation_index", "predicted_sec",
                "annotation_sec", "signed_error_sec", "absolute_error_sec"])
    errors = [r for r in summary["videos"] if r["outcome"] in {"FP", "FN"}]
    _write_csv(output / "video_errors.csv", errors, list(summary["videos"][0]))
    lines = ["# Full-video binary baseline evaluation", "", f"Checkpoint SHA-256: `{p['checkpoint_sha256']}`", "",
             f"Training events SHA-256: `{p['checkpoint']['data_manifest']['events_sha256']}`", "",
             f"Evaluation events SHA-256: `{report['evaluation_events_sha256']}`", "",
             "Metrics (full-video protocol; not annotation-centred F1):", "", "```json",
             json.dumps(summary["metrics"], indent=2), "```", "", "## Error analysis", ""]
    for row in errors:
        lines.append(f"- {row['clip_id']}: {row['outcome']}, max probability={row['probability']:.4f}, "
                     f"{row['candidates']} candidates over {row['duration_sec']:.2f}s. "
                     + ("At least one alarm on an accepted negative video." if row["outcome"] == "FP" else "No window reached the fixed threshold."))
    if not errors:
        lines.append("No video-level errors in this pass; this does not establish event localization.")
    lines.extend(["", "Temporal misses and unresolved candidates are listed in matching.csv. "
                  "MAE uses matched pairs only; inspect missed counts alongside it. "
                  "Frame-level explanations require local human inspection; no causal explanation is inferred from scores.",
                  "", "## Limitations", "", *[f"- {item}" for item in LIMITATIONS]])
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def archive_full_report(output_dir):
    """Explicit file allowlist: no media, model weights, training folders or secrets."""
    output = Path(output_dir)
    names = ("evaluation.json", "video_predictions.csv", "window_predictions.csv", "candidates.csv",
             "matching.csv", "video_errors.csv", "report.md")
    for name in names:
        if not (output / name).is_file() or (output / name).is_symlink():
            raise ValueError(f"Missing/unsafe report artifact: {name}")
    archive = output.with_suffix(".zip")
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as stream:
        for name in names:
            stream.write(output / name, arcname=f"{output.name}/{name}")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/full_video.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--events-path", type=Path, default=ROOT / "data/annotations/events.csv")
    parser.add_argument("--split-path", type=Path, default=ROOT / "data/processed/split_proposal.csv")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data/raw/videos")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prepared = prepare_full_evaluation(args.checkpoint, ScanConfig(**config["scan"]), device=args.device,
                                       events_path=args.events_path, split_path=args.split_path, video_dir=args.video_dir)
    result = evaluate_full_prepared(prepared, tolerance_sec=config["matching_tolerance_sec"])
    save_full_report(prepared, result, args.output_dir)
    print(archive_full_report(args.output_dir))


if __name__ == "__main__":
    main()
