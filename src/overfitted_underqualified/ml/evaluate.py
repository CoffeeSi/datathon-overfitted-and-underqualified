"""Evaluate saved weights on the fixed validation windows, with a fixed 0.5 threshold.

Paths may be relocated explicitly; annotation/split hashes must still match the run.
No pretrained weights are downloaded, and no optimizer is constructed.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import platform
import time
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import torch
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision.models.video import R3D_18_Weights

from .dataset import VideoWindowDataset, load_records, sha256
from .train import ROOT, load_checkpoint, run_epoch, seed_everything


def resolve_checkpoint(checkpoint_path=None, *, root=ROOT):
    """Use an explicit checkpoint, or the only available best.pt; never guess among runs."""
    root = Path(root).resolve()
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. Copy best.pt from Brev into this repository "
                "or set CHECKPOINT_PATH to its actual location."
            )
        return path
    candidates = sorted((root / "runs").glob("*/best.pt"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            f"No runs/*/best.pt under {root}. Download the training run from Brev, "
            "or run evaluation on Brev where the checkpoint exists."
        )
    available = "\n".join(str(path.relative_to(root)) for path in candidates)
    raise ValueError(f"Multiple checkpoints found. Set CHECKPOINT_PATH explicitly:\n{available}")


def prepare_evaluation(checkpoint_path, *, device="cpu", events_path=None,
                       split_path=None, video_dir=None):
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_digest = sha256(checkpoint_path)
    model, checkpoint = load_checkpoint(Path(checkpoint_path), device)
    if sha256(checkpoint_path) != checkpoint_digest:
        raise ValueError("Checkpoint changed during loading")
    config = dict(checkpoint["config"])
    config["device"] = str(torch.device(device))
    for key, override in (("events_path", events_path), ("split_path", split_path),
                          ("video_dir", video_dir)):
        path = Path(override if override is not None else config[key])
        config[key] = str(path if path.is_absolute() else ROOT / path)
    seed_everything(config["seed"], config["cpu_threads"])
    manifest = checkpoint["data_manifest"]
    for key in ("events", "split"):
        if sha256(Path(config[f"{key}_path"])) != manifest[f"{key}_sha256"]:
            raise ValueError(f"{key} hash differs from checkpoint provenance")
    records = load_records(Path(config["events_path"]), Path(config["split_path"]),
                           Path(config["video_dir"]))
    videos = [{"clip_id": r.clip_id, "split": r.split, "sha256": r.sha256,
               "source_clip_ids": list(r.aliases), "event_times": list(r.event_times),
               "label": r.label} for r in records]
    if videos != manifest["videos"]:
        raise ValueError("Video records differ from checkpoint provenance")
    dataset = VideoWindowDataset(records, "validation", R3D_18_Weights[config["weights"]].transforms(),
                                 config["window_sec"], config["frames"], config["negative_windows"])
    windows = [{"clip_id": r.clip_id, "center_sec": t, "label": r.label} for r, t in dataset.samples]
    if windows != manifest["windows"]["validation"]:
        raise ValueError("Validation windows differ from checkpoint provenance")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    return dict(model=model, checkpoint=checkpoint, config=config, dataset=dataset,
                loader=loader, device=torch.device(device), checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_digest)


def evaluate_prepared(prepared):
    """Inference only. run_epoch sets eval mode and disables gradient recording."""
    return run_epoch(prepared["model"], prepared["loader"], prepared["device"])


def evaluate_checkpoint(checkpoint_path, **kwargs):
    return evaluate_prepared(prepare_evaluation(checkpoint_path, **kwargs))


def aggregate_video_predictions(window_rows):
    """One vote per canonical video: max window probability, fixed threshold 0.5.

    This is aggregation of known annotation-centred windows, not full-video inference.
    Multiple markers on one canonical video must not count as separate videos.
    """
    grouped = {}
    for row in window_rows:
        probability = float(row["probability"])
        if row["label"] not in (0, 1) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid window label or probability")
        grouped.setdefault(row["clip_id"], []).append(row)
    if not grouped:
        raise ValueError("No video predictions")
    videos = []
    for clip_id, rows in grouped.items():
        if len({row["label"] for row in rows}) != 1:
            raise ValueError(f"Conflicting labels for {clip_id}")
        probability = max(float(row["probability"]) for row in rows)
        label, prediction = rows[0]["label"], int(probability >= 0.5)
        outcome = "TP" if label == prediction == 1 else "TN" if label == prediction == 0 else "FP" if prediction else "FN"
        videos.append(dict(clip_id=clip_id, label=label, probability=probability,
                           prediction=prediction, windows=len(rows), outcome=outcome))
    truth, predicted = [row["label"] for row in videos], [row["prediction"] for row in videos]
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
    metrics = dict(videos=len(videos), precision=float(precision), recall=float(recall), f1=float(f1),
                   accuracy=float(accuracy_score(truth, predicted)),
                   confusion_matrix=confusion_matrix(truth, predicted, labels=[0, 1]).tolist())
    return metrics, videos


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_evaluation_report(checkpoint_path, output_dir, **kwargs):
    """Evaluate once, time the complete window pass, and save auditable artifacts.

    Timing includes decoding, preprocessing, transfers, forward passes and metrics.
    It excludes preparation/model loading (reported separately) and report serialization.
    No warm-up: the first pass includes cold-start overhead. This is not full-video speed.
    """
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new report directory: {output}")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_digest = sha256(checkpoint_path)
    start = time.perf_counter()
    prepared = prepare_evaluation(checkpoint_path, **kwargs)
    _sync(prepared["device"])
    preparation_sec = time.perf_counter() - start
    start = time.perf_counter()
    window_metrics, windows = evaluate_prepared(prepared)
    _sync(prepared["device"])
    processing_sec = time.perf_counter() - start
    return save_evaluation_report(prepared, checkpoint_path, output, window_metrics, windows,
                                  processing_sec=processing_sec, preparation_sec=preparation_sec,
                                  checkpoint_digest=checkpoint_digest)


def save_evaluation_report(prepared, checkpoint_path, output_dir, window_metrics, windows, *,
                           processing_sec, preparation_sec=None, checkpoint_digest=None):
    """Save results already computed by a notebook; never run inference again."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new report directory: {output}")
    if not math.isfinite(processing_sec) or processing_sec <= 0:
        raise ValueError("processing_sec must be positive and finite")
    checkpoint_path = Path(checkpoint_path).resolve()
    if "checkpoint_path" in prepared and checkpoint_path != prepared["checkpoint_path"]:
        raise ValueError("Report checkpoint differs from loaded model")
    checkpoint_digest = prepared.get("checkpoint_sha256", checkpoint_digest) or sha256(checkpoint_path)
    expected_windows = [(record.clip_id, float(center)) for record, center in prepared["dataset"].samples]
    actual_windows = [(row["clip_id"], float(row["center_sec"])) for row in windows]
    if actual_windows != expected_windows:
        raise ValueError("Incomplete or reordered validation window coverage")
    video_metrics, videos = aggregate_video_predictions(windows)
    expected = {record.clip_id for record, _ in prepared["dataset"].samples}
    if {row["clip_id"] for row in videos} != expected:
        raise ValueError("Incomplete validation video coverage")
    config = prepared["config"]
    for key in ("events", "split"):
        if sha256(Path(config[f"{key}_path"])) != prepared["checkpoint"]["data_manifest"][f"{key}_sha256"]:
            raise ValueError(f"{key} changed during evaluation")
    if sha256(checkpoint_path) != checkpoint_digest:
        raise ValueError("Checkpoint changed during evaluation")
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint_path),
        checkpoint_sha256=checkpoint_digest, epoch=prepared["checkpoint"]["epoch"],
        saved_config=prepared["checkpoint"]["config"], effective_config=config,
        data_manifest=prepared["checkpoint"]["data_manifest"],
        protocol="within_room_unique_video; annotation-centred window aggregation",
        aggregation="max probability per canonical video", threshold=0.5,
        confusion_matrix_order=["no_violation", "violation"],
        confusion_matrix_axes="rows=true; columns=predicted",
        video_metrics=video_metrics, window_metrics=window_metrics,
        errors=[row for row in videos if row["outcome"] in ("FP", "FN")],
        timing=dict(preparation_sec=preparation_sec, processing_sec=processing_sec,
                    mean_sec_per_video=processing_sec / len(videos),
                    mean_sec_per_window=processing_sec / len(windows),
                    videos_per_sec=len(videos) / processing_sec,
                    scope="single cold pass: decode + preprocessing + transfer + inference + window metrics; no full-video scan",
                    batch_size=config["batch_size"], warmup_passes=0),
        environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                         torchvision=str(torchvision.__version__), device=str(prepared["device"]),
                         cpu=platform.processor(), cpu_threads=torch.get_num_threads(),
                         gpu=torch.cuda.get_device_name(prepared["device"]) if prepared["device"].type == "cuda" else None),
        limitations=["Known positive timestamps determine evaluated windows; no blind full-video detection",
                     "Shared room/people and unknown sessions may cross subsets",
                     "Validation was used for checkpoint selection; no independent held-out test",
                     "Max aggregation can increase false alarms when the number of windows increases",
                     "Single timing pass is not a latency benchmark; per-video time is an amortized mean"],
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows in (("video_predictions", videos), ("window_predictions", windows)):
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    matrix = video_metrics["confusion_matrix"]
    lines = ["# Baseline evaluation", "", f"Checkpoint SHA-256: `{checkpoint_digest}`; epoch {report['epoch']}.",
             f"Events SHA-256: `{report['data_manifest']['events_sha256']}`.",
             f"Split SHA-256: `{report['data_manifest']['split_sha256']}`.", "",
             "Protocol: max probability per canonical validation video over known annotation-centred windows; threshold 0.5.",
             "", f"Videos: {len(videos)}; windows: {len(windows)}.",
             f"Precision={video_metrics['precision']:.4f}; recall={video_metrics['recall']:.4f}; F1={video_metrics['f1']:.4f}; accuracy={video_metrics['accuracy']:.4f}.",
             f"Confusion matrix (true rows / predicted columns; no_violation, violation): `{matrix}`.", "",
             f"Processing: {processing_sec:.3f} s; preparation: {str(preparation_sec) + ' s' if preparation_sec is not None else 'not measured'}; amortized {processing_sec/len(videos):.3f} s/video.",
             "Timing includes decoding/preprocessing/inference and window metrics; one cold pass, no full-video scan.", "", "## Errors", ""]
    lines += [f"- {row['outcome']} {row['clip_id']}: p={row['probability']:.6f}, true={row['label']}, predicted={row['prediction']}." for row in report["errors"]]
    if not report["errors"]:
        lines += ["No FP/FN in this validation pass; small-sample limitations still apply."]
    lines += ["", "No visual cause is inferred from a score alone. Review evidence and annotations before proposing changes.",
              "", "## Limitations", ""] + [f"- {item}" for item in report["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # A metrics-only figure; no banking frames or annotations are embedded.
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=torch.tensor(matrix).numpy(),
                           display_labels=["no_violation", "violation"]).plot(ax=axis, colorbar=False)
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=120)
    plt.close(figure)
    return report


def archive_evaluation_report(output_dir, checkpoint_path):
    """ZIP an explicit list of result files, never weights, frames or .env.

    Matching training provenance is required before including optional run artifacts.
    A pre-existing archive is never overwritten.
    """
    output = Path(output_dir).resolve()
    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(checkpoint_path).resolve()
    if sha256(checkpoint_path) != report["checkpoint_sha256"]:
        raise ValueError("Archive checkpoint differs from evaluation")
    training_dir = checkpoint_path.parent
    training_manifest = training_dir / "data_manifest.json"
    optional = []
    if training_manifest.is_file() and json.loads(training_manifest.read_text(encoding="utf-8")) == report["data_manifest"]:
        for name in ("history.json", "config.yaml", "environment.json", "data_manifest.json"):
            source = training_dir / name
            if source.is_file():
                target = output / ("training_" + name)
                if target.exists():
                    if target.read_bytes() != source.read_bytes():
                        raise FileExistsError(f"Different training artifact already exists: {target}")
                else:
                    shutil.copy2(source, target)
                optional.append(target.name)
    allowed = ["evaluation.json", "video_predictions.csv", "window_predictions.csv", "report.md",
               "confusion_matrix.png", "error_analysis.csv", "sampling_diagnostics.csv", *optional]
    archive = output.with_suffix(output.suffix + ".zip")
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as stream:
        for name in allowed:
            path = output / name
            if path.is_file():
                stream.write(path, arcname=output.name + "/" + name)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--events-path", type=Path, default=ROOT / "data/annotations/events.csv")
    parser.add_argument("--split-path", type=Path, default=ROOT / "data/processed/split_proposal.csv")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data/raw/videos")
    args = vars(parser.parse_args())
    report = write_evaluation_report(args.pop("checkpoint"), args.pop("output_dir"), **args)
    print(json.dumps(dict(video_metrics=report["video_metrics"], timing=report["timing"]), indent=2))


if __name__ == "__main__":
    main()
