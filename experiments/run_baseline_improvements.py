"""Predeclared binary ablations. Preserve old runs; no annotation edits or threshold search."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from overfitted_underqualified.ml.dataset import load_records, decode_frames, sample_times, sha256
from overfitted_underqualified.ml.preprocessing import video_transform
from overfitted_underqualified.ml.train import ROOT, run
from overfitted_underqualified.ml.evaluate import prepare_evaluation, evaluate_prepared, aggregate_video_predictions
from overfitted_underqualified.ml.full_video_evaluation import (
    prepare_full_evaluation, evaluate_full_prepared, save_full_report, archive_full_report,
)
from overfitted_underqualified.ml.inference import ScanConfig


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def inspect_train_roi(output):
    records = load_records(ROOT / "data/annotations/events.csv", ROOT / "data/processed/split_proposal.csv", ROOT / "data/raw/videos")
    records = [r for r in records if r.split == "train"]
    chosen = [r for r in records if r.clip_id in {"X-1", "X-3", "X-4", "X-5", "X-11", "X-12", "X-18", "X-21"}]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 3*len(chosen)), squeeze=False)
    for axes_row, record in zip(axes, chosen):
        center = record.event_times[0] if record.event_times else record.last_time / 2
        frames = decode_frames(record, sample_times(center, record.last_time, 4., 16))
        axes_row[0].imshow(frames[8].permute(1, 2, 0).numpy())
        axes_row[0].set_title(f"TRAIN {record.clip_id} label={record.label}, raw")
        for ax, roi in zip(axes_row[1:], (None, [0., .1, 1., .7])):
            transform = video_transform(dict(weights="KINETICS400_V1", preprocessing="aspect_letterbox_v1", roi=roi))
            tensor = transform(frames)
            mean = torch.tensor(transform.mean).view(3,1,1,1)
            std = torch.tensor(transform.std).view(3,1,1,1)
            ax.imshow((tensor*std+mean)[:,8].clamp(0,1).permute(1,2,0).numpy(), interpolation="nearest")
            ax.set_title("Full frame, preserve aspect" if roi is None else "Work-area ROI, preserve aspect")
        for ax in axes_row: ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "train_roi_review.jpg", dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--baseline", type=Path, default=ROOT / "runs/r3d18_notebook_20261017T223905_499373Z/best.pt")
    parser.add_argument("--resume", type=Path, help="Existing study directory; skip only fully completed runs")
    parser.add_argument("--baseline-summary", type=Path, help="Reuse an already verified full-video baseline report")
    args = parser.parse_args()
    if args.inspect_only:
        output = ROOT / "data/local_artifacts" / "improvements_train_roi"
        output.mkdir(parents=True, exist_ok=True)
        inspect_train_roi(output)
        print(output / "train_roi_review.jpg")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This planned experiment requires local CUDA; no silent CPU fallback")
    study = args.resume or ROOT / "runs" / ("binary_improvements_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    definitions = [("negatives", "r3d18_negatives.yaml"), ("geometry", "r3d18_geometry.yaml"), ("roi", "r3d18_roi.yaml")]
    configurations = []
    for name, filename in definitions:
        config = yaml.safe_load((ROOT / "configs" / filename).read_text(encoding="utf-8"))
        config["output_dir"] = str(study / name)
        configurations.append(dict(name=name, config=config))
    stable = dict(configurations[0]["config"], negative_windows=1, negative_sampling="interior",
                  loss_weighting="none", output_dir=str(study / "control_stable"))
    configurations.insert(0, dict(name="control_stable", config=stable))
    wide = dict(configurations[-1]["config"], window_sec=8., output_dir=str(study / "roi_context8"))
    configurations.append(dict(name="roi_context8", config=wide))
    plan = dict(schema="binary_improvements_v2", events_sha256=sha256(ROOT / "data/annotations/events.csv"),
                split_sha256=sha256(ROOT / "data/processed/split_proposal.csv"), baseline_checkpoint=str(args.baseline),
                baseline_sha256=sha256(args.baseline), experiments=configurations,
                fixed_threshold=.5, temporal_tolerance_sec=5., independent_evaluation=False,
                baseline_summary=str(args.baseline_summary) if args.baseline_summary else None,
                implementation_sha256={p.name: sha256(p) for p in (ROOT / "src/overfitted_underqualified/ml").glob("*.py")},
                note="Known validation errors informed hypotheses. Stable head control plus four preset ablations; no threshold optimization or relabeling. Earlier failed batch-2 attempt preserved separately.")
    if args.resume:
        saved = json.loads((study / "plan.json").read_text(encoding="utf-8"))
        if saved != plan: raise ValueError("Study plan changed; use a new study")
    else:
        study.mkdir(parents=True, exist_ok=False)
        write_json(study / "plan.json", plan)
    summary = []
    experiments = [dict(name="baseline", checkpoint=args.baseline, config=dict(window_sec=4.))] + configurations
    for item in experiments:
        name = item["name"]
        report_path = study / (name + "_summary.json")
        if report_path.exists():
            summary.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue
        print("EXPERIMENT", name, flush=True)
        if name == "baseline" and args.baseline_summary:
            row = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
            provenance = json.loads((Path(row["report_dir"]) / "evaluation.json").read_text(encoding="utf-8"))
            if row["checkpoint_sha256"] != plan["baseline_sha256"] or any(
                provenance["data_manifest"][key] != plan[key] for key in ("events_sha256", "split_sha256")):
                raise ValueError("Reused baseline differs from study inputs")
            row["reused_report"] = True
            write_json(report_path, row)
            summary.append(row)
            write_json(study / "comparison.json", summary)
            continue
        if name == "baseline":
            checkpoint = args.baseline
        else:
            run_dir = Path(item["config"]["output_dir"])
            if run_dir.exists():
                raise FileExistsError(f"Unfinished run, preserve it and start a new study: {run_dir}")
            checkpoint = run(item["config"]) / "best.pt"
        evaluation = prepare_evaluation(checkpoint, device="cuda")
        metrics, rows = evaluate_prepared(evaluation)
        window_summary = aggregate_video_predictions(rows)
        del evaluation
        torch.cuda.empty_cache()
        scan = ScanConfig(window_sec=item["config"]["window_sec"], step_sec=2., batch_size=2, threshold=.5)
        full = prepare_full_evaluation(checkpoint, scan, device="cuda")
        result = evaluate_full_prepared(full, tolerance_sec=5.)
        report_dir = ROOT / "reports" / ("evaluation_improvements_" + study.name + "_" + name)
        saved = save_full_report(full, result, report_dir)
        archive = archive_full_report(report_dir)
        row = dict(name=name, checkpoint=str(checkpoint), checkpoint_sha256=saved["checkpoint_sha256"],
                   checkpoint_epoch=saved["checkpoint_epoch"], window_metrics=metrics,
                   full_metrics=saved["metrics"], timing=saved["timing"], report_dir=str(report_dir), archive=str(archive))
        write_json(report_path, row)
        summary.append(row)
        write_json(study / "comparison.json", summary)
        print("RESULT", json.dumps(row), flush=True)
        del full, result
        torch.cuda.empty_cache()
    print("STUDY", study, flush=True)


if __name__ == "__main__": main()
