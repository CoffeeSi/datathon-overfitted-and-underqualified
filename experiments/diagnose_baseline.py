"""Local-only crop/context diagnostics; no training and no threshold selection.

Run with PYTHONPATH=src. Images go to ignored data/local_artifacts/, never notebook outputs.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights

from overfitted_underqualified.ml.dataset import decode_frames, load_records, sample_times, sha256
from overfitted_underqualified.ml.train import ROOT, load_checkpoint, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / "data/local_artifacts" / ("baseline_diagnostics_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.full_report) as archive:
        name = next(n for n in archive.namelist() if n.endswith("/window_predictions.csv"))
        scan_windows = list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))
    records = load_records(ROOT / "data/annotations/events.csv", ROOT / "data/processed/split_proposal.csv", ROOT / "data/raw/videos")
    selected = [r for r in records if r.clip_id in {"X-14", "X-16", "X-31", "X-39", "X-34", "X-44"}]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    seed_everything(checkpoint["config"]["seed"], checkpoint["config"]["cpu_threads"])
    transform = R3D_18_Weights.KINETICS400_V1.transforms()
    mean = torch.tensor(transform.mean).view(3, 1, 1, 1)
    std = torch.tensor(transform.std).view(3, 1, 1, 1)
    observations, probes, appearance_controls = [], [], []
    for record in selected:
        windows = [w for w in scan_windows if w["clip_id"] == record.clip_id]
        peak = max(windows, key=lambda w: float(w["probability"]))
        center = float(peak["center_sec"])
        reference = record.event_times[0] if record.event_times else record.last_time / 2
        fig, axes = plt.subplots(6, 4, figsize=(16, 15), squeeze=False)
        row_index = 0
        for title, target, span in (("peak", center, 4.), ("annotation" if record.label else "midpoint", reference, 4.),
                                    ("wide-context", reference, 12.)):
            stamps = sample_times(target, record.last_time, span, 16)
            frames = decode_frames(record, stamps)
            processed = transform(frames)
            crop = (processed * std + mean).clamp(0, 1)
            height, width = frames.shape[-2:]
            # torchvision resize is exactly H=128, W=171, then central 112x112.
            left, top = round((171 - 112) / 2), round((128 - 112) / 2)
            x, y, cw, ch = left / 171 * width, top / 128 * height, 112 / 171 * width, 112 / 128 * height
            for column, index in enumerate((0, 5, 10, 15)):
                ax = axes[row_index, column]
                ax.imshow(frames[index].permute(1, 2, 0).numpy())
                ax.add_patch(Rectangle((x, y), cw, ch, fill=False, edgecolor="yellow", linewidth=1.5))
                ax.set_title(f"{title}: raw {stamps[index]:.2f}s; yellow=model FOV", fontsize=9)
                axes[row_index + 1, column].imshow(crop[:, index].permute(1, 2, 0).numpy(), interpolation="nearest")
                axes[row_index + 1, column].set_title("actual model pixels: 112 x 112", fontsize=9)
            row_index += 2
        for ax in axes.flat: ax.axis("off")
        fig.suptitle(f"{record.clip_id} | source binary={record.label} | peak={center:.2f}s p={float(peak['probability']):.3f} | reference={reference:.2f}s", fontsize=14)
        fig.tight_layout()
        fig.savefig(output / f"{record.clip_id}_crop_context.jpg", dpi=120)
        plt.close(fig)
        observation = dict(clip_id=record.clip_id, source_label=record.label, peak_window=peak,
                           reference_sec=reference, source_dimensions=[width, height],
                           crop_bounds_source=[x, y, cw, ch],
                           reference_windows=[w for w in windows if abs(float(w["center_sec"]) - reference) <= 4])
        observations.append(observation)
        print(json.dumps({k: observation[k] for k in ("clip_id", "reference_sec", "source_dimensions", "crop_bounds_source")}), flush=True)
        # Exploratory sensitivity only: same checkpoint, no retraining. Wider spans
        # change motion sampling and do not establish an improved validation score.
        for tag, target in (("peak", center), ("reference", reference)):
            for span in (2., 4., 8., 12.):
                stamps = sample_times(target, record.last_time, span, 16)
                frames = decode_frames(record, stamps)
                with torch.inference_mode():
                    score = float(model(transform(frames).unsqueeze(0).to(device)).softmax(1)[0, 1])
                    if span == 4.:
                        static = frames[8:9].repeat(16, 1, 1, 1)
                        static_score = float(model(transform(static).unsqueeze(0).to(device)).softmax(1)[0, 1])
                        reverse_score = float(model(transform(frames.flip(0)).unsqueeze(0).to(device)).softmax(1)[0, 1])
                        appearance_controls.append(dict(clip_id=record.clip_id, anchor=tag,
                            original_probability=score, static_probability=static_score,
                            reversed_probability=reverse_score))
                probes.append(dict(clip_id=record.clip_id, anchor=tag, requested_center=target,
                                   start_sec=float(stamps[0]), end_sec=float(stamps[-1]), window_sec=span,
                                   probability=score, sampling_hz=15 / (stamps[-1] - stamps[0])))
        print(record.clip_id, "done", flush=True)
    result = dict(checkpoint_sha256=sha256(args.checkpoint), checkpoint_epoch=checkpoint["epoch"],
                  config=checkpoint["config"], device=device, preprocessing=str(transform), observations=observations,
                  context_probes=probes, appearance_controls=appearance_controls,
                  limitations="Exploratory on known validation errors; no new quality estimate or threshold tuning. Static/reverse inputs are distribution shifts, not a causal attribution proof.")
    (output / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OUTPUT", output, flush=True)


if __name__ == "__main__":
    main()
