"""Evaluate saved weights on the fixed validation windows, with a fixed 0.5 threshold.

Paths may be relocated explicitly; annotation/split hashes must still match the run.
No pretrained weights are downloaded, and no optimizer is constructed.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.video import R3D_18_Weights

from .dataset import VideoWindowDataset, load_records, sha256
from .train import ROOT, load_checkpoint, run_epoch, seed_everything


def prepare_evaluation(checkpoint_path, *, device="cpu", events_path=None,
                       split_path=None, video_dir=None):
    model, checkpoint = load_checkpoint(Path(checkpoint_path), device)
    config = dict(checkpoint["config"])
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
                loader=loader, device=torch.device(device))


def evaluate_prepared(prepared):
    """Inference only. run_epoch sets eval mode and disables gradient recording."""
    return run_epoch(prepared["model"], prepared["loader"], prepared["device"])


def evaluate_checkpoint(checkpoint_path, **kwargs):
    return evaluate_prepared(prepare_evaluation(checkpoint_path, **kwargs))
