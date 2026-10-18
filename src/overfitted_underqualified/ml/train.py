"""Train/evaluate R3D-18 on the fixed within-room split.

Run from repository root: python -m overfitted_underqualified.ml.train --config configs/r3d18.yaml
with PYTHONPATH=src (see docs/r3d18-baseline.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import R3D_18_Weights, r3d_18

from .dataset import LABELS, VideoWindowDataset, load_records, sha256
from .preprocessing import video_transform
from .features import WeightedWindows, cache_frozen_loaders

ROOT = Path(__file__).resolve().parents[3]


def seed_everything(seed: int, threads: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_model(weights_name: str | None, freeze_backbone: bool) -> nn.Module:
    weights = R3D_18_Weights[weights_name] if weights_name else None
    model = r3d_18(weights=weights)
    if freeze_backbone:
        model.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def set_training_mode(model: nn.Module, freeze_backbone: bool):
    # Frozen BatchNorm statistics must not drift while training the head.
    model.eval() if freeze_backbone else model.train()
    model.fc.train()


def run_epoch(model, loader, device, optimizer=None, freeze_backbone=True):
    training = optimizer is not None
    if training:
        set_training_mode(model, freeze_backbone)
    else:
        model.eval()
    loss_sum = 0.0
    rows = []
    with torch.set_grad_enabled(training):
        for batch in loader:
            if "features" in batch and training and not freeze_backbone:
                raise ValueError("Cached features cannot train the backbone")
            inputs = batch["features" if "features" in batch else "video"].to(device)
            labels = batch["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model.fc(inputs) if "features" in batch else model(inputs)
            losses = nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = (losses * batch["sample_weight"].to(device)).mean() if training and "sample_weight" in batch else losses.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            probabilities = logits.detach().softmax(dim=1)[:, 1].cpu().tolist()
            for clip_id, center, label, prob in zip(batch["clip_id"], batch["center_sec"].tolist(),
                                                   labels.cpu().tolist(), probabilities):
                rows.append({"clip_id": clip_id, "center_sec": center, "label": label,
                             "probability": prob, "prediction": int(prob >= 0.5)})
    if not rows:
        raise ValueError("Empty loader")
    truth = [r["label"] for r in rows]
    pred = [r["prediction"] for r in rows]
    metrics = {"loss": loss_sum / len(rows), "windows": len(rows),
               "accuracy": float(accuracy_score(truth, pred)),
               "precision": float(precision_score(truth, pred, zero_division=0)),
               "recall": float(recall_score(truth, pred, zero_division=0)),
               "f1": float(f1_score(truth, pred, zero_division=0)),
               "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1]).tolist()}
    return metrics, rows


def save_checkpoint(path, model, optimizer, epoch, config, manifest, metrics):
    torch.save({"format_version": 1, "architecture": "r3d_18", "class_to_idx": LABELS,
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "config": config,
                "data_manifest": manifest, "metrics": metrics}, path)


def load_checkpoint(path: Path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != "r3d_18" or checkpoint["class_to_idx"] != LABELS:
        raise ValueError("Incompatible checkpoint")
    model = build_model(None, checkpoint["config"]["freeze_backbone"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def prepare_training(config: dict) -> dict:
    config = dict(config)
    transform = video_transform(config)
    if config.get("loss_weighting", "none") not in {"none", "video_class_balanced"}:
        raise ValueError("Unknown loss weighting policy")
    if config.get("cache_frozen_features", False) and not config["freeze_backbone"]:
        raise ValueError("Feature cache requires freeze_backbone=true")
    if config["weights"] != "KINETICS400_V1":
        raise ValueError("Baseline requires pretrained KINETICS400_V1 weights")
    if not isinstance(config["freeze_backbone"], bool):
        raise ValueError("freeze_backbone must be boolean")
    for field in ("epochs", "batch_size", "cpu_threads", "negative_windows", "validation_negative_windows"):
        if field == "validation_negative_windows" and field not in config:
            continue
        if not isinstance(config[field], int) or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if config["num_workers"] < 0 or config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid loader or optimizer settings")
    if config["frames"] != 16 or not np.isfinite(config["window_sec"]) or config["window_sec"] <= 0:
        raise ValueError("Use 16 frames and a positive finite window_sec")
    for key in ("events_path", "split_path", "video_dir", "output_dir", "weights_cache"):
        path = Path(config[key])
        config[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory to preserve earlier runs: {output}")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    seed_everything(config["seed"], config["cpu_threads"])
    torch.hub.set_dir(config["weights_cache"])
    records = load_records(Path(config["events_path"]), Path(config["split_path"]), Path(config["video_dir"]))
    datasets = {subset: VideoWindowDataset(records, subset, transform, config["window_sec"],
                                          config["frames"], config.get("validation_negative_windows", config["negative_windows"])
                                          if subset == "validation" else config["negative_windows"],
                                          negative_sampling=config.get("negative_sampling", "interior"))
                for subset in ("train", "validation")}
    supplemental = None
    if config.get("supplemental_manifest"):
        from .augmented_data import read_audit, AugmentedBinaryWindows
        audit_path = Path(config["supplemental_manifest"])
        audit_path = audit_path if audit_path.is_absolute() else ROOT / audit_path
        config["supplemental_manifest"] = str(audit_path.resolve())
        audit = read_audit(audit_path, config["supplemental_manifest_sha256"],
                          config["events_path"], config["split_path"])
        datasets["train"] = AugmentedBinaryWindows(datasets["train"], audit, "train")
        supplemental = dict(audit_sha256=config["supplemental_manifest_sha256"],
                            training_sources=datasets["train"].source_manifest,
                            validation_policy="Original validation only during training; descendants evaluated separately")
    loaders = {subset: DataLoader(WeightedWindows(dataset) if subset == "train" and config.get("loss_weighting") == "video_class_balanced" else dataset,
                                 batch_size=config["batch_size"], shuffle=subset == "train",
                                 num_workers=config["num_workers"], worker_init_fn=seed_worker,
                                 generator=torch.Generator().manual_seed(config["seed"]),
                                 pin_memory=device.type == "cuda") for subset, dataset in datasets.items()}
    manifest = {"events_sha256": sha256(Path(config["events_path"])),
                "split_sha256": sha256(Path(config["split_path"])),
                "evaluation": "annotation-centred window classification; provisional within-room split",
                "independent_evaluation": False,
                "videos": [{"clip_id": r.clip_id, "split": r.split, "sha256": r.sha256,
                            "source_clip_ids": list(r.aliases), "event_times": list(r.event_times),
                            "label": r.label} for r in records],
                "windows": {subset: [{"clip_id": r.clip_id, "center_sec": t, "label": r.label}
                                      for r, t in dataset.samples] for subset, dataset in datasets.items()}}
    if supplemental is not None:
        manifest["supplemental"] = supplemental
    return dict(config=config, output=output, device=device, records=records,
                datasets=datasets, loaders=loaders, manifest=manifest)


def train_prepared(prepared: dict, model: nn.Module) -> Path:
    """Train an explicitly loaded model using validated data; never reuse a populated run directory."""
    config, output, device = (prepared[key] for key in ("config", "output", "device"))
    loaders, manifest = prepared["loaders"], prepared["manifest"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    for key, manifest_key in (("events_path", "events_sha256"), ("split_path", "split_sha256")):
        if sha256(Path(config[key])) != manifest[manifest_key]:
            raise ValueError(f"Input changed after preparation: {key}")
    model.to(device)
    if config.get("cache_frozen_features", False):
        print("Caching frozen features once per run (train/validation remain separate)", flush=True)
        loaders = cache_frozen_loaders(model, loaders, device, config["batch_size"], config["seed"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["learning_rate"], weight_decay=config["weight_decay"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "torchvision": str(torchvision.__version__), "cuda_runtime": torch.version.cuda,
                   "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "preprocessing": str(video_transform(config)), "seed": config["seed"],
                   "feature_cache": "per_run_cpu_memory" if config.get("cache_frozen_features") else "disabled"}
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    history = []
    best_loss = float("inf")
    for epoch in range(1, config["epochs"] + 1):
        train_metrics, _ = run_epoch(model, loaders["train"], device, optimizer, config["freeze_backbone"])
        val_metrics, predictions = run_epoch(model, loaders["validation"], device)
        metrics = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(metrics)
        save_checkpoint(output / "last.pt", model, optimizer, epoch, config, manifest, metrics)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(output / "best.pt", model, optimizer, epoch, config, manifest, metrics)
            with (output / "best_validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(metrics), flush=True)
    return output


def run(config: dict) -> Path:
    prepared = prepare_training(config)
    model = build_model(prepared["config"]["weights"], prepared["config"]["freeze_backbone"])
    return train_prepared(prepared, model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r3d18.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--video-dir")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("epochs", "device", "output_dir", "video_dir"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    print(f"Saved experiment: {run(config)}")


if __name__ == "__main__":
    main()
