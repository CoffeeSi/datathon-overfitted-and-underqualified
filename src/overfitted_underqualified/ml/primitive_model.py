"""Opt-in R3D visible-fact experiment, reusing masked training and frozen feature caching."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from .action_train import action_epoch, training_positive_weights, write_json
from .dataset import sha256
from .features import cache_frozen_loaders
from .preprocessing import video_transform
from .primitive_data import CLASSES, DEFINITIONS, SCHEMA, PrimitiveDataset, PrimitiveTransform, prepare_primitive_data
from .train import ROOT, build_model, seed_everything


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("events_path", "split_path", "video_dir", "output_dir", "weights_cache"):
        config[key] = str((ROOT / config[key]).resolve())
    validate_config(config)
    return config


def validate_config(config):
    import math
    if config.get("schema") != SCHEMA or config.get("frames") != 16 or config.get("weights") != "KINETICS400_V1":
        raise ValueError("Invalid primitive architecture/preprocessing configuration")
    for name in ("epochs", "batch_size", "cpu_threads"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"Invalid {name}")
    for name in ("window_sec", "learning_rate", "scan_step_sec"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if not 0 < config["threshold"] < 1:
        raise ValueError("Invalid primitive threshold")
    if config.get("sampling") != "causal_endpoint_v1":
        raise ValueError("Primitive state labels require causal endpoint sampling")
    video_transform(config)
    PrimitiveTransform(config)
    from .cash_sequence import CashSequence
    CashSequence(**config["sequence"])
    if config["scan_step_sec"] > config["sequence"]["max_gap_sec"]:
        raise ValueError("Scan step exceeds permitted temporal gap")


class MultiViewPrimitives(nn.Module):
    """Shared frozen backbone, concatenated spatial views, one trainable linear head."""
    def __init__(self, backbone, views):
        super().__init__()
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        backbone.requires_grad_(False)
        self.backbone, self.views = backbone, views
        self.fc = nn.Linear(features * views, len(CLASSES))

    def forward(self, inputs):
        if inputs.ndim != 6 or inputs.shape[1] != self.views:
            raise ValueError("Primitive input view count differs from checkpoint")
        batch, views = inputs.shape[:2]
        features = self.backbone(inputs.reshape(batch * views, *inputs.shape[2:]))
        return self.fc(features.reshape(batch, -1))


def build_primitive_model(weights=None, *, views=1):
    model = build_model(weights, True)
    if views == 2:
        return MultiViewPrimitives(model, views)
    if views != 1:
        raise ValueError("Invalid primitive view count")
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def run_primitive_training(config):
    validate_config(config)
    _, samples, audit = prepare_primitive_data(config)
    if not audit["training_ready"]:
        raise ValueError("Each primitive needs independently reviewed train positives AND negatives")
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError("Use a fresh primitive run directory")
    seed_everything(config["seed"], config["cpu_threads"])
    device = torch.device(config["device"])
    torch.hub.set_dir(config["weights_cache"])
    model = build_primitive_model(config["weights"], views=len(config.get("view_rois", [config["roi"]]))).to(device)
    # Fingerprint all frozen tensors, including BatchNorm buffers; verify after training.
    before = backbone_digest(model)
    loaders = {subset: DataLoader(PrimitiveDataset([s for s in samples if s.record.split == subset],
                                PrimitiveTransform(config), config["window_sec"]),
                                batch_size=min(config["batch_size"], 2), num_workers=0)
               for subset in ("train", "validation")}
    if any(not len(loader.dataset) for loader in loaders.values()):
        raise ValueError("Empty primitive train/validation dataset")
    start = time.perf_counter()
    loaders = cache_frozen_loaders(model, loaders, device, config["batch_size"], config["seed"])
    feature_sec = time.perf_counter() - start
    weights = training_positive_weights(samples, CLASSES)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=config["learning_rate"])
    for path_key, digest_key in (("events_path", "events_sha256"), ("split_path", "split_sha256")):
        if sha256(Path(config[path_key])) != config[digest_key]:
            raise ValueError("Data changed during feature extraction")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "data_manifest.json", audit["manifest"])
    write_json(output / "coverage.json", audit["coverage"])
    write_json(output / "environment.json", dict(torch=str(torch.__version__), device=str(device),
               gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    history = []
    for epoch in range(1, config["epochs"] + 1):
        train, _ = action_epoch(model, loaders["train"], device, CLASSES, config["threshold"], optimizer, weights)
        validation, rows = action_epoch(model, loaders["validation"], device, CLASSES, config["threshold"])
        history.append(dict(epoch=epoch, train=train, validation=validation))
    frozen_unchanged = before == backbone_digest(model)
    if not frozen_unchanged:
        raise RuntimeError("Frozen primitive backbone or BatchNorm changed")
    checkpoint = dict(schema=SCHEMA, architecture="r3d_18_visible_primitives", class_names=list(CLASSES),
                      definitions=DEFINITIONS, config=config, model_state_dict=model.state_dict(),
                      data_manifest=audit["manifest"], epoch=config["epochs"], history=history)
    torch.save(checkpoint, output / "last.pt")
    write_json(output / "history.json", history)
    write_json(output / "validation_predictions.json", rows)
    write_json(output / "checks.json", dict(frozen_backbone_and_bn_unchanged=frozen_unchanged,
               feature_extraction_sec=feature_sec, total_sec=time.perf_counter() - start,
               positive_weights_train_only=weights.tolist(), selection="fixed last epoch; fixed thresholds"))
    return output / "last.pt"


def backbone_digest(model):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith("fc."):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_primitive_checkpoint(path, device="cpu"):
    path = Path(path)
    digest = sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema") != SCHEMA or checkpoint.get("architecture") != "r3d_18_visible_primitives"
            or checkpoint.get("class_names") != list(CLASSES) or checkpoint.get("definitions") != DEFINITIONS):
        raise ValueError("Incompatible primitive checkpoint")
    config = checkpoint["config"]
    validate_config(config)
    if any(checkpoint["data_manifest"][key] != config[key] for key in ("events_sha256", "split_sha256")):
        raise ValueError("Primitive checkpoint provenance mismatch")
    model = build_primitive_model(views=len(config.get("view_rois", [config["roi"]])))  # weights=None; no download.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sha256(path) != digest:
        raise ValueError("Checkpoint changed during loading")
    return model.to(device).eval(), checkpoint, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/primitives.yaml")
    p.add_argument("--train", action="store_true")
    args = p.parse_args()
    config = read_config(args.config)
    if args.train:
        print(run_primitive_training(config))
    else:
        _, _, audit = prepare_primitive_data(config)
        print(json.dumps({k: audit[k] for k in ("training_ready", "coverage")}, indent=2))


if __name__ == "__main__":
    main()
