"""Opt-in frozen R3D-18 multilabel training; missing labels are masked, not negatives."""
import argparse
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from .action_data import ActionDataset, SCHEMA, prepare_action_data, validate_review_manifest
from .dataset import sha256
from .train import ROOT, build_model, seed_everything, seed_worker, set_training_mode
from .preprocessing import video_transform
from .features import cache_frozen_loaders
from ..rules.catalog import CATALOG_PATH, supported_event_types, catalog_digest_matches


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("events_path", "split_path", "video_dir", "output_dir", "weights_cache"):
        config[key] = str((ROOT / config[key]).resolve())
    for key in ("epochs", "batch_size", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    for key in ("window_sec", "learning_rate"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("threshold", "background_ceiling"):
        if not math.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if config["background_ceiling"] >= config["threshold"]:
        raise ValueError("Leave an abstention band between background and positive decisions")
    if config["frames"] != 16 or config["weights"] != "KINETICS400_V1":
        raise ValueError("Expected pretrained KINETICS400_V1 preprocessing and 16 frames")
    video_transform(config)  # Validate geometry without decoding or downloading weights.
    if type(config.get("cache_frozen_features", False)) is not bool:
        raise ValueError("cache_frozen_features must be boolean")
    return config


def build_action_model(classes, weights=None):
    if not classes or len(set(classes)) != len(classes) or not set(classes) <= supported_event_types():
        raise ValueError("Invalid action class mapping")
    model = build_model(weights, True)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    return model


def action_metrics(rows, classes, threshold):
    """Per-class 2x2 matrices on known targets only. No fabricated TN/precision."""
    results = []
    for index, name in enumerate(classes):
        known = [r for r in rows if r["mask"][index]]
        tp = sum(r["targets"][index] == 1 and r["probabilities"][index] >= threshold for r in known)
        fn = sum(r["targets"][index] == 1 and r["probabilities"][index] < threshold for r in known)
        fp = sum(r["targets"][index] == 0 and r["probabilities"][index] >= threshold for r in known)
        tn = len(known) - tp - fn - fp
        evaluable = bool(tp + fn) and bool(tn + fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        results.append(dict(event_type=name, support=tp + fn, known_negatives=tn + fp,
                            unknown=len(rows) - len(known), confusion_matrix=[[tn, fp], [fn, tp]],
                            precision=precision if evaluable else None, recall=recall,
                            f1=f1 if evaluable else None, evaluable=evaluable))
    eligible = [r for r in results if r["evaluable"]]
    return dict(per_class=results, macro={k: sum(r[k] for r in eligible) / len(eligible) if eligible else None
                                          for k in ("precision", "recall", "f1")},
                macro_class_count=len(eligible), threshold=threshold)


def training_positive_weights(samples, classes):
    """Balance known train targets only; validation/unknown labels never set weights."""
    result = []
    for name in classes:
        train = [s for s in samples if s.record.split == "train"]
        positives = sum(name in s.positives for s in train)
        negatives = sum(name in s.negatives or (s.complete and name not in s.positives) for s in train)
        if not positives or not negatives:
            raise ValueError("Positive weighting requires known train positives and negatives")
        result.append(negatives / positives)
    return torch.tensor(result, dtype=torch.float32)


def action_epoch(model, loader, device, classes, threshold, optimizer=None, positive_weights=None):
    training = optimizer is not None
    set_training_mode(model, True) if training else model.eval()
    rows, total, count = [], 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            targets, mask = batch["targets"].to(device), batch["mask"].to(device)
            if "features" in batch:
                if any(p.requires_grad for name,p in model.named_parameters() if not name.startswith("fc.")):
                    raise ValueError("Cached features require a frozen backbone")
                logits = model.fc(batch["features"].to(device))
            else:
                logits = model(batch["video"].to(device))
            if logits.shape != targets.shape or not torch.isfinite(logits).all():
                raise ValueError("Invalid action logits")
            if mask.sum() > 0:
                loss = (nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none",
                    pos_weight=positive_weights.to(device) if positive_weights is not None else None
                ) * mask).sum() / mask.sum()
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.fc.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                total += float(loss.detach()) * float(mask.sum())
                count += int(mask.sum())
            for i, probabilities in enumerate(logits.detach().sigmoid().cpu().tolist()):
                rows.append(dict(clip_id=batch["clip_id"][i], center_sec=float(batch["center_sec"][i]),
                                 probabilities=probabilities, targets=batch["targets"][i].tolist(),
                                 mask=batch["mask"][i].tolist()))
    metrics = action_metrics(rows, classes, threshold)
    metrics["loss"] = total / count if count else None
    return metrics, rows


def load_action_checkpoint(path, device="cpu"):
    path = Path(path)
    digest = sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = checkpoint["class_names"]
    review = checkpoint["data_manifest"].get("review", {})
    if review.get("label_mode", "listed_actions") not in {"listed_actions", "per_type_v1"}:
        raise ValueError("Unknown action label review mode")
    if (review.get("listed_actions_verified") is not True
            or review.get("events_sha256") != checkpoint["data_manifest"].get("events_sha256")):
        raise ValueError("Action checkpoint lacks confirmed label/mapping review provenance")
    if review.get("label_mode") == "per_type_v1":
        validate_review_manifest(checkpoint["data_manifest"], classes, checkpoint["config"]["window_sec"])
    elif review.get("catalog_mapping_verified") is not True or review.get("complete_action_lists") is not True:
        raise ValueError("Action checkpoint lacks confirmed label/mapping review provenance")
    if (checkpoint.get("schema") != SCHEMA or checkpoint.get("architecture") != "r3d_18_multilabel"
            or checkpoint["class_to_idx"] != {name: i for i, name in enumerate(classes)}
            or checkpoint["config"]["weights"] != "KINETICS400_V1" or checkpoint["config"]["frames"] != 16
            or not catalog_digest_matches(checkpoint["data_manifest"]["catalog_sha256"])):
        raise ValueError("Incompatible action checkpoint, class mapping or catalog")
    video_transform(checkpoint["config"])
    requested = checkpoint["config"].get("target_types")
    if requested is not None and sorted(requested) != classes:
        raise ValueError("Action checkpoint target_types differ from class mapping")
    model = build_action_model(classes)  # weights=None: restoring never downloads weights.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sha256(path) != digest:
        raise ValueError("Action checkpoint changed during loading")
    return model.to(device).eval(), checkpoint, digest


def run_action_training(config):
    records, samples, audit = prepare_action_data(config)
    classes = audit["class_names"]
    if not classes:
        raise ValueError("Action training blocked: no reviewed training positives; complete per-type review in events.csv")
    if audit["training_blockers"]:
        raise ValueError("Action training blocked: confirmed negative examples missing for "
                         + ", ".join(audit["training_blockers"]))
    if not audit["catalog_mapping_verified"]:
        raise ValueError("Action training blocked: catalog mapping has not been verified against visible actions")
    weighting = config.get("positive_weighting", "none")
    if weighting not in {"none", "train_ratio"}:
        raise ValueError("Unknown positive weighting")
    positive_weights = training_positive_weights(samples, classes) if weighting == "train_ratio" else None
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError("Use a new action run directory")
    seed_everything(config["seed"], config["cpu_threads"])
    device = torch.device(config["device"])
    torch.hub.set_dir(config["weights_cache"])
    transform = video_transform(config)
    loaders = {subset: DataLoader(ActionDataset([s for s in samples if s.record.split == subset], classes,
                                                transform, config["window_sec"]),
                                  batch_size=config["batch_size"], shuffle=subset == "train", num_workers=0,
                                  worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config["seed"]))
               for subset in ("train", "validation")}
    model = build_action_model(classes, config["weights"]).to(device)
    feature_start = time.perf_counter()
    if config.get("cache_frozen_features", False):
        loaders = cache_frozen_loaders(model, loaders, device, config["batch_size"], config["seed"])
    feature_sec = time.perf_counter() - feature_start
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=config["learning_rate"])
    # Recheck inputs after construction before publishing any artifact.
    for key, field in (("events_path", "events_sha256"), ("split_path", "split_sha256")):
        if sha256(Path(config[key])) != audit["manifest"][field]:
            raise ValueError("Action data changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "data_manifest.json", audit["manifest"])
    write_json(output / "coverage.json", audit)
    write_json(output / "loss_settings.json", dict(positive_weighting=weighting,
               class_names=classes, train_positive_weights=positive_weights.tolist() if positive_weights is not None else None,
               validation_loss="unweighted masked BCE"))
    write_json(output / "environment.json", dict(python=platform.python_version(), torch=str(torch.__version__), device=str(device)))
    write_json(output / "preparation.json", dict(cached_features=config.get("cache_frozen_features", False),
                                                feature_extraction_sec=feature_sec, preprocessing=str(transform)))
    start, history = time.perf_counter(), []
    for epoch in range(1, config["epochs"] + 1):
        train, _ = action_epoch(model, loaders["train"], device, classes, config["threshold"], optimizer, positive_weights)
        validation, rows = action_epoch(model, loaders["validation"], device, classes, config["threshold"])
        history.append(dict(epoch=epoch, train=train, validation=validation))
        write_json(output / "history.json", history)
    # Fixed last epoch: no validation threshold/epoch selection masquerading as test.
    checkpoint = dict(schema=SCHEMA, architecture="r3d_18_multilabel", class_names=classes,
                      class_to_idx={name: i for i, name in enumerate(classes)}, config=config,
                      model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                      epoch=config["epochs"], data_manifest=audit["manifest"], history=history,
                      pretrained_origin="torchvision R3D_18_Weights.KINETICS400_V1",
                      validated_types=[r["event_type"] for r in validation["per_class"] if r["evaluable"]],
                      processing_sec=time.perf_counter() - start)
    torch.save(checkpoint, output / "last.pt")
    write_json(output / "validation_predictions.json", rows)
    return output / "last.pt"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/actions.yaml")
    parser.add_argument("--train", action="store_true", help="Explicit opt-in; review blockers still apply")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.train:
        print(run_action_training(config))
    else:
        _, _, audit = prepare_action_data(config)
        if args.audit_output:
            if args.audit_output.exists():
                raise FileExistsError(args.audit_output)
            write_json(args.audit_output, audit)
        print(json.dumps({k: audit[k] for k in ("label_mode", "reviewed_windows", "training_ready", "class_names", "training_blockers", "coverage")}, indent=2))


if __name__ == "__main__":
    main()
