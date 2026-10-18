"""Small supervised counter-banknote segmenter; boxes come only from reviewed frames.

This is a visual primitive, not a legal violation classifier. Bounding rectangles
are weak segmentation targets. A missing detection does not certify an empty or
unoccluded counter. Training uses the fixed train parents and a fixed last epoch.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from .dataset import VideoInfo, decode_frames, load_records, sha256
from .primitive_data import PrimitiveSample, validate_review
from .train import ROOT, seed_everything

SCHEMA = "counter_banknotes_v1"
ARCHITECTURE = "tiny_unet_groupnorm_v1"
PROTOCOL_V1 = "weighted_bce_dice_v1"
PROTOCOL_V2 = "balanced_bce_dice_v2"
PROTOCOL_V3 = "hard_negative_bce_dice_v3"
ARCHITECTURES = {PROTOCOL_V1: ARCHITECTURE, PROTOCOL_V2: "tiny_unet_groupnorm_leaky_v2",
                 PROTOCOL_V3: "tiny_unet_groupnorm_leaky_hard_negative_v3"}
HARD_NEGATIVE_RATIO = 3
HARD_NEGATIVE_MIN_FRACTION = .01
DEFINITION = "Visible banknotes resting on the counter; not papers, cards, or notes only in hands/machine."


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("events_path", "split_path", "video_dir", "output_dir", "supplemental_manifest", "initialize_checkpoint"):
        if config.get(key):
            config[key] = str((ROOT / config[key]).resolve())
    if config.get("training_protocol") == PROTOCOL_V3:
        config.setdefault("hard_negative_ratio", HARD_NEGATIVE_RATIO)
        config.setdefault("hard_negative_min_fraction", HARD_NEGATIVE_MIN_FRACTION)
    validate_config(config)
    return config


def validate_config(config):
    if config.get("schema") != SCHEMA or config.get("resume"):
        raise ValueError("Use a new counter_banknotes_v1 experiment; resume is unsupported")
    if config.get("training_protocol", PROTOCOL_V1) not in ARCHITECTURES:
        raise ValueError("Unsupported banknote optimization protocol")
    if config.get("training_protocol") == PROTOCOL_V3:
        if (config.get("hard_negative_ratio", HARD_NEGATIVE_RATIO) != HARD_NEGATIVE_RATIO
                or config.get("hard_negative_min_fraction", HARD_NEGATIVE_MIN_FRACTION) != HARD_NEGATIVE_MIN_FRACTION):
            raise ValueError("The v3 protocol fixes hard-negative ratio=3 and minimum fraction=.01")
    if bool(config.get("initialize_checkpoint")) != bool(config.get("initialize_checkpoint_sha256")):
        raise ValueError("Initialization requires both a checkpoint path and exact hash")
    if config.get("initialize_checkpoint") and config.get("training_protocol") != PROTOCOL_V3:
        raise ValueError("Only the explicit v2-to-v3 banknote initialization is supported")
    roi = config.get("roi")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("ROI must contain normalized xyxy coordinates")
    x1, y1, x2, y2 = roi
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in roi) or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("Invalid counter ROI")
    for name in ("resize_width", "epochs", "batch_size", "cpu_threads", "min_component_area"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if config["resize_width"] < 16 or type(config.get("seed")) is not int:
        raise ValueError("Invalid image width or seed")
    if (not isinstance(config.get("threshold"), (float, int)) or not 0 < config["threshold"] < 1
            or not math.isfinite(config.get("learning_rate", float("nan"))) or config["learning_rate"] <= 0):
        raise ValueError("Invalid threshold or learning rate")
    digest_keys = ["events_sha256", "split_sha256"]
    if config.get("initialize_checkpoint"):
        digest_keys.append("initialize_checkpoint_sha256")
    for key in digest_keys:
        value = config.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid {key}")
    if config.get("supplemental_manifest") and not config.get("supplemental_manifest_sha256"):
        raise ValueError("Supplemental manifest must have an exact hash")
    variants = config.get("supplemental_variants", ["warm_sepia", "cool_camera", "cctv_low_quality"])
    if (not isinstance(variants, list) or not 1 <= len(variants) <= 3 or len(set(variants)) != len(variants)
            or not set(variants) <= {"warm_sepia", "cool_camera", "cctv_low_quality", "mirror"}):
        raise ValueError("Choose at most three explicit supplemental transforms")


def validate_boxes(point, roi):
    """Return explicit complete full-frame xyxy boxes; unknown frames are excluded."""
    if point.get("banknote_boxes_complete") is not True:
        return None
    label = point.get("labels", {}).get("banknotes_on_counter")
    boxes = point.get("banknote_boxes")
    if type(label) is not int or label not in (0, 1) or not isinstance(boxes, list):
        raise ValueError("Complete boxes require an explicit banknote target and box list")
    if bool(boxes) != bool(label):
        raise ValueError("Positive frame needs boxes; a negative frame must have none")
    result = []
    for box in boxes:
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
            raise ValueError("Banknote boxes must be finite normalized xyxy rectangles")
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Invalid banknote box bounds")
        if not (roi[0] <= x1 < x2 <= roi[2] and roi[1] <= y1 < y2 <= roi[3]):
            raise ValueError("Reviewed banknote box lies outside the configured ROI")
        result.append([x1, y1, x2, y2])
    return sorted(result)


@dataclass(frozen=True)
class BanknoteSample:
    record: object
    time_sec: float
    boxes: tuple
    source_ids: tuple


def prepare_banknote_data(config):
    validate_config(config)
    events, split = Path(config["events_path"]), Path(config["split_path"])
    if sha256(events) != config["events_sha256"] or sha256(split) != config["split_sha256"]:
        raise ValueError("Banknote source/split hash differs from the fixed experiment")
    records = load_records(events, split, Path(config["video_dir"]))
    with events.open(encoding="utf-8-sig", newline="") as stream:
        source = {row["clip_id"]: row for row in csv.DictReader(stream)}
    samples, reviews = [], []
    for record in records:
        points = {}
        for alias in record.aliases:
            raw = source[alias].get("primitive_review")
            if not raw:
                continue
            review = json.loads(raw)
            validate_review(review, record, 2.)
            for point in review["points"]:
                boxes = validate_boxes(point, config["roi"])
                if boxes is None:
                    continue
                stamp = float(point["time_sec"])
                if stamp in points and points[stamp][0] != boxes:
                    raise ValueError("Conflicting banknote boxes across canonical aliases")
                points.setdefault(stamp, (boxes, []))[1].append(alias)
                reviews.append(dict(clip_id=record.clip_id, source_id=alias,
                                    video_sha256=review["video_sha256"], reviewer=review["reviewer"],
                                    reviewed_at=review["reviewed_at"], method=review["method"], point=point))
        for stamp, (boxes, aliases) in sorted(points.items()):
            samples.append(BanknoteSample(record, stamp, tuple(tuple(box) for box in boxes), tuple(aliases)))
    manifest = dict(schema=SCHEMA, definition=DEFINITION,
                    events_sha256=config["events_sha256"], split_sha256=config["split_sha256"],
                    source_reviews=reviews,
                    samples=[dict(clip_id=s.record.clip_id, source_ids=list(s.source_ids),
                                  video_sha256=s.record.sha256, split=s.record.split,
                                  time_sec=s.time_sec, boxes=[list(b) for b in s.boxes]) for s in samples],
                    train_parents=sorted({s.record.clip_id for s in samples if s.record.split == "train"}),
                    validation_parents=sorted({s.record.clip_id for s in samples if s.record.split == "validation"}))
    if set(manifest["train_parents"]) & set(manifest["validation_parents"]):
        raise ValueError("Banknote parent leakage")
    return samples, manifest


def supplemental_sources(config, samples):
    """Resolve descendants only for reviewed train parents, never external labels."""
    if not config.get("supplemental_manifest"):
        return {}, None
    from .supplemental_data import resolve_augmentations
    path = Path(config["supplemental_manifest"])
    if sha256(path) != config["supplemental_manifest_sha256"]:
        raise ValueError("Supplemental manifest differs from the pinned experiment")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    chosen = config.get("supplemental_variants", ["warm_sepia", "cool_camera", "cctv_low_quality"])
    sources, provenance = {}, []
    for sample in samples:
        if sample.record.split != "train" or sample.record.clip_id in sources:
            continue
        reviewed = PrimitiveSample(sample.record, sample.time_sec,
                                   ("banknotes_on_counter",) if sample.boxes else (),
                                   () if sample.boxes else ("banknotes_on_counter",))
        variants = resolve_augmentations(manifest, reviewed, events_path=config["events_path"],
                                         split_path=config["split_path"], inverse_mirror=True)
        sources[sample.record.clip_id] = sorted((v for v in variants if v.augmentation in chosen),
                                                key=lambda v: chosen.index(v.augmentation))
        provenance.extend(dict(parent_id=v.parent_id, parent_sha256=sample.record.sha256,
                               augmentation=v.augmentation, path=str(v.path), sha256=v.sha256,
                               inverse_horizontal_flip=v.inverse_horizontal_flip)
                          for v in sources[sample.record.clip_id])
    return sources, dict(manifest_path=str(path), manifest_sha256=sha256(path), sources=provenance,
                         selection="Uniform original-or-variant per reviewed train point each epoch; no parent multiplication",
                         targets="Inherited original reviewed boxes and timestamps; external CSV/workbook labels unused")


def prepare_frame(rgb, config, boxes=()):
    """Crop/resize once at useful spatial resolution; retain exact inverse geometry."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 HWC frame")
    height, width = rgb.shape[:2]
    if min(height, width) < 2:
        raise ValueError("Frame is too small")
    roi = config["roi"]
    x1, y1 = int(math.floor(roi[0] * width)), int(math.floor(roi[1] * height))
    x2, y2 = int(math.ceil(roi[2] * width)), int(math.ceil(roi[3] * height))
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Empty counter crop")
    out_width = config["resize_width"]
    out_height = max(1, round((y2 - y1) * out_width / (x2 - x1)))
    crop = cv2.resize(rgb[y1:y2, x1:x2], (out_width, out_height), interpolation=cv2.INTER_LINEAR)
    target = np.zeros((out_height, out_width), np.float32)
    for bx1, by1, bx2, by2 in boxes:
        left = max(0, math.floor((bx1 * width - x1) * out_width / (x2 - x1)))
        right = min(out_width, math.ceil((bx2 * width - x1) * out_width / (x2 - x1)))
        top = max(0, math.floor((by1 * height - y1) * out_height / (y2 - y1)))
        bottom = min(out_height, math.ceil((by2 * height - y1) * out_height / (y2 - y1)))
        if left >= right or top >= bottom:
            raise ValueError("Banknote box disappears during preprocessing")
        target[top:bottom, left:right] = 1.
    pad_bottom, pad_right = (-out_height) % 4, (-out_width) % 4
    image = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
    valid = np.pad(np.ones_like(target), ((0, pad_bottom), (0, pad_right)))
    target = np.pad(target, ((0, pad_bottom), (0, pad_right)))
    return dict(image=torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255),
                mask=torch.from_numpy(target)[None], valid=torch.from_numpy(valid)[None],
                geometry=dict(frame_height=height, frame_width=width, crop=[x1, y1, x2, y2],
                              resized_height=out_height, resized_width=out_width))


class BanknoteFrames(Dataset):
    """Decoded in memory once; augmentation only belongs to train frames."""
    def __init__(self, samples, config, *, augment=False, variants=None):
        if augment and any(s.record.split != "train" for s in samples):
            raise ValueError("Validation frames cannot enter augmented training")
        self.samples, self.augment = samples, augment
        self.frames, self.variants = [], []
        for sample in samples:
            rgb = decode_frames(sample.record, np.asarray([sample.time_sec]))[0].permute(1, 2, 0).numpy()
            self.frames.append(prepare_frame(rgb, config, sample.boxes))
            choices = []
            for source in (variants or {}).get(sample.record.clip_id, []):
                if not augment or source.parent_id != sample.record.clip_id:
                    raise ValueError("Supplemental frame must inherit a reviewed training parent")
                record = VideoInfo(sample.record.clip_id, source.path, sample.record.fps, sample.record.frame_count)
                # Invert mirror in TCHW space, before HWC conversion and fixed ROI.
                frames = source.restore_orientation(decode_frames(record, np.asarray([sample.time_sec])))
                transformed = frames[0].permute(1, 2, 0).numpy()
                if transformed.shape != rgb.shape:
                    raise ValueError("Augmentation geometry differs from reviewed parent")
                choices.append(prepare_frame(transformed, config, sample.boxes))
            self.variants.append(choices)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        value = self.frames[index]
        if self.augment and self.variants[index]:
            choice = int(torch.randint(0, len(self.variants[index]) + 1, ()))
            if choice:
                value = self.variants[index][choice - 1]
        image, mask, valid = value["image"].clone(), value["mask"].clone(), value["valid"].clone()
        if self.augment:
            if torch.rand(()) < .5:
                image, mask, valid = image.flip(-1), mask.flip(-1), valid.flip(-1)
            # Mild camera/light variation; positions and masks remain aligned.
            image = (image * (.85 + torch.rand(()) * .30) + (torch.rand(()) - .5) * .08).clamp_(0, 1)
        return dict(image=image, mask=mask, valid=valid, index=index)


def collate_frames(items):
    height = max(item["image"].shape[-2] for item in items)
    width = max(item["image"].shape[-1] for item in items)
    result = {}
    for key in ("image", "mask", "valid"):
        result[key] = torch.stack([nn.functional.pad(item[key], (0, width - item[key].shape[-1],
                                                                0, height - item[key].shape[-2])) for item in items])
    result["index"] = [item["index"] for item in items]
    return result


def _block(inputs, outputs, protocol):
    activation = nn.ReLU if protocol == PROTOCOL_V1 else nn.LeakyReLU
    activation_args = {} if protocol == PROTOCOL_V1 else {"negative_slope": .1}
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), activation(**activation_args),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), activation(**activation_args))


class TinyBanknoteUNet(nn.Module):
    def __init__(self, protocol=PROTOCOL_V1):
        super().__init__()
        if protocol not in ARCHITECTURES:
            raise ValueError("Unsupported banknote architecture protocol")
        self.first, self.second, self.center = _block(3, 16, protocol), _block(16, 32, protocol), _block(32, 64, protocol)
        self.up_second, self.up_first = _block(96, 32, protocol), _block(48, 16, protocol)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, value):
        first = self.first(value)
        second = self.second(nn.functional.max_pool2d(first, 2))
        center = self.center(nn.functional.max_pool2d(second, 2))
        second = self.up_second(torch.cat([nn.functional.interpolate(center, size=second.shape[-2:], mode="nearest"), second], dim=1))
        first = self.up_first(torch.cat([nn.functional.interpolate(second, size=first.shape[-2:], mode="nearest"), first], dim=1))
        return self.output(first)


def segmentation_loss(logits, target, valid, positive_weight, protocol=PROTOCOL_V1):
    if protocol in (PROTOCOL_V2, PROTOCOL_V3):
        # Equal class mass prevents tiny reviewed rectangles from disappearing
        # under tens of thousands of valid background pixels. Unknown/pad pixels
        # never enter either average. Empty frames contribute background BCE.
        losses = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        positive, negative = target * valid, (1 - target) * valid
        positive_bce = (losses * positive).sum() / positive.sum().clamp_min(1)
        if protocol == PROTOCOL_V3:
            # The easy empty counter must not dilute rare document/hand mistakes.
            # The training gradient uses reviewed train masks only; validation
            # may report this same loss under disabled gradients, without updates.
            negative_losses = losses[negative > 0]
            count = negative_losses.numel()
            k = min(count, max(math.ceil(HARD_NEGATIVE_RATIO * float(positive.sum().detach())),
                               math.ceil(HARD_NEGATIVE_MIN_FRACTION * count)))
            negative_bce = negative_losses.topk(k).values.mean() if k else logits.sum() * 0
        else:
            negative_bce = (losses * negative).sum() / negative.sum().clamp_min(1)
        bce = .5 * (positive_bce + negative_bce)
        probabilities = logits.sigmoid() * valid
        positive_pixels = positive.sum((1, 2, 3))
        has_positive = positive_pixels > 0
        intersection = (probabilities * target).sum((1, 2, 3))
        dice = 1 - (2 * intersection + 1) / (probabilities.sum((1, 2, 3)) + positive_pixels + 1)
        return bce + (dice * has_positive).sum() / has_positive.sum().clamp_min(1)
    if protocol != PROTOCOL_V1:
        raise ValueError("Unsupported banknote loss protocol")
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                                        pos_weight=logits.new_tensor(positive_weight))
    bce = (loss * valid).sum() / valid.sum().clamp_min(1)
    probabilities = logits.sigmoid() * valid
    intersection = (probabilities * target).sum((1, 2, 3))
    dice = 1 - (2 * intersection + 1) / (probabilities.sum((1, 2, 3)) + (target * valid).sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def extract_objects(probabilities, geometry, threshold, min_component_area):
    """Connected segmentation components in full-frame normalized coordinates."""
    height, width = geometry["resized_height"], geometry["resized_width"]
    values = np.asarray(probabilities, dtype=np.float32)[:height, :width]
    if values.shape != (height, width) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid banknote probability map")
    count, components, stats, _ = cv2.connectedComponentsWithStats((values >= threshold).astype(np.uint8), connectivity=8)
    x1, y1, x2, y2 = geometry["crop"]
    objects = []
    for index in range(1, count):
        left, top, size_x, size_y, area = stats[index].tolist()
        if area < min_component_area:
            continue
        box = [(x1 + left / width * (x2 - x1)) / geometry["frame_width"],
               (y1 + top / height * (y2 - y1)) / geometry["frame_height"],
               (x1 + (left + size_x) / width * (x2 - x1)) / geometry["frame_width"],
               (y1 + (top + size_y) / height * (y2 - y1)) / geometry["frame_height"]]
        objects.append(dict(box=box, score=float(values[components == index].mean()), area_px=area))
    return sorted(objects, key=lambda item: item["score"], reverse=True)


def _epoch(model, loader, device, positive_weight, optimizer=None, protocol=PROTOCOL_V1):
    training = optimizer is not None
    model.train(training)
    total, count = 0., 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            image, target, valid = (batch[k].to(device) for k in ("image", "mask", "valid"))
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(image), target, valid, positive_weight, protocol)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite banknote loss")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
            total += float(loss.detach()) * image.shape[0]
            count += image.shape[0]
    return total / count if count else None


def _model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def initialize_model(model, config, manifest):
    """Explicit train-only v2 weights -> v3 optimization, with a fresh optimizer."""
    if not config.get("initialize_checkpoint"):
        return None
    validate_config(config)
    parent = load(config["initialize_checkpoint"], device="cpu")
    if parent.checkpoint_sha256 != config["initialize_checkpoint_sha256"]:
        raise ValueError("Initialization checkpoint hash differs from the pinned parent")
    source = parent.checkpoint
    if (source["config"].get("training_protocol", PROTOCOL_V1) != PROTOCOL_V2
            or config.get("training_protocol") != PROTOCOL_V3):
        raise ValueError("Initialization only supports v2-to-v3, not arbitrary resume")
    if source["data_manifest"] != manifest:
        raise ValueError("Initialization source manifest differs from the current reviewed data")
    for key in ("roi", "resize_width", "events_sha256", "split_sha256"):
        if source["config"][key] != config[key]:
            raise ValueError("Initialization preprocessing/source differs from the fixed experiment")
    model.load_state_dict(parent.model.state_dict(), strict=True)
    return dict(checkpoint_path=str(config["initialize_checkpoint"]), checkpoint_sha256=parent.checkpoint_sha256,
                parent_architecture=source["architecture"], parent_config=source["config"],
                parent_epoch=source["epoch"], parent_data_manifest_sha256=_manifest_digest(source["data_manifest"]),
                optimizer="new AdamW; parent optimizer state is never restored",
                selection="Explicit preselected training initialization; validation never updates weights")


def _validate_initialization(config, manifest, initialization):
    if bool(initialization) != bool(config.get("initialize_checkpoint")):
        raise ValueError("Banknote initialization provenance presence mismatch")
    if initialization is None:
        return
    parent_config = initialization.get("parent_config", {})
    if (initialization.get("checkpoint_sha256") != config["initialize_checkpoint_sha256"]
            or initialization.get("parent_architecture") != ARCHITECTURES[PROTOCOL_V2]
            or parent_config.get("training_protocol", PROTOCOL_V1) != PROTOCOL_V2
            or initialization.get("parent_data_manifest_sha256") != _manifest_digest(manifest)):
        raise ValueError("Banknote initialization provenance mismatch")
    if any(parent_config.get(key) != config[key] for key in ("roi", "resize_width", "events_sha256", "split_sha256")):
        raise ValueError("Banknote initialization preprocessing/source mismatch")
    if initialization.get("parent_epoch") != parent_config.get("epochs"):
        raise ValueError("Banknote initialization was not a fixed final parent epoch")


def train(config):
    config = dict(config)
    if config.get("training_protocol") == PROTOCOL_V3:
        config.setdefault("hard_negative_ratio", HARD_NEGATIVE_RATIO)
        config.setdefault("hard_negative_min_fraction", HARD_NEGATIVE_MIN_FRACTION)
    validate_config(config)
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError("Use a fresh banknote experiment directory")
    samples, manifest = prepare_banknote_data(config)
    training = [sample for sample in samples if sample.record.split == "train"]
    validation = [sample for sample in samples if sample.record.split == "validation"]
    if not training or not any(s.boxes for s in training) or not any(not s.boxes for s in training):
        raise ValueError("Banknote training requires reviewed positive AND negative train frames")
    variants, supplemental = supplemental_sources(config, training)
    if supplemental is not None:
        manifest["supplemental"] = supplemental
    seed_everything(config["seed"], config["cpu_threads"])
    device = torch.device(config.get("device", "cpu"))
    started = time.perf_counter()
    datasets = dict(train=BanknoteFrames(training, config, augment=True, variants=variants),
                    validation=BanknoteFrames(validation, config))
    positives = sum(float(frame["mask"].sum()) for frame in datasets["train"].frames)
    pixels = sum(float(frame["valid"].sum()) for frame in datasets["train"].frames)
    positive_weight = min(50., max(1., (pixels - positives) / positives))
    generator = torch.Generator().manual_seed(config["seed"])
    loaders = {subset: DataLoader(dataset, batch_size=config["batch_size"], shuffle=subset == "train",
                                 num_workers=0, collate_fn=collate_frames, generator=generator)
               for subset, dataset in datasets.items()}
    protocol = config.get("training_protocol", PROTOCOL_V1)
    model = TinyBanknoteUNet(protocol).to(device)
    initialization = initialize_model(model, config, manifest)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    for path_key, digest_key in (("events_path", "events_sha256"), ("split_path", "split_sha256")):
        if sha256(Path(config[path_key])) != config[digest_key]:
            raise ValueError("Source changed during banknote decoding")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", config)
    _write_json(output / "data_manifest.json", manifest)
    history = []
    for epoch in range(1, config["epochs"] + 1):
        train_loss = _epoch(model, loaders["train"], device, positive_weight, optimizer, protocol)
        before = _model_digest(model)
        val_loss = _epoch(model, loaders["validation"], device, positive_weight, protocol=protocol)
        if before != _model_digest(model):
            raise RuntimeError("Validation changed banknote parameters or buffers")
        history.append(dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss))
        print(f"banknotes epoch {epoch}/{config['epochs']}: train={train_loss:.5f} validation={val_loss}", flush=True)
    checkpoint = dict(schema=SCHEMA, architecture=ARCHITECTURES[protocol], definition=DEFINITION,
                      model_state_dict=model.state_dict(), config=config, data_manifest=manifest,
                      epoch=config["epochs"], selection="fixed_last_epoch", history=history,
                      initialization=initialization)
    torch.save(checkpoint, output / "last.pt")
    _write_json(output / "history.json", history)
    _write_json(output / "checks.json", dict(validation_parameters_unchanged=True,
               positive_weight_train_pixels_only=positive_weight if protocol == PROTOCOL_V1 else None,
               training_protocol=protocol, train_frames=len(training),
               initialization=initialization,
               validation_frames=len(validation), total_sec=time.perf_counter() - started,
               selection="fixed last epoch; no validation threshold selection",
               torch=str(torch.__version__), device=str(device),
               gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    return output / "last.pt"


class BanknoteDetector:
    def __init__(self, model, config, *, checkpoint=None, checkpoint_sha256=None, device="cpu"):
        self.model = model.to(device).eval()
        self.config, self.checkpoint, self.checkpoint_sha256 = config, checkpoint, checkpoint_sha256
        self.device = torch.device(device)

    def predict(self, rgb):
        started = time.perf_counter()
        frame = prepare_frame(rgb, self.config)
        self.model.eval()
        with torch.inference_mode():
            probabilities = self.model(frame["image"][None].to(self.device)).sigmoid()[0, 0].cpu().numpy()
        objects = extract_objects(probabilities, frame["geometry"], self.config["threshold"], self.config["min_component_area"])
        height, width = frame["geometry"]["resized_height"], frame["geometry"]["resized_width"]
        return dict(score=max((item["score"] for item in objects), default=0.),
                    boxes=[item["box"] for item in objects], objects=objects,
                    max_pixel_probability=float(probabilities[:height, :width].max()),
                    inference_sec=time.perf_counter() - started,
                    absence_certified=False, checkpoint_sha256=self.checkpoint_sha256)


def load(path, device="cpu"):
    path = Path(path)
    digest = sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema") != SCHEMA or checkpoint.get("architecture") not in ARCHITECTURES.values()
            or checkpoint.get("definition") != DEFINITION or checkpoint.get("selection") != "fixed_last_epoch"):
        raise ValueError("Incompatible banknote checkpoint")
    config, manifest = checkpoint["config"], checkpoint["data_manifest"]
    validate_config(config)
    protocol = config.get("training_protocol", PROTOCOL_V1)
    if checkpoint["architecture"] != ARCHITECTURES[protocol]:
        raise ValueError("Banknote architecture differs from the pinned optimization protocol")
    if any(manifest[key] != config[key] for key in ("events_sha256", "split_sha256")):
        raise ValueError("Banknote checkpoint provenance mismatch")
    if set(manifest["train_parents"]) & set(manifest["validation_parents"]):
        raise ValueError("Banknote checkpoint contains leaking parents")
    supplemental = manifest.get("supplemental")
    if bool(supplemental) != bool(config.get("supplemental_manifest")):
        raise ValueError("Banknote supplemental provenance presence mismatch")
    if supplemental is not None:
        if supplemental.get("manifest_sha256") != config["supplemental_manifest_sha256"]:
            raise ValueError("Banknote supplemental manifest provenance mismatch")
        if any(source.get("parent_id") not in manifest["train_parents"] for source in supplemental.get("sources", [])):
            raise ValueError("Supplemental banknote source is not a training parent")
    if checkpoint["epoch"] != config["epochs"]:
        raise ValueError("Checkpoint is not the configured fixed final epoch")
    _validate_initialization(config, manifest, checkpoint.get("initialization"))
    model = TinyBanknoteUNet(protocol)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sha256(path) != digest:
        raise ValueError("Banknote checkpoint changed during loading")
    return BanknoteDetector(model, config, checkpoint=checkpoint, checkpoint_sha256=digest, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.train:
        print(train(config))
    else:
        samples, manifest = prepare_banknote_data(config)
        print(json.dumps(dict(frames=len(samples), train_parents=manifest["train_parents"],
                              validation_parents=manifest["validation_parents"]), indent=2))


if __name__ == "__main__":
    main()
