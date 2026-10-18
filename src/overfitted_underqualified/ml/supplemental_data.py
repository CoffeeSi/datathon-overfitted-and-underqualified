"""Audit external videos without editing the fixed split or adopting external labels.

The JSON manifest is generated provenance, not another annotation source. Real
augmentations inherit reviewed labels only from the current events.csv parent.
Workbook descriptions remain unverified source claims; synthetic clips are never
validation examples. No model, torch, or spreadsheet-engine import is required.
"""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile


SCHEMA = "supplemental_video_audit_v1"
AUGMENTATIONS = {"mirror", "warm_sepia", "cool_camera", "cctv_low_quality"}
VIDEO_SUFFIXES = {".mov", ".mp4", ".mkv", ".avi", ".m4v"}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        result = list(reader)
    if any(None in row for row in result):
        raise ValueError("Malformed CSV row")
    return result


def _safe_file(root, name):
    if (not isinstance(name, str) or not name or name != name.strip()
            or "/" in name or "\\" in name or ":" in name or name in {".", ".."}
            or Path(name).suffix.lower() not in VIDEO_SUFFIXES):
        raise ValueError("Unsafe or unsupported video filename")
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        raise ValueError("Missing video or path outside source directory")
    return path


def inspect_metadata(path):
    """Read container timing/geometry; this does not certify visual identity."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Unreadable video")
        values = {"fps": cap.get(cv2.CAP_PROP_FPS),
                  "frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
                  "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}
    finally:
        cap.release()
    if any(not math.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("Invalid video metadata")
    for key in ("frame_count", "width", "height"):
        if values[key] != int(values[key]):
            raise ValueError("Non-integral video geometry/count")
        values[key] = int(values[key])
    values["duration_sec"] = values["frame_count"] / values["fps"]
    return values


def _same_metadata(actual, expected):
    return (all(actual[k] == expected[k] for k in ("frame_count", "width", "height"))
            and math.isclose(actual["fps"], expected["fps"], rel_tol=1e-4, abs_tol=1e-3))


def _seconds(value):
    value = str(value).strip().lower()
    if value in {"", "-", "—", "нет", "nan"}:
        return None
    minutes = re.fullmatch(r"(\d+)\s*(?:минута|минуты|минут|мин\.?|м)\s*(\d+(?:[.,]\d+)?)\s*"
                           r"(?:секунда|секунды|секунд|сек\.?|с)", value)
    if minutes:
        value = f"{minutes[1]}:{minutes[2]}"
    value = re.sub(r"\s*(секунда|секунды|секунд|сек\.?|с)\s*$", "", value)
    parts = value.replace(",", ".").split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as error:
        raise ValueError("Unrecognized timestamp") from error
    if (len(numbers) > 3 or any(not math.isfinite(n) or n < 0 for n in numbers)
            or any(n >= 60 for n in numbers[1:])):
        raise ValueError("Invalid timestamp")
    return sum(n * 60 ** i for i, n in enumerate(reversed(numbers)))


def _label(value):
    labels = {"нарушение": "violation", "без нарушения": "no_violation",
              "violation": "violation", "no_violation": "no_violation"}
    try:
        return labels[value.strip().lower()]
    except (KeyError, AttributeError) as error:
        raise ValueError("Unrecognized binary source label") from error


def _normalize_id(value):
    match = re.fullmatch(r"[XХ]-(\d+)", value, re.I)
    if not match:
        raise ValueError("Invalid source video ID")
    return f"X-{int(match[1])}"


def read_workbook(path):
    """Read cell values from XLSX XML; no formulas/macros or instructions execute."""
    tables = []
    with ZipFile(path) as archive:
        def xml(name):
            info = archive.getinfo(name)
            if info.file_size > 10 * 1024 * 1024:
                raise ValueError("Workbook XML member is unexpectedly large")
            return ET.fromstring(archive.read(name))
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(n.itertext()) for n in xml("xl/sharedStrings.xml").findall("m:si", NS)]
        names = sorted(n for n in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        for name in names:
            rows = []
            for row in xml(name).findall("m:sheetData/m:row", NS):
                values = {}
                for cell in row.findall("m:c", NS):
                    key = re.sub(r"\d", "", cell.get("r", ""))
                    node = cell.find("m:v", NS)
                    # Read cached formula results as source claims only. Timing,
                    # counts and geometry are independently checked below.
                    if cell.find("m:f", NS) is not None and node is None:
                        raise ValueError("Formula has no cached source value; execution is disabled")
                    value = node.text if node is not None else ""
                    if cell.get("t") == "inlineStr":
                        node = cell.find("m:is", NS)
                        value = "".join(node.itertext()) if node is not None else ""
                    elif cell.get("t") == "s":
                        value = strings[int(value)]
                    if value:
                        values[key] = value
                if values:
                    rows.append(values)
            headers = next((r for r in rows if r.get("A") == "video_id"), None)
            if headers is not None:
                start = rows.index(headers) + 1
                tables.append([{headers[k]: row.get(k, "") for k in headers}
                               for row in rows[start:] if row.get("A")])
    if len(tables) != 2:
        raise ValueError("Expected video inventory and event-claim worksheets")
    return tables


def _parents(events_path, split_path, video_dir):
    source = _rows(events_path)
    events = {r["clip_id"]: r for r in source}
    if len(events) != len(source) or not events:
        raise ValueError("Invalid events IDs")
    paths = {}
    for path in Path(video_dir).rglob("*"):
        match = re.fullmatch(r"[XХ]-(\d+)", path.stem, re.I)
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and match:
            source_id = f"X-{int(match[1])}"
            if source_id in paths:
                raise ValueError("Ambiguous original video")
            paths[source_id] = path.resolve()
    parents, aliases, used_hashes = {}, {}, set()
    for row in _rows(split_path):
        canonical = row["clip_id"]
        members = json.loads(row["source_clip_ids"])
        if (canonical in parents or canonical not in members or len(set(members)) != len(members)
                or row["split"] not in {"train", "validation"} or row["sha256"] in used_hashes):
            raise ValueError("Invalid canonical split")
        path = paths.get(canonical)
        if path is None or sha256(path) != row["sha256"]:
            raise ValueError("Original video hash differs from fixed split")
        for alias in members:
            if alias in aliases or alias not in events:
                raise ValueError("Invalid source alias coverage")
            aliases[alias] = canonical
            if events[alias]["source_label"] != row["source_label"]:
                raise ValueError("Split binary label differs from current events")
            if alias in paths and sha256(paths[alias]) != row["sha256"]:
                raise ValueError("Duplicate alias differs from canonical video")
        stamps = sorted({_seconds(events[a]["timecode"]) for a in members
                         if _seconds(events[a]["timecode"]) is not None})
        if stamps != sorted(json.loads(row["event_times_sec"])):
            raise ValueError("Split timestamps differ from current events")
        metadata = inspect_metadata(path)
        parents[canonical] = dict(clip_id=canonical, source_ids=members, split=row["split"],
                                  path=str(path), sha256=row["sha256"], metadata=metadata)
        used_hashes.add(row["sha256"])
    if set(aliases) != set(events):
        raise ValueError("Split does not cover current events exactly")
    return events, parents, aliases


def audit_supplemental_data(*, events_path, split_path, video_dir, augmented_dir,
                            labels_path, synthetic_dir, workbook_path):
    """Produce a fail-closed inventory, preserving every input and original split."""
    input_paths = dict(events=events_path, split=split_path, labels=labels_path, workbook=workbook_path)
    inputs = {k: dict(path=str(Path(p).resolve()), sha256=sha256(p)) for k, p in input_paths.items()}
    events, parents, aliases = _parents(events_path, split_path, video_dir)
    augmented, seen_names, seen_digests = [], {}, {}
    for number, row in enumerate(_rows(labels_path), 2):
        item = dict(row_number=number, source_claim=row, status="rejected", training_eligible=False)
        try:
            name, augmentation = row["наименование файла"], row["augmentation"]
            path = _safe_file(augmented_dir, name)
            item.update(path=str(path), sha256=sha256(path), metadata=inspect_metadata(path))
            if name in seen_names:
                seen_names[name].update(status="rejected", training_eligible=False,
                                        reason="Duplicate augmentation filename in labels")
                raise ValueError("Duplicate augmentation filename in labels")
            seen_names[name] = item
            source_id = _normalize_id(row["source_group"])
            if source_id not in aliases or augmentation not in AUGMENTATIONS:
                raise ValueError("Unknown parent or augmentation")
            source_file = row["source_file"]
            name_parts = Path(name).stem.split("__")
            if (not re.fullmatch(r"[XХ]-\d+\.(mov|mp4|avi|mkv|m4v)", source_file, re.I)
                    or _normalize_id(Path(source_file).stem) != source_id or len(name_parts) != 2
                    or _normalize_id(name_parts[0]) != source_id or name_parts[1] != augmentation):
                raise ValueError("Augmentation filename/source identity mismatch")
            canonical, expected = aliases[source_id], events[source_id]
            parent = parents[canonical]
            item.update(parent_id=canonical, source_id=source_id, parent_sha256=parent["sha256"],
                        inherited_split=parent["split"], augmentation=augmentation,
                        inverse_horizontal_flip=augmentation == "mirror")
            if (_label(row["нарушение / без нарушения"]) != expected["source_label"]
                    or _seconds(row["таймстемп"]) != _seconds(expected["timecode"])):
                raise ValueError("External binary label/timestamp differs from current events")
            if not _same_metadata(item["metadata"], parent["metadata"]):
                raise ValueError("Augmentation timing/geometry differs from parent; no safe time mapping")
            previous = seen_digests.get(item["sha256"])
            if previous:
                reason = ("Exact duplicate crosses canonical source groups" if previous["parent_id"] != canonical
                          else "Exact duplicate augmentation content")
                previous.update(status="rejected", training_eligible=False, reason=reason)
                raise ValueError(reason)
            if item["sha256"] in {p["sha256"] for p in parents.values()}:
                raise ValueError("Augmentation duplicates an original video")
            seen_digests[item["sha256"]] = item
            item.update(status="eligible_train_parent" if parent["split"] == "train" else "quarantined_validation_descendant",
                        training_eligible=parent["split"] == "train", inherited_time_sec=_seconds(expected["timecode"]),
                        supervision="Only reviewed current parent targets; external binary labels are consistency checks")
        except (KeyError, ValueError, OSError) as error:
            item["reason"] = str(error)
        augmented.append(item)
    listed = {Path(i["path"]).name for i in augmented if i.get("path")}
    for path in sorted(Path(augmented_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and path.name not in listed:
            augmented.append(dict(path=str(path.resolve()), sha256=sha256(path),
                                  status="quarantined_unlabelled", training_eligible=False))

    inventory, claims = read_workbook(workbook_path)
    synthetic, synthetic_names, synthetic_digests = [], set(), set()
    claims_by_id = {}
    for claim in claims:
        claims_by_id.setdefault(claim["video_id"], []).append(claim)
    for row in inventory:
        item = dict(source_claim=row, event_claims=claims_by_id.pop(row["video_id"], []),
                    origin="synthetic_generated", role="auxiliary_train_only", independent_evidence=False,
                    training_eligible=False, label_import_authorized=False, status="rejected")
        try:
            # Workbook paths document former layout. Resolve only the validated basename
            # inside the explicitly supplied directory; never follow workbook paths.
            relative = PurePosixPath(row["Файл"].replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
                raise ValueError("Unsafe workbook video path")
            path = _safe_file(synthetic_dir, relative.name)
            item.update(path=str(path), sha256=sha256(path), metadata=inspect_metadata(path),
                        source_id=row["source_id"], augmentation=row["augmentation"])
            if (path.name in synthetic_names or item["sha256"] in synthetic_digests
                    or path.stem != row["video_id"] or not re.fullmatch(r"S\d+", row["source_id"])
                    or row["augmentation"] not in {"none", "hflip"}):
                raise ValueError("Duplicate or invalid synthetic source identity")
            expected_id = f"{row['source_id']}_{'original' if row['augmentation'] == 'none' else 'hflip'}"
            if row["video_id"] != expected_id:
                raise ValueError("Synthetic video/source identity mismatch")
            synthetic_names.add(path.name); synthetic_digests.add(item["sha256"])
            expected = {"fps": float(row["FPS"]), "frame_count": int(row["Кадров"]),
                        "width": int(row["Ширина, px"]), "height": int(row["Высота, px"])}
            if (item["sha256"] != row["SHA-256"] or not _same_metadata(item["metadata"], expected)
                    or not math.isclose(item["metadata"]["duration_sec"], float(row["Длительность, с"]), abs_tol=1e-4)):
                raise ValueError("Synthetic hash/metadata differs from workbook")
            if int(row["Событий"]) != len(item["event_claims"]):
                raise ValueError("Workbook event count mismatch")
            for claim in item["event_claims"]:
                start, end = float(claim["Начало, с"]), float(claim["Конец, с"])
                if (claim["source_id"] != row["source_id"] or claim["augmentation"] != row["augmentation"]
                        or float(claim["FPS"]) != expected["fps"]
                        or not 0 <= start < end <= item["metadata"]["duration_sec"] + 1e-6
                        or not math.isclose(int(claim["start_frame"]) / expected["fps"], start, abs_tol=1e-6)
                        or not math.isclose(int(claim["end_frame_exclusive"]) / expected["fps"], end, abs_tol=1e-6)):
                    raise ValueError("Invalid workbook event timing/source claim")
            item.update(status="auxiliary_train_only_requires_visual_review",
                        reason="Generated scenario claims are not independently verified real labels or legal findings")
        except (KeyError, ValueError, OSError) as error:
            item["reason"] = str(error)
        synthetic.append(item)
    if claims_by_id:
        raise ValueError("Workbook event claims reference unknown videos")
    original_groups = {r.get("source_id") for r in synthetic
                       if r.get("augmentation") == "none" and r["status"] != "rejected"}
    for item in synthetic:
        if item.get("augmentation") == "hflip" and item.get("source_id") not in original_groups:
            item.update(status="rejected", reason="Synthetic mirror has no verified original source")
    for path in sorted(Path(synthetic_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and path.name not in synthetic_names:
            if any(i.get("path") == str(path.resolve()) for i in synthetic):
                continue
            synthetic.append(dict(path=str(path.resolve()), sha256=sha256(path), origin="unverified",
                                  status="quarantined_unlabelled", training_eligible=False,
                                  independent_evidence=False))
    if any(sha256(v["path"]) != v["sha256"] for v in inputs.values()):
        raise ValueError("Source files changed during audit")
    return dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs,
                parents=parents, augmentations=augmented, synthetic=synthetic,
                counts=dict(augmentations=dict(Counter(i["status"] for i in augmented)),
                            synthetic=dict(Counter(i["status"] for i in synthetic))),
                limitations=["Same canonical source and all descendants stay in the original partition.",
                             "Metadata and hashes do not independently prove augmentation content identity.",
                             "Synthetic scenarios and mirrored copies are not independent real evidence.",
                             "No external legal claim is automatically a reviewed action target."])


@dataclass(frozen=True)
class AugmentedSource:
    path: Path
    sha256: str
    parent_id: str
    augmentation: str
    inverse_horizontal_flip: bool

    def restore_orientation(self, frames):
        """Flip CHW/TCHW tensors or arrays before fixed ROI; HWC is unsupported."""
        if getattr(frames, "ndim", None) is not None and (frames.ndim not in (3, 4) or frames.shape[-3] != 3):
            raise ValueError("Orientation restoration requires RGB CHW or TCHW frames")
        if not self.inverse_horizontal_flip:
            return frames
        return frames.flip(-1) if hasattr(frames, "flip") else frames[..., ::-1].copy()


def resolve_augmentations(manifest, reviewed_sample, *, events_path, split_path, inverse_mirror=False):
    """Resolve verified train-parent variants; no labels come from this manifest.

    Pass an existing ActionSample/PrimitiveSample with explicit reviewed targets.
    The consumer decodes these paths at the same sample times and calls
    restore_orientation before applying the parent's spatial preprocessing.
    """
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported supplemental manifest")
    for key, path in (("events", events_path), ("split", split_path)):
        if sha256(path) != manifest["inputs"][key]["sha256"]:
            raise ValueError("Supplemental manifest no longer matches current data/split")
    record = reviewed_sample.record
    if record.split != "train":
        raise ValueError("Validation descendants must never enter training")
    if not (reviewed_sample.positives or getattr(reviewed_sample, "negatives", ())):
        raise ValueError("Explicitly reviewed parent targets are required")
    parent = manifest["parents"].get(record.clip_id)
    if (not parent or parent["sha256"] != record.sha256 or parent["split"] != "train"
            or not 0 <= reviewed_sample.center_sec <= record.last_time):
        raise ValueError("Reviewed sample differs from audited canonical parent")
    selected, seen = [], set()
    # Prefer canonical file over a duplicate alias; X-4/X-47 share video imagery
    # while their distinct reviewed event times remain in the caller's samples.
    candidates = sorted(manifest["augmentations"], key=lambda i: (i.get("source_id") != record.clip_id, i.get("path", "")))
    for item in candidates:
        if item.get("parent_id") != record.clip_id or item.get("status") != "eligible_train_parent":
            continue
        if (not item.get("training_eligible") or item.get("inherited_split") != "train"
                or item.get("parent_sha256") != record.sha256):
            raise ValueError("Invalid augmentation partition/provenance")
        flip = item["augmentation"] == "mirror"
        if flip and not inverse_mirror:
            continue
        if item["augmentation"] in seen:
            continue
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError("Augmentation content changed after audit")
        selected.append(AugmentedSource(path, item["sha256"], record.clip_id, item["augmentation"], flip))
        seen.add(item["augmentation"])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--events", type=Path, default=root / "data/annotations/events.csv")
    parser.add_argument("--split", type=Path, default=root / "data/processed/split_proposal.csv")
    parser.add_argument("--video-dir", type=Path, default=root / "data/raw/videos")
    parser.add_argument("--augmented-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--synthetic-dir", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new audit output path")
    result = audit_supplemental_data(events_path=args.events, split_path=args.split, video_dir=args.video_dir,
                                   augmented_dir=args.augmented_dir, labels_path=args.labels,
                                   synthetic_dir=args.synthetic_dir, workbook_path=args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
