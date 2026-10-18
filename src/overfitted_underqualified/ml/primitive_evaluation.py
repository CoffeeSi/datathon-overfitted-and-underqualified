"""Fixed-checkpoint primitive metrics and temporal candidates on unchanged validation videos."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import torch
from torch.utils.data import DataLoader

from .action_train import action_epoch, write_json
from .dataset import sha256
from .full_video_evaluation import match_events
from .primitive_data import CLASSES, PrimitiveDataset, prepare_primitive_data
from .primitive_inference import PrimitiveScanner
from .primitive_model import read_config
from .train import ROOT


def evaluate_primitives(config, checkpoint_path, output_dir, *, device="cpu"):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    records, samples, audit = prepare_primitive_data(config)
    scanner = PrimitiveScanner(checkpoint_path, device)
    # Runtime paths/device may be remapped; no silent change of inference or learning protocol.
    runtime = {"device", "events_path", "split_path", "video_dir", "output_dir", "weights_cache"}
    saved = scanner.config
    if ({k: v for k, v in config.items() if k not in runtime}
            != {k: v for k, v in saved.items() if k not in runtime}
            or audit["manifest"] != scanner.checkpoint["data_manifest"]):
        raise ValueError("Primitive evaluation differs from checkpoint protocol/provenance")
    validation = [s for s in samples if s.record.split == "validation"]
    loader = DataLoader(PrimitiveDataset(validation, scanner.transform, config["window_sec"]),
                        batch_size=2, num_workers=0)
    metrics, predictions = action_epoch(scanner.model, loader, scanner.device, CLASSES, config["threshold"])
    errors = [dict(clip_id=r["clip_id"], time_sec=r["center_sec"], primitive=name,
                   target=int(r["targets"][i]), score=r["probabilities"][i],
                   error="FN" if r["targets"][i] else "FP")
              for r in predictions for i, name in enumerate(CLASSES)
              if r["mask"][i] and int(r["probabilities"][i] >= config["threshold"]) != r["targets"][i]]
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "data_manifest.json", audit["manifest"])
    write_json(output / "primitive_metrics.json", metrics)
    write_json(output / "primitive_predictions.json", predictions)
    write_json(output / "primitive_errors.json", errors)
    # Known positive type points only. Binary no_violation is NOT catalog-wide absence.
    with Path(config["events_path"]).open(encoding="utf-8-sig", newline="") as stream:
        source = {r["clip_id"]: r for r in csv.DictReader(stream)}
    scans, matching = [], []
    for record in records:
        if record.split != "validation":
            continue
        scan = scanner.scan(record.path)
        if scan["video_sha256"] != record.sha256:
            raise ValueError("Scanned video differs from validated split")
        scan["clip_id"] = record.clip_id
        scans.append(scan)
        write_json(output / f"{record.clip_id}_scan.json", scan)
        truth = set()
        for alias in record.aliases:
            review = json.loads(source[alias]["action_review"]) if source[alias].get("action_review") else {}
            if "unsecured_accepted_cash" in review.get("present_types", []):
                truth.add(review["center_sec"])
        matching.extend(dict(clip_id=record.clip_id, **m) for m in match_events(
            [e["timestamp_sec"] for e in scan["events"]], sorted(truth), config["match_tolerance_sec"]))
        print(record.clip_id, "candidates", len(scan["events"]), "seconds", round(scan["processing_sec"], 2), flush=True)
    for key, digest_key in (("events_path", "events_sha256"), ("split_path", "split_sha256")):
        if sha256(Path(config[key])) != config[digest_key]:
            raise ValueError("Data changed during primitive evaluation")
    matched = [r for r in matching if r["status"] == "matched"]
    missed = [r for r in matching if r["status"] == "unmatched_annotation"]
    summary = dict(schema="primitive_evaluation_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                   checkpoint=str(Path(checkpoint_path).resolve()), checkpoint_sha256=scanner.checkpoint_sha256,
                   events_sha256=config["events_sha256"], split_sha256=config["split_sha256"],
                   coverage=audit["coverage"], primitive_metrics=metrics,
                   temporal=dict(matched=len(matched), missed=len(missed),
                       recall=len(matched) / (len(matched) + len(missed)) if matched or missed else None,
                       precision=None, f1=None, reason="No complete type-specific whole-video negative labels",
                       timestamp_mae_sec=sum(r["absolute_error_sec"] for r in matched) / len(matched) if matched else None),
                   total_candidates=sum(len(s["events"]) for s in scans),
                   videos=[dict(clip_id=s["clip_id"], candidates=len(s["events"]),
                                uncertain_observations=s["uncertain_observations"], processing_sec=s["processing_sec"])
                           for s in scans],
                   full_scan_processing_sec=sum(s["processing_sec"] for s in scans),
                   deployment_enabled=False, independent_evaluation=False,
                   limitations=["Sparse frame reviews; correlated timestamps are not independent videos.",
                                "Same room/people and repeatedly inspected validation; exploratory experiment.",
                                "Three visual primitives do not establish ownership, accounting completion or legal violation.",
                                "Nominal frame/FPS timestamps; no variable-frame-rate ground truth.",
                                "False primitive states can break or invent an entire temporal chain."])
    write_json(output / "summary.json", summary)
    write_json(output / "matching.json", matching)
    # Complete numeric/provenance package only; never frames, raw video, or weights.
    with ZipFile(output.with_suffix(".zip"), "x", ZIP_DEFLATED) as archive:
        for path in sorted(output.glob("*.json")):
            archive.write(path, path.name)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/primitives.yaml")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    result = evaluate_primitives(read_config(args.config), args.checkpoint, args.output, device=args.device)
    print(json.dumps(result["temporal"], indent=2))


if __name__ == "__main__":
    main()
