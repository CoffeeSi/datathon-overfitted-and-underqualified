"""Verify real checkpoints, immutable backbone and split/source provenance after a study."""
import argparse
import json
from pathlib import Path

import torch

from overfitted_underqualified.ml.dataset import sha256
from overfitted_underqualified.ml.preprocessing import video_transform
from overfitted_underqualified.ml.train import ROOT, load_checkpoint


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.study/"plan.json").read_text(encoding="utf-8"))
    rows = json.loads((args.study/"comparison.json").read_text(encoding="utf-8"))
    if len(rows) != len(plan["experiments"]) + 1: raise ValueError("Study incomplete")
    for name, digest in plan["implementation_sha256"].items():
        if sha256(ROOT/"src/overfitted_underqualified/ml"/name) != digest:
            raise ValueError(f"Study implementation changed: {name}")
    baseline = torch.load(plan["baseline_checkpoint"], map_location="cpu", weights_only=True)
    checks = []
    for row in rows:
        if sha256(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError("Changed checkpoint bytes")
        model, checkpoint = load_checkpoint(Path(row["checkpoint"]), "cpu")
        state = checkpoint["model_state_dict"]
        if any(not torch.equal(tensor, baseline["model_state_dict"][key]) for key, tensor in state.items() if not key.startswith("fc.")):
            raise ValueError("Backbone/BatchNorm changed")
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError("Nonfinite checkpoint")
        manifest = checkpoint["data_manifest"]
        for field, source in (("events_sha256", "data/annotations/events.csv"), ("split_sha256", "data/processed/split_proposal.csv")):
            if manifest[field] != sha256(ROOT/source): raise ValueError("Input drift")
        videos = {v["clip_id"]: v for v in manifest["videos"]}
        train_hashes = {v["sha256"] for v in videos.values() if v["split"] == "train"}
        val_hashes = {v["sha256"] for v in videos.values() if v["split"] == "validation"}
        if train_hashes & val_hashes: raise ValueError("Duplicate leakage")
        for subset, samples in manifest["windows"].items():
            for window in samples:
                video = videos[window["clip_id"]]
                if video["split"] != subset or window["label"] != video["label"]:
                    raise ValueError("Window label/split drift")
                if video["label"] and window["center_sec"] not in video["event_times"]:
                    raise ValueError("Unreviewed positive centre")
        preserved = sorted(w["center_sec"] for w in manifest["windows"]["train"] if w["clip_id"] == "X-4")
        if preserved != [57.,68.]: raise ValueError("X-4/X-47 source event lost")
        transform = video_transform(checkpoint["config"])
        sample = transform(torch.zeros(16, 3, 848, 464, dtype=torch.uint8))
        if tuple(sample.shape) != (3,16,112,112): raise ValueError("Invalid saved preprocessing")
        with torch.inference_mode():
            logits = model(sample.unsqueeze(0))
        if logits.shape != (1,2) or not torch.isfinite(logits).all(): raise ValueError("Restored inference failed")
        checks.append(dict(name=row["name"], checkpoint_sha256=row["checkpoint_sha256"], restored=True,
                           backbone_and_bn_bitwise_equal_to_baseline=True, split_isolated=True,
                           alias_event_times_preserved=preserved, preprocessing=str(transform)))
        del model, checkpoint, state
    output = args.study/"verification.json"
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(dict(checks=checks, assertions="Real saved models; no download, no training, zero-frame forward shape sanity only."),
                                 indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Verified", len(checks), "real checkpoints; backbone, BatchNorm and source/split preserved")


if __name__ == "__main__": main()
