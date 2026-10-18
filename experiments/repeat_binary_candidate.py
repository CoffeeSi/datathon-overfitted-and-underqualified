"""Repeat the recommended configuration with the same seed; compare actual weights."""
import argparse
import json
from pathlib import Path

import torch

from overfitted_underqualified.ml.train import run
from overfitted_underqualified.ml.dataset import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.study/"comparison.json").read_text(encoding="utf-8"))
    if len(rows) != 6: raise ValueError("Complete the predeclared study first")
    selected = max((r for r in rows if r["name"] != "baseline"), key=lambda r: (
        r["full_metrics"]["video"]["f1"], -r["full_metrics"]["false_alarms"]["candidates"],
        r["full_metrics"]["temporal"]["known_marker_recall"]))
    original = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)
    config = dict(original["config"], output_dir=str(args.study/("repeat_"+selected["name"])))
    repeated = run(config)/"best.pt"
    checkpoint = torch.load(repeated, map_location="cpu", weights_only=True)
    same = all(torch.equal(v, checkpoint["model_state_dict"][k]) for k,v in original["model_state_dict"].items())
    max_error = max(float((v-checkpoint["model_state_dict"][k]).abs().max())
                    for k,v in original["model_state_dict"].items())
    result = dict(candidate=selected["name"], original_checkpoint=selected["checkpoint"],
                  original_sha256=sha256(Path(selected["checkpoint"])), repeated_checkpoint=str(repeated),
                  repeated_sha256=sha256(repeated), state_dict_bitwise_equal=same, max_abs_weight_difference=max_error,
                  original_epoch=original["epoch"], repeated_epoch=checkpoint["epoch"],
                  seed=config["seed"], same_epoch=original["epoch"] == checkpoint["epoch"],
                  note="Same-seed reproducibility, not an independent quality test. Artifact hashes differ because saved output paths differ.")
    output = args.study/"reproducibility.json"
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print("REPRODUCIBILITY", json.dumps(result), flush=True)
    if not same or not result["same_epoch"]: raise RuntimeError("Repeat differs; inspect before recommending")


if __name__ == "__main__": main()
