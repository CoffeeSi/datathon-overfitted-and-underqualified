"""Compare fixed localization policies on saved full-video scores; no extra inference."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from overfitted_underqualified.ml.dataset import load_records, sha256
from overfitted_underqualified.ml.full_video_evaluation import summarize_scans
from overfitted_underqualified.ml.inference import ScanConfig, merge_candidates
from overfitted_underqualified.ml.train import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    output = args.study / "localization_comparison.json"
    if output.exists(): raise FileExistsError(output)
    records = [r for r in load_records(ROOT/"data/annotations/events.csv", ROOT/"data/processed/split_proposal.csv", ROOT/"data/raw/videos") if r.split == "validation"]
    comparison = json.loads((args.study / "comparison.json").read_text(encoding="utf-8"))
    results = []
    for item in comparison:
        directory = Path(item["report_dir"])
        report = json.loads((directory/"evaluation.json").read_text(encoding="utf-8"))
        if report["checkpoint_sha256"] != item["checkpoint_sha256"]:
            raise ValueError("Checkpoint provenance differs")
        for key, file in (("events_sha256", "data/annotations/events.csv"), ("split_sha256", "data/processed/split_proposal.csv")):
            if report["data_manifest"][key] != sha256(ROOT/file): raise ValueError("Label/split drift")
        with (directory/"video_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            videos = {row["clip_id"]: row for row in csv.DictReader(stream)}
        with (directory/"window_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            windows = list(csv.DictReader(stream))
        for policy in ("support_union", "window_peaks"):
            config = ScanConfig(**{**report["scan_config"], "candidate_policy": policy, "peak_separation_sec": 4.})
            scans = []
            for record in records:
                rows = [{k: int(w[k]) if k in {"window_index", "prediction"} else float(w[k])
                         for k in ("window_index", "start_sec", "end_sec", "center_sec", "probability", "prediction")}
                        for w in windows if w["clip_id"] == record.clip_id]
                scans.append(dict(clip_id=record.clip_id, sha256=record.sha256, fps=record.fps,
                                  frame_count=record.frame_count, duration_sec=record.frame_count/record.fps,
                                  processing_sec=float(videos[record.clip_id]["processing_sec"]),
                                  scan_config=asdict(config), windows=rows, candidates=merge_candidates(rows, config)))
            summary = summarize_scans(records, scans, 5.)
            results.append(dict(name=item["name"], policy=policy, config=asdict(config), metrics=summary["metrics"],
                                matching=summary["matching"], candidates=summary["candidates"],
                                checkpoint_sha256=item["checkpoint_sha256"]))
    output.write_text(json.dumps(dict(tolerance_sec=5., independent_evaluation=False, results=results,
                                     note="Postprocessing comparison on previously examined validation; peak candidates may inflate false alarms."),
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in results:
        print(row["name"], row["policy"], row["metrics"]["temporal"], flush=True)


if __name__ == "__main__": main()
