"""Create a numerical, media-free study summary; retain unsuccessful variants."""
import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from overfitted_underqualified.ml.train import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.study/"plan.json").read_text(encoding="utf-8"))
    comparison = json.loads((args.study/"comparison.json").read_text(encoding="utf-8"))
    if len(comparison) != len(plan["experiments"]) + 1: raise ValueError("Study incomplete")
    verification = json.loads((args.study/"verification.json").read_text(encoding="utf-8"))
    reproducibility = json.loads((args.study/"reproducibility.json").read_text(encoding="utf-8"))
    policies = json.loads((args.study/"localization_comparison.json").read_text(encoding="utf-8"))
    # This is an exploratory recommendation on an already examined validation set.
    # It is not a replacement of the public service or a new independent test.
    eligible = [r for r in comparison if r["name"] != "baseline"]
    selected = max(eligible, key=lambda r: (r["full_metrics"]["video"]["f1"],
                   -r["full_metrics"]["false_alarms"]["candidates"], r["full_metrics"]["temporal"]["known_marker_recall"]))
    output = ROOT/"reports"/("evaluation_binary_study_"+args.study.name)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for result in comparison:
        video = result["full_metrics"]["video"]
        temporal = result["full_metrics"]["temporal"]
        rows.append(dict(experiment=result["name"], epoch=result["checkpoint_epoch"], window_f1=result["window_metrics"]["f1"],
                         full_precision=video["precision"], full_recall=video["recall"], full_f1=video["f1"],
                         full_accuracy=video["accuracy"], negative_candidates=result["full_metrics"]["false_alarms"]["candidates"],
                         marker_recall=temporal["known_marker_recall"], matched_mae_sec=temporal["matched_mae_sec"],
                         processing_sec=result["timing"]["processing_sec"], checkpoint_sha256=result["checkpoint_sha256"]))
    with (output/"comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = dict(plan=plan, experiments=comparison, selected=selected, verification=verification, reproducibility=reproducibility,
                   localization=policies, independent_evaluation=False,
                   limitations=["10 shared-room validation videos, already examined; no independent test.",
                                "Best epochs use window-validation loss; candidate recommendation uses exploratory full-video F1.",
                                "One video changes accuracy by 10 percentage points. Improvements need new data and sessions.",
                                "Binary results do not classify action types or establish legal applicability.",
                                "Unmatched predictions on positive videos remain unresolved, not confirmed false positives."])
    (output/"study.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    labels = [r["experiment"] for r in rows]
    for ax, keys, title, ylabel in ((axes[0],("full_f1","full_recall"),"Full-video classification, threshold 0.5","Fraction"),
                                  (axes[1],("marker_recall",),"Known markers within +/-5 seconds","Fraction"),
                                  (axes[2],("negative_candidates",),"Alarms on 5 accepted negative videos","Candidate count")):
        for j, key in enumerate(keys):
            positions = [i + j * .32 for i in range(len(rows))]
            ax.bar(positions,[r[key] for r in rows],width=.3,label=key)
        ax.set(xticks=list(range(len(rows))), xticklabels=labels,title=title,ylabel=ylabel,xlabel="Experiment")
        ax.tick_params(axis="x", rotation=50)
        if len(keys)>1: ax.legend()
    fig.suptitle("Exploratory comparison: same 10 within-room validation videos; no independent test")
    fig.tight_layout()
    fig.savefig(output/"comparison.png", dpi=130)
    plt.close(fig)
    lines = ["# Binary R3D-18 improvement study — 18 October 2026", "",
             "Exploratory comparison on the same 10 validation videos; known errors informed the hypotheses.", "",
             f"Selected research candidate: `{selected['name']}`; checkpoint `{selected['checkpoint']}`.",
             f"Checkpoint SHA-256: `{selected['checkpoint_sha256']}`.", "",
             "Selection: full-video F1, then fewer negative-video alarm candidates, then known-marker recall.",
             "This recommendation does not automatically deploy a production model.", "", "## Results", ""]
    for r in rows:
        lines.append(f"- {r['experiment']}: full-video precision={r['full_precision']:.3f}, recall={r['full_recall']:.3f}, "
                     f"F1={r['full_f1']:.3f}; negative alarm candidates={r['negative_candidates']}; "
                     f"known-marker recall={r['marker_recall']:.3f}; window F1={r['window_f1']:.3f}.")
    lines += ["", "## Fixed settings and training", "",
              "20 epochs, batch 16, AdamW lr=0.0003, seed 42; frozen backbone with per-run feature cache. "
              "The stable control retains one midpoint negative and legacy geometry. Other variants use five "
              "coverage negatives per train video and video/class-balanced loss. Geometry preserves aspect ratio; "
              "ROI=[0,0.1,1,0.7] was inspected on train. The final variant changes ROI context from 4 to 8 seconds.",
              "", "Earlier batch-2 weighted-negative attempt failed and is preserved in "
              "`runs/binary_improvements_20261017T225920_013664Z`. Its model predicted all negative on window validation; "
              "NumPy scalar timestamps also prevented safe checkpoint loading. Timestamp serialization was fixed "
              "to Python floats; safe weights_only loading remains enabled. These results were not discarded to hide failure.",
              "", "## Scope and checks", "",
              "All real checkpoints restored; backbone/BatchNorm are bitwise equal to the original frozen baseline. "
              "events.csv, the 30/10 split, SHA-256 grouping and X-4/X-47 separate marks are preserved. "
              "No unmarked positive segment was relabeled negative. No threshold search was performed.",
              "", f"Same-seed repeat: state_dict bitwise equal={reproducibility['state_dict_bitwise_equal']}; "
              f"same best epoch={reproducibility['same_epoch']}. This verifies reproducibility, not generalization.",
              "", "Window-peaks localization was compared on saved scores with fixed 4-second separation. "
              "Additional candidates can improve marker recall while increasing false alarms; legacy support_union remains default.",
              "", "No bank frames or weights are included in this report archive. Local crop previews remain under ignored data/local_artifacts/.",
              "", "## Limitations", "", *["- "+s for s in summary["limitations"]]]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    with ZipFile(output.with_suffix(".zip"), "x", ZIP_DEFLATED) as archive:
        for name in ("comparison.csv","comparison.png","study.json","report.md"):
            archive.write(output/name,arcname=name)
    print(json.dumps(dict(selected=selected["name"], checkpoint=selected["checkpoint"], report_dir=str(output),
                          archive=str(output.with_suffix(".zip"))),indent=2))


if __name__ == "__main__": main()
