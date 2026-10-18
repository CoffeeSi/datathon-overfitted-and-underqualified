"""Import a dataset owner's explicit batch attestation, preserving unknown cases.

This records a human statement, not an invented frame-by-frame annotation session.
Absent labels cover ONLY the sampled 4-second window; no video-wide absence follows.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import yaml
from overfitted_underqualified.ml.action_data import mapped_types, prepare_action_data
from overfitted_underqualified.ml.action_review import apply_decisions, REVIEW_SCHEMA
from overfitted_underqualified.ml.action_train import read_config, write_json
from overfitted_underqualified.ml.dataset import annotation_fields_digest, load_records, parse_time, sha256
from overfitted_underqualified.rules.catalog import CATALOG_PATH, supported_event_types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-complete-window-labels", action="store_true", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--statement", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "configs/actions.yaml"
    config = read_config(config_path)
    events, split = Path(config["events_path"]), Path(config["split_path"])
    with events.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    records = load_records(events, split, Path(config["video_dir"]))
    by_alias = {a: r for r in records for a in r.aliases}
    catalog_hash, source_hash, split_hash = sha256(CATALOG_PATH), sha256(events), sha256(split)
    classes = supported_event_types()
    stamp = datetime.now(timezone.utc)
    decisions, skipped = [], []
    for row in rows:
        record = by_alias[row["clip_id"]]
        positives = mapped_types(row["observed_event_type"])
        background = row["source_label"] == "no_violation" and row["observed_event_type"] == "not_reported"
        # Explicit unknown conditions/excluded legal claims cannot be resolved by
        # a blanket attestation. Leave ALL their action targets unknown.
        if not positives and not background:
            skipped.append(row["clip_id"]); continue
        center = parse_time(row["timecode"])
        center = record.last_time / 2 if center is None else center
        value = dict(schema=REVIEW_SCHEMA, catalog_sha256=catalog_hash, video_sha256=record.sha256,
                     center_sec=center, window_sec=config["window_sec"], reviewer=args.reviewer,
                     reviewed_at=stamp.isoformat(), present_types=positives,
                     absent_types=sorted(classes-set(positives)), whole_video_absent_types=[],
                     evidence_note="Owner batch attestation in chat: " + args.statement
                     + ". Listed mapped actions and completeness confirmed for this window; "
                       "not a new frame-by-frame review. No whole-video absence attested.")
        decisions.append(dict(clip_id=row["clip_id"], review=value))
    envelope = dict(schema="action_review_decisions_v1", events_sha256=source_hash,
                    split_sha256=split_hash, catalog_sha256=catalog_hash, rows=decisions)
    result = apply_decisions(events, split, config["video_dir"], envelope, config["window_sec"], apply=args.apply)
    if not args.apply:
        print(json.dumps(dict(**result, skipped_unknown_ids=skipped), ensure_ascii=False, indent=2)); return
    after = sha256(events)
    with events.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f); new_fields = reader.fieldnames; new_rows = list(reader)
    assert annotation_fields_digest(rows, fields) == annotation_fields_digest(new_rows, fields)
    registry_path = ROOT / "data/annotation_revisions.json"
    entries = json.loads(registry_path.read_text(encoding="utf-8"))
    old_digests = {source_hash} | {e["before_sha256"] for e in entries} | {e["after_sha256"] for e in entries}
    for digest in sorted(old_digests):
        snapshot = ROOT / "data/local_artifacts/annotation_snapshots" / digest / "events.csv"
        if not snapshot.exists():
            raise FileNotFoundError("Cannot register without exact original: " + str(snapshot))
        with snapshot.open(encoding="utf-8-sig", newline="") as f: old = list(csv.DictReader(f))
        allowed = [k for k in ("observed_event", "legal_basis") if any(a[k] != b[k] for a,b in zip(old,new_rows))] + ["action_review"]
        protected = [k for k in new_fields if k not in allowed]
        assert annotation_fields_digest(old,protected) == annotation_fields_digest(new_rows,protected)
        entries.append(dict(schema="binary_action_review_revision_v1", before_sha256=digest, after_sha256=after,
                            allowed_changed_fields=allowed, preserved_fields=protected,
                            preserved_fields_sha256=annotation_fields_digest(new_rows,protected),
                            created_utc=stamp.isoformat(), reason="Owner-confirmed action-window labels added; binary supervision unchanged."))
    write_json(registry_path, entries)
    run_name = "r3d18_actions_owner_" + stamp.strftime("%Y%m%dT%H%M%S_%fZ")
    config["review"].update(events_sha256=after, catalog_mapping_verified=True,
                             provenance="Owner batch attestation: " + args.statement + "; unresolved tags masked; no whole-video absence.")
    config["output_dir"] = str(ROOT / "runs" / run_name)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # Keep portable repository paths in the checked-in configuration.
    for key in ("events_path", "split_path", "video_dir", "output_dir", "weights_cache"):
        config[key] = Path(config[key]).relative_to(ROOT).as_posix()
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _, _, audit = prepare_action_data(read_config(config_path))
    output = ROOT / "reports" / (run_name + "_attestation.json")
    write_json(output, dict(statement=args.statement, reviewer=args.reviewer, scope="4-second windows only",
                            original_events_sha256=source_hash, events_sha256=after, split_sha256=split_hash,
                            skipped_unknown_ids=skipped, reviewed_source_ids=result["changed_ids"],
                            audit=audit))
    print(json.dumps(dict(run=run_name, device=config["device"], reviewed_windows=audit["reviewed_windows"],
                         classes=audit["class_names"], training_ready=audit["training_ready"],
                         skipped_unknown_ids=skipped, attestation=str(output)), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
