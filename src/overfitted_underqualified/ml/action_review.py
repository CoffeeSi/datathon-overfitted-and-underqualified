"""Per-type, three-state review stored in events.csv, never inferred from binary labels."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path

from .dataset import parse_time, sha256
from ..rules.catalog import CATALOG_PATH, supported_event_types

REVIEW_SCHEMA = "action_window_review_v1"
REVIEW_COLUMN = "action_review"

# Catalog mappings are research hypotheses. These tags omit a defining condition.
# Keep the catalog/API and historical coverage intact; never use these automatically
# as supervised action targets. An explicit reviewed target can resolve the condition.
SEMANTIC_GAPS = {
    "unattended_bag_contents_unknown": "Содержимое сумки и факт предыдущего приёма наличных не установлены.",
    "employee_handles_suspected_cash_without_client": "Работа с предполагаемыми наличными не устанавливает нарушение хранения принятых денег.",
    "suspected_accepted_cash_left_after_client_departure": "Не установлены предмет (наличные) и факт его предыдущего приёма.",
    "employee_steps_away_suspected_cash_on_desk": "Уход от стола не подтверждает, что обслуживание клиента ещё продолжалось.",
}


def validate_review(value, *, center, window_sec, video_sha256):
    """Validate a review independently of its storage (CSV, import or checkpoint).

    Missing types remain unknown. Whole-video absence requires an additional,
    explicit per-type declaration; a negative window never implies video absence.
    """
    if not isinstance(value, dict) or value.get("schema") != REVIEW_SCHEMA:
        raise ValueError("Invalid action review schema")
    if value.get("catalog_sha256") != sha256(CATALOG_PATH) or value.get("video_sha256") != video_sha256:
        raise ValueError("Action review refers to stale catalog/video")
    for key, expected in (("center_sec", center), ("window_sec", window_sec)):
        actual = value.get(key)
        if type(actual) not in (int, float) or not math.isfinite(actual) or abs(actual - expected) > 1e-6:
            raise ValueError(f"Action review refers to another input window: {key}")
    for key in ("reviewer", "evidence_note"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"Action review needs {key}")
    try:
        stamp = datetime.fromisoformat(value["reviewed_at"].replace("Z", "+00:00"))
        if stamp.utcoffset() is None:
            raise ValueError("timezone missing")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError("Action review needs a timestamp with timezone") from exc
    for key in ("present_types", "absent_types", "whole_video_absent_types"):
        names = value.get(key)
        if (not isinstance(names, list) or any(not isinstance(n, str) for n in names)
                or len(set(names)) != len(names) or not set(names) <= supported_event_types()):
            raise ValueError(f"Invalid action review types: {key}")
    present, absent, whole = (set(value[k]) for k in ("present_types", "absent_types", "whole_video_absent_types"))
    if present & absent or not whole <= absent or not present | absent:
        raise ValueError("Conflicting or empty action review")
    return value


def read_review(row, record, window_sec):
    raw = row.get(REVIEW_COLUMN, "").strip()
    if not raw:
        return None
    center = parse_time(row["timecode"])
    if center is None:
        center = record.last_time / 2
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid action_review JSON: {row['clip_id']}") from exc
    return validate_review(value, center=center, window_sec=window_sec, video_sha256=record.sha256)


def apply_decisions(events_path, split_path, video_dir, decisions, window_sec, *, apply=False, replace_existing=False):
    """Dry-run by default. Import explicit decisions into the sole annotation CSV.

    The downloaded JSON is a transfer envelope, never read by training. Preserve an
    immutable pre-edit CSV for historical binary checkpoint provenance before apply.
    """
    from .dataset import load_records
    events_path, split_path = Path(events_path), Path(split_path)
    before, split_digest = sha256(events_path), sha256(split_path)
    if (decisions.get("schema") != "action_review_decisions_v1" or decisions.get("events_sha256") != before
            or decisions.get("split_sha256") != split_digest or decisions.get("catalog_sha256") != sha256(CATALOG_PATH)):
        raise ValueError("Decisions refer to stale annotations, split or catalog")
    records = load_records(events_path, split_path, Path(video_dir))
    by_alias = {a: r for r in records for a in r.aliases}
    with events_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames), list(reader)
    entries = decisions.get("rows", [])
    if not entries or len({r["clip_id"] for r in entries}) != len(entries):
        raise ValueError("Import needs unique, nonempty source IDs")
    original = {r["clip_id"]: dict(r) for r in rows}
    edits = {r["clip_id"]: r for r in entries}
    if not edits.keys() <= original.keys():
        raise ValueError("Unknown source IDs")
    if REVIEW_COLUMN not in fields:
        fields.append(REVIEW_COLUMN)
    for row in rows:
        if row["clip_id"] in edits:
            if row.get(REVIEW_COLUMN, "").strip() and not replace_existing:
                raise ValueError(f"Existing action review would be replaced: {row['clip_id']}; use --replace-existing after reviewing all its decisions")
            value = edits[row["clip_id"]]["review"]
            row[REVIEW_COLUMN] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        # Validate both new and previously stored decisions, not just imported rows.
        read_review(row, by_alias[row["clip_id"]], window_sec)
        if any(row[k] != original[row["clip_id"]][k] for k in original[row["clip_id"]] if k != REVIEW_COLUMN):
            raise ValueError("Import changed a source annotation")
    # Reuse the actual preparation path to detect duplicate/overlapping contradictions.
    # This pure check does not touch data or require the legacy review flags.
    from .action_data import reviewed_samples
    reviewed_samples(records, {r["clip_id"]: r for r in rows}, window_sec)
    result = dict(changed_ids=sorted(edits), previous_events_sha256=before, applied=apply)
    if apply:
        snapshot = events_path.parents[2] / "data/local_artifacts" / "annotation_snapshots" / before / "events.csv"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        original_bytes = events_path.read_bytes()
        if snapshot.exists() and snapshot.read_bytes() != original_bytes:
            raise ValueError("Historical annotation snapshot conflicts")
        if not snapshot.exists():
            snapshot.write_bytes(original_bytes)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = events_path.with_name(events_path.name + ".review.tmp")
        if sha256(events_path) != before or sha256(split_path) != split_digest:
            raise ValueError("Annotations changed during import")
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(buffer.getvalue())
        if sha256(events_path) != before or sha256(split_path) != split_digest:
            temporary.unlink()
            raise ValueError("Annotations changed before replacement")
        temporary.replace(events_path)
        result.update(events_sha256=sha256(events_path), historical_events_path=str(snapshot))
    return result


def main():
    from .action_train import read_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--replace-existing", action="store_true", help="Explicitly replace an existing row review, including its previous known types")
    args = parser.parse_args()
    config = read_config(args.config)
    result = apply_decisions(config["events_path"], config["split_path"], config["video_dir"],
                             json.loads(args.decisions.read_text(encoding="utf-8")), config["window_sec"],
                             apply=args.apply, replace_existing=args.replace_existing)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.apply:
        print("Update review.events_sha256 in the action config to the printed digest. "
              "Historical binary evaluation must use historical_events_path; its digest checks remain strict.")


if __name__ == "__main__":
    main()
