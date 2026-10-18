"""Partial label review safety, independent of real banking ground truth."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from overfitted_underqualified.ml.action_data import ActionDataset, prepare_action_data, reviewed_samples, mapped_types, validate_review_manifest
from overfitted_underqualified.ml.action_review import REVIEW_SCHEMA, apply_decisions, validate_review
from overfitted_underqualified.ml.action_evaluation import temporal_summary
from overfitted_underqualified.ml.dataset import sha256
from overfitted_underqualified.rules.catalog import CATALOG_PATH

A, B = "safe_key_left_in_lock", "unsecured_accepted_cash"


def review(record, center=5., present=(), absent=(), whole=()):
    return dict(schema=REVIEW_SCHEMA, catalog_sha256=sha256(CATALOG_PATH), video_sha256=record.sha256,
                center_sec=center, window_sec=4., reviewer="SYNTHETIC TEST", reviewed_at="2026-10-18T00:00:00Z",
                evidence_note="Synthetic fixture, not a real video review", present_types=list(present),
                absent_types=list(absent), whole_video_absent_types=list(whole))


def row(clip_id, value=None, time="00:05", tag="safe_key_left_in_lock"):
    return dict(clip_id=clip_id, timecode=time, observed_event_type=tag,
                action_review=json.dumps(value) if value else "")


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.record=SimpleNamespace(clip_id="X-4", aliases=("X-4", "X-47"), split="train", sha256="a"*64, last_time=100.)

    def test_partial_targets_unknown_and_duplicate_source_points(self):
        rows={"X-4":row("X-4",review(self.record,present=[A],absent=[B])),
              "X-47":row("X-47",review(self.record,center=68.,present=[B]),time="01:08")}
        samples=reviewed_samples([self.record],rows,4.)
        self.assertEqual([s.center_sec for s in samples],[5.,68.])
        transform=lambda _:torch.zeros(3,16,112,112)
        with patch("overfitted_underqualified.ml.action_data.decode_frames",return_value=None):
            data=ActionDataset(samples,[A,B,"client_identity_check_not_visible"],transform,4.)
            self.assertEqual(data[0]["mask"].tolist(),[1,1,0])
            self.assertEqual(data[0]["targets"].tolist(),[1,0,0])
            self.assertEqual(data[1]["mask"].tolist(),[0,1,0])

    def test_invalid_review_provenance_conflicts_and_scopes(self):
        for change in (dict(present_types=[A],absent_types=[A]),dict(absent_types=[B],whole_video_absent_types=[A]),
                       dict(window_sec=8),dict(reviewer=""),dict(reviewed_at="yesterday"),dict(video_sha256="b"*64),
                       dict(catalog_sha256="stale"),dict(present_types=["violation"]),dict(present_types=[A,A])):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_review({**review(self.record,present=[A]),**change},center=5.,window_sec=4.,video_sha256=self.record.sha256)

    def test_absence_cannot_contradict_alias_or_overlapping_positive(self):
        for center,whole in ((5.,False),(6.,False),(68.,True)):
            rows={"X-4":row("X-4",review(self.record,present=[A])),
                  "X-47":row("X-47",review(self.record,center=center,absent=[A],whole=[A] if whole else []),time=str(center))}
            with self.subTest(center=center),self.assertRaisesRegex(ValueError,"Conflicting|overlaps"):
                reviewed_samples([self.record],rows,4.)

    def test_window_absence_is_not_whole_video_false_positive(self):
        sample=reviewed_samples([self.record],{"X-4":row("X-4",review(self.record,absent=[A])),"X-47":row("X-47")},4.)[0]
        scan={"X-4":dict(typed_events=[dict(event_type=A,timestamp_sec=30.)],windows=[],scan_config=dict(threshold=.5))}
        result=temporal_summary([sample],scan,[A])["per_class"][0]
        self.assertEqual(result["unresolved_predictions"],1)
        self.assertEqual(result["confirmed_background_false_positives"],0)
        sample.whole_video_absent_types=(A,)
        self.assertEqual(temporal_summary([sample],scan,[A])["per_class"][0]["confirmed_background_false_positives"],1)

    def test_partial_review_ignores_unreviewed_source_and_restores_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            events=Path(directory)/"events.csv"; split=Path(directory)/"split.csv"
            split.write_text("synthetic")
            rows=[row("X-4",review(self.record,present=[A])),row("X-47",review(self.record,center=68.,absent=[A]),time="01:08")]
            with events.open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,rows[0].keys());writer.writeheader();writer.writerows(rows)
            config=dict(events_path=str(events),split_path=str(split),video_dir=directory,window_sec=4.,
                        review=dict(label_mode="per_type_v1",events_sha256=sha256(events),listed_actions_verified=True))
            with patch("overfitted_underqualified.ml.action_data.load_records",return_value=[self.record]):
                _,_,audit=prepare_action_data(config)
            self.assertEqual(audit["class_names"],[A]);self.assertEqual(audit["training_blockers"],[])
            validate_review_manifest(audit["manifest"],[A],4.)
            audit["manifest"]["points"][0]["negative_types"]=[B]
            with self.assertRaises(ValueError):validate_review_manifest(audit["manifest"],[A],4.)
            # Unsupported, unreviewed tags do not become background or training targets.
            self.assertEqual(mapped_types("employee_handles_suspected_cash_without_client"),[])
            self.assertEqual(mapped_types("employee_steps_away_suspected_cash_on_desk"),[])
            config['target_types']=[A]
            with patch('overfitted_underqualified.ml.action_data.load_records',return_value=[self.record]):
                _,_,selected=prepare_action_data(config)
                self.assertEqual(selected['class_names'],[A])
                config['target_types']=[B]
                with self.assertRaisesRegex(ValueError,'train positives'):prepare_action_data(config)

    def test_import_dry_run_stale_guard_and_only_review_column_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);events=root/"data/annotations/events.csv";events.parent.mkdir(parents=True)
            split=root/"split.csv";split.write_text("synthetic")
            rows=[row("X-4"),row("X-47",time="01:08")]
            with events.open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,rows[0].keys());writer.writeheader();writer.writerows(rows)
            original=events.read_bytes()
            decisions=dict(schema="action_review_decisions_v1",events_sha256=sha256(events),split_sha256=sha256(split),
                           catalog_sha256=sha256(CATALOG_PATH),rows=[dict(clip_id="X-4",review=review(self.record,present=[A]))])
            with patch("overfitted_underqualified.ml.dataset.load_records",return_value=[self.record]):
                result=apply_decisions(events,split,root,decisions,4.)
                self.assertFalse(result["applied"]);self.assertEqual(events.read_bytes(),original)
                result=apply_decisions(events,split,root,decisions,4.,apply=True)
                self.assertEqual(Path(result["historical_events_path"]).read_bytes(),original)
                with events.open(encoding="utf-8",newline="") as stream:changed=list(csv.DictReader(stream))
                self.assertEqual(changed[0]["timecode"],"00:05");self.assertEqual(changed[1],rows[1])
                self.assertEqual(json.loads(changed[0]["action_review"])["present_types"],[A])
                with self.assertRaisesRegex(ValueError,"stale"):apply_decisions(events,split,root,decisions,4.)
                decisions["events_sha256"]=sha256(events)
                with self.assertRaisesRegex(ValueError,"Existing action review"):
                    apply_decisions(events,split,root,decisions,4.)


if __name__=="__main__":unittest.main()
