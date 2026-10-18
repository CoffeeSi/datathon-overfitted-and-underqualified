"""Scope/provenance tests use synthetic scans and do not measure model quality."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from overfitted_underqualified.ml import banknote_evaluation as evaluation
from overfitted_underqualified.ml.dataset import sha256
from overfitted_underqualified.ml.primitive_data import CLASSES, PrimitiveSample


def record(name, last_time=3.4):
    return SimpleNamespace(clip_id=name, aliases=(name,), sha256=name.ljust(64, "a"),
                           last_time=last_time, fps=10., frame_count=35, split="validation",
                           path=Path(name + ".mp4"))


def grid_review(video, references=()):
    return dict(schema=evaluation.GRID_SCHEMA, visual_target=evaluation.TARGET,
                mapped_candidate_type=evaluation.EVENT_TYPE, video_sha256=video.sha256,
                sampled_seconds=evaluation._grid(video.last_time), grid_step_sec=1.,
                complete_for_sampled_grid=True, continuous_legal_review=False,
                reference_times_sec=list(references), method="full_duration_sampled_frame_sequence_review",
                reviewer="synthetic_test", reviewed_at="2026-10-18", note="Synthetic fixture",
                evidence_manifest="fixture.json")


def source_row(video, review=None, action_present=False):
    primitive = dict(points=[dict(time_sec=2., labels={CLASSES[0]: 1, CLASSES[1]: 0},
                                 banknote_boxes=[], banknote_boxes_complete=True,
                                 banknote_box_review=dict(method="synthetic_test"))])
    if review is not None:
        primitive["sequence_grid_review"] = review
    action = dict(video_sha256=video.sha256, center_sec=3.,
                  present_types=[evaluation.EVENT_TYPE] if action_present else [])
    return dict(clip_id=video.clip_id, primitive_review=json.dumps(primitive), action_review=json.dumps(action))


def scan(video, times=()):
    return dict(clip_id=video.clip_id, video_sha256=video.sha256, processing_sec=.1,
                events=[dict(event_type=evaluation.EVENT_TYPE, timestamp_sec=t) for t in times],
                observations=[dict(time_sec=t) for t in evaluation._grid(video.last_time)],
                fps=video.fps, frame_count=video.frame_count, uncertain_observations=0)


class BanknoteEvaluationTests(unittest.TestCase):
    def test_complete_grid_requires_identity_scope_endpoints_and_references(self):
        video = record("X-38")
        good = grid_review(video, [3.])
        self.assertEqual(evaluation.validate_sequence_grid(good, video), good)
        for mutation in ("hash", "end", "hole", "duplicate", "scope", "target", "reference", "evidence"):
            bad = copy.deepcopy(good)
            if mutation == "hash": bad["video_sha256"] = "wrong"
            if mutation == "end": bad["sampled_seconds"].pop()
            if mutation == "hole": bad["sampled_seconds"].pop(1)
            if mutation == "duplicate": bad["sampled_seconds"][1] = 0.
            if mutation == "scope": bad["continuous_legal_review"] = True
            if mutation == "target": bad["visual_target"] = "legal_violation"
            if mutation == "reference": bad["reference_times_sec"] = [3.2]
            if mutation == "evidence": bad.pop("evidence_manifest")
            with self.assertRaises(ValueError, msg=mutation):
                evaluation.validate_sequence_grid(bad, video)

    def test_unknown_and_incomplete_video_reviews_are_not_negative_labels(self):
        videos = [record(name) for name in ("X-2", "X-31", "X-38", "X-39", "X-10", "X-14")]
        source = {}
        for video in videos:
            review = grid_review(video, [3.] if video.clip_id == "X-38" else [])
            if video.clip_id == "X-10": review = None
            if video.clip_id == "X-14":
                review["complete_for_sampled_grid"] = False
                review["sampled_seconds"] = [1.]
            source[video.clip_id] = source_row(video, review)
        scans = [scan(v, [3.] if v.clip_id in ("X-38", "X-10", "X-14") else []) for v in videos]
        result = evaluation.summarize_grid_sequences(videos, scans, source, 1.)
        self.assertEqual(result["video"]["confusion_matrix"], [[3, 0], [0, 1]])
        self.assertEqual(result["video"]["f1"], 1.)
        self.assertEqual(result["video"]["videos"], 4)
        self.assertEqual(result["excluded_unknown_videos"], ["X-10", "X-14"])
        self.assertFalse(result["continuous_legal_review"])
        self.assertFalse(result["independent_evaluation"])

    def test_video_success_does_not_hide_temporal_miss_or_duplicate_alarm(self):
        videos = [record("positive", 10.), record("negative", 10.)]
        source = {v.clip_id: source_row(v, grid_review(v, [8.] if i == 0 else [])) for i, v in enumerate(videos)}
        result = evaluation.summarize_grid_sequences(videos, [scan(videos[0], [1., 2.]), scan(videos[1])], source, 1.)
        self.assertEqual(result["video"]["f1"], 1.)
        self.assertEqual(result["temporal"]["matched"], 0)
        self.assertEqual(result["temporal"]["missed"], 1)
        self.assertEqual(result["temporal"]["unmatched_predictions"], 2)
        self.assertEqual(result["temporal"]["f1"], 0.)
        self.assertIsNone(result["temporal"]["timestamp_mae_sec"])

    def test_duplicate_alias_conflicts_and_scan_hash_mismatch_are_rejected(self):
        video = record("X-4")
        video.aliases = ("X-4", "X-47")
        first, second = grid_review(video), grid_review(video, [3.])
        source = {alias: source_row(video, review) for alias, review in zip(video.aliases, (first, second))}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            evaluation.summarize_grid_sequences([video], [scan(video)], source, 1.)
        bad = scan(video); bad["video_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hash"):
            evaluation.summarize_grid_sequences([video], [bad], source, 1.)

    def test_source_extension_preserves_trained_points_and_split(self):
        old = dict(events_sha256="old", split_sha256="split", videos=[dict(clip_id="X-1", split="train")],
                   reviews=[dict(clip_id="X-1", source_id="X-1", review=dict(points=[
                       dict(time_sec=2., labels={CLASSES[0]: 1, CLASSES[1]: 0}, note="original")]))])
        current = copy.deepcopy(old)
        current["events_sha256"] = "new"
        current["reviews"][0]["review"]["sequence_grid_review"] = {}
        point = current["reviews"][0]["review"]["points"][0]
        point.update(banknote_boxes=[], banknote_boxes_complete=True, banknote_box_review={})
        evaluation.verify_primitive_manifest(old, current)
        for mutation in ("label", "point", "split", "evidence"):
            bad = copy.deepcopy(current)
            if mutation == "label": bad["reviews"][0]["review"]["points"][0]["labels"][CLASSES[0]] = 0
            if mutation == "point": bad["reviews"][0]["review"]["points"].append(copy.deepcopy(point))
            if mutation == "split": bad["videos"][0]["split"] = "validation"
            if mutation == "evidence": bad["reviews"][0]["review"]["points"][0]["note"] = "changed"
            with self.assertRaisesRegex(ValueError, "differ"):
                evaluation.verify_primitive_manifest(old, bad)

    def test_existing_report_or_archive_blocks_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "result"
            with patch.object(evaluation, "PrimitiveScanner") as scanner:
                out.mkdir()
                with self.assertRaises(FileExistsError):
                    evaluation.evaluate_banknote_pipeline("a", "b", "e", "s", "v", out)
                out.rmdir(); out.with_suffix(".zip").write_bytes(b"keep")
                with self.assertRaises(FileExistsError):
                    evaluation.evaluate_banknote_pipeline("a", "b", "e", "s", "v", out)
                scanner.assert_not_called()
                self.assertEqual(out.with_suffix(".zip").read_bytes(), b"keep")

    def test_scan_tracking_configuration_is_pinned(self):
        video = record("X-38")
        tracking = dict(schema="synthetic_image_tracking_v1", max_gap_sec=2.)
        scanner = SimpleNamespace(config=dict(scan_step_sec=1.), checkpoint_sha256="primitive",
                                  banknote_detector=SimpleNamespace(checkpoint_sha256="banknote"),
                                  banknote_tracking_config=tracking)
        result = dict(scan(video), checkpoint_sha256="primitive", banknote_checkpoint_sha256="banknote",
                      banknote_tracking_config=copy.deepcopy(tracking))
        evaluation._verify_scan(video, result, scanner)
        for wrong in (None, dict(tracking, max_gap_sec=3.)):
            result["banknote_tracking_config"] = wrong
            with self.assertRaisesRegex(ValueError, "tracking protocol"):
                evaluation._verify_scan(video, result, scanner)

    def test_pipeline_scans_all_validation_but_reports_proxy_on_reviewed_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, split = root / "events.csv", root / "split.csv"
            primitive, banknote = root / "primitive.pt", root / "banknote.pt"
            primitive.write_bytes(b"primitive"); banknote.write_bytes(b"banknote"); split.write_bytes(b"fixed split")
            videos = [record(name) for name in ("X-2", "X-10", "X-14", "X-16", "X-28", "X-31", "X-34", "X-38", "X-39", "X-44")]
            for video in videos:
                video.path = root / (video.clip_id + ".mp4")
                video.path.write_bytes(video.clip_id.encode())
                video.sha256 = sha256(video.path)
            reviewed = {"X-2", "X-31", "X-38", "X-39"}
            rows = [source_row(v, grid_review(v, [3.] if v.clip_id == "X-38" else [])
                               if v.clip_id in reviewed else None, v.clip_id == "X-38") for v in videos]
            with events.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, list(rows[0])); writer.writeheader(); writer.writerows(rows)
            old_manifest = dict(events_sha256="old", split_sha256=sha256(split),
                                reviews=[dict(clip_id=r["clip_id"], source_id=r["clip_id"], review=json.loads(r["primitive_review"])) for r in rows])
            for row in old_manifest["reviews"]:
                row["review"].pop("sequence_grid_review", None)
                for point in row["review"]["points"]:
                    for key in ("banknote_boxes", "banknote_boxes_complete", "banknote_box_review"):
                        point.pop(key, None)
            current_manifest = copy.deepcopy(old_manifest)
            current_manifest["events_sha256"] = sha256(events)
            current_manifest["reviews"] = [dict(clip_id=r["clip_id"], source_id=r["clip_id"], review=json.loads(r["primitive_review"])) for r in rows]
            box_manifest = dict(events_sha256=sha256(events), split_sha256=sha256(split), samples=[])
            cfg = dict(events_sha256="old", split_sha256=sha256(split), threshold=.7, window_sec=2.,
                       scan_step_sec=1., match_tolerance_sec=1.)
            detector = SimpleNamespace(config=dict(events_sha256=sha256(events), split_sha256=sha256(split)),
                checkpoint=dict(config={"preserved": True}, data_manifest=box_manifest), checkpoint_sha256=sha256(banknote),
                predict=lambda rgb: dict(score=.1, boxes=[]))
            calls = []
            def fake_scan(path):
                calls.append(path)
                video = next(v for v in videos if v.path == path)
                result = scan(video, [3.] if video.clip_id in {"X-38", "X-14"} else [])
                return dict(result, checkpoint_sha256=sha256(primitive), banknote_checkpoint_sha256=sha256(banknote),
                            banknote_tracking_config=copy.deepcopy(scanner.banknote_tracking_config))
            scanner = SimpleNamespace(config=cfg, checkpoint=dict(config=cfg, data_manifest=old_manifest),
                                      checkpoint_sha256=sha256(primitive), banknote_detector=detector,
                                      transform=lambda x: x, model=None, device="cpu", scan=fake_scan,
                                      banknote_tracking_config=None)
            samples = [PrimitiveSample(v, 2., (CLASSES[0],), (CLASSES[1],)) for v in videos]
            predictions = [dict(clip_id=v.clip_id, center_sec=2., probabilities=[.9, .8, .2],
                                targets=[1., 0., 0.], mask=[1., 1., 0.]) for v in videos]
            with patch.object(evaluation, "PrimitiveScanner", return_value=scanner), \
                    patch.object(evaluation, "prepare_primitive_data", return_value=(videos, samples, dict(manifest=current_manifest, coverage=[]))), \
                    patch.object(evaluation, "prepare_banknote_data", return_value=([], box_manifest)), \
                    patch.object(evaluation, "action_epoch", side_effect=lambda *args: ({"old_metric": True}, copy.deepcopy(predictions))), \
                    patch.object(evaluation, "decode_frames", return_value=torch.zeros(1, 3, 4, 4, dtype=torch.uint8)):
                out = root / "report"
                result = evaluation.evaluate_banknote_pipeline(primitive, banknote, events, split, root, out)
                scanner.banknote_tracking_config = dict(schema="synthetic_image_tracking_v1", max_gap_sec=2.)
                tracked_out = root / "tracked_report"
                tracked = evaluation.evaluate_banknote_pipeline(primitive, banknote, events, split, root,
                                                                tracked_out, track_banknotes=True)
            self.assertEqual(calls, [v.path for v in videos] * 2)
            self.assertEqual(result["validation_videos"], 10)
            self.assertEqual(result["sampled_grid_proxy"]["video"]["confusion_matrix"], [[3, 0], [0, 1]])
            self.assertIsNone(result["known_action_markers"]["precision"])
            self.assertIsNone(result["known_action_markers"]["f1"])
            self.assertEqual(result["known_action_markers"]["recall"], 1.)
            self.assertFalse(result["banknote_tracking_enabled"])
            self.assertIsNone(result["banknote_tracking_config"])
            self.assertTrue(tracked["banknote_tracking_enabled"])
            self.assertEqual(tracked["banknote_tracking_config"], scanner.banknote_tracking_config)
            self.assertEqual(tracked["primitive_metrics"], result["primitive_metrics"])
            self.assertEqual(json.loads((tracked_out / "evaluation_config.json").read_text())["banknote_tracking_config"],
                             scanner.banknote_tracking_config)
            self.assertEqual(json.loads((tracked_out / "primitive_predictions.json").read_text()),
                             json.loads((out / "primitive_predictions.json").read_text()))
            saved = json.loads((out / "primitive_predictions.json").read_text())
            self.assertEqual(saved[0]["r3d_probabilities"], [.9, .8, .2])
            self.assertEqual(saved[0]["probabilities"], [.9, .1, .2])
            self.assertEqual(json.loads((out / "primitive_checkpoint_config.json").read_text()), cfg)
            with ZipFile(out.with_suffix(".zip")) as archive:
                self.assertTrue(all(name.endswith(".json") for name in archive.namelist()))
                self.assertEqual(len([name for name in archive.namelist() if name.endswith("_scan.json")]), 10)


if __name__ == "__main__":
    unittest.main()
