"""Full-video scans must be annotation-free, immutable and auditable."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights

from overfitted_underqualified.backend.schemas import EvidenceFrame
from overfitted_underqualified.ml.dataset import VideoRecord, sha256
from overfitted_underqualified.ml.inference import (
    ActionObservation, FullVideoScanner, ScanConfig, evidence_frames, inspect_video,
    link_action, merge_candidates, prepare_scanner, sliding_windows,
)
from overfitted_underqualified.ml.full_video_evaluation import (
    archive_full_report, match_events, save_full_report, summarize_scans,
)
from overfitted_underqualified.rules.coverage import catalog_coverage


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm3d(3)
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(self.bn(x).mean(dim=(2, 3, 4)))


class FullVideoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "arbitrary-name.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for i in range(30):
            writer.write(np.full((48, 64, 3), i * 5, np.uint8))
        writer.release()
        self.model = TinyModel()
        self.config = ScanConfig(window_sec=1, step_sec=0.5, batch_size=2)
        self.scanner = FullVideoScanner(self.model, R3D_18_Weights.KINETICS400_V1.transforms(), self.config)

    def tearDown(self):
        self.temp.cleanup()

    def test_windows_cover_boundaries_short_and_single_frame_videos(self):
        for last in (0, 0.1, 1.0, 2.9, 8.1):
            rows = sliding_windows(last, self.config)
            self.assertEqual(rows[0]["start_sec"], 0)
            self.assertAlmostEqual(rows[-1]["end_sec"], last)
            self.assertTrue(all(b["start_sec"] <= a["end_sec"] for a, b in zip(rows, rows[1:])))
            self.assertEqual(len({r["start_sec"] for r in rows}), len(rows))
        self.assertEqual(len(sliding_windows(0.3, self.config)), 1)
        for kwargs in (dict(step_sec=5), dict(threshold=float("nan")), dict(batch_size=0), dict(merge_gap_sec=-1)):
            with self.assertRaises(ValueError):
                ScanConfig(**kwargs)

    def test_scan_has_no_ground_truth_dependency_and_preserves_parameters_bn(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.train()
        original_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.suffix in {".csv", ".json"}:
                raise AssertionError("Scanner tried to read annotations")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded_open):
            result = self.scanner.scan(self.video)
        self.assertEqual(len(result["windows"]), 5)
        self.assertEqual(result["sha256"], sha256(self.video))
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(before[name], tensor, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(not layer.training for layer in self.model.modules()))
        self.assertTrue(all(c["event_type"] is None and c["type_confidence"] is None and not c["source_links"]
                            for c in result["candidates"]))
        # Identity and annotation-free filename cannot affect predictions.
        renamed = self.root / "X-47.avi"
        renamed.write_bytes(self.video.read_bytes())
        self.assertEqual(result["windows"], self.scanner.scan(renamed)["windows"])

    def test_corrupt_and_failed_decode_are_errors_not_skipped_windows(self):
        bad = self.root / "bad.avi"
        bad.write_bytes(b"not-video")
        with self.assertRaises(ValueError):
            inspect_video(bad)
        capture = cv2.VideoCapture
        class BrokenDecode:
            def __init__(self, path):
                self.cap = capture(path)
            def read(self):
                return False, None
            def __getattr__(self, name):
                return getattr(self.cap, name)
        with patch("overfitted_underqualified.ml.dataset.cv2.VideoCapture", BrokenDecode):
            with self.assertRaisesRegex(ValueError, "Cannot decode frame"):
                self.scanner.scan(self.video)

    def test_merging_peak_tie_and_gap_are_deterministic(self):
        config = ScanConfig(window_sec=1, step_sec=1)
        windows = sliding_windows(5, config)
        for window, probability in zip(windows, (.8, .9, .1, .9, .9)):
            window["probability"] = probability
        result = merge_candidates(list(reversed(windows)), config)
        self.assertEqual([c["window_indices"] for c in result], [[0, 1], [3, 4]])
        self.assertEqual([c["timestamp_sec"] for c in result], [1.5, 3.5])
        bridged = merge_candidates(windows, ScanConfig(window_sec=1, step_sec=1, merge_gap_sec=1))
        self.assertEqual(len(bridged), 1)
        self.assertEqual(bridged[0]["timestamp_sec"], 1.5)

    def test_evidence_backend_jpeg_and_actual_frame_positions(self):
        frames = evidence_frames(self.video, 1.0)
        self.assertEqual(len(frames), 3)
        for frame in frames:
            EvidenceFrame(**{k: frame[k] for k in ("timestamp_sec", "image_base64")})
            self.assertAlmostEqual(frame["timestamp_sec"], frame["frame_index"] / 10, places=4)
        self.assertLessEqual(len(evidence_frames(self.video, 0)), 3)
        with self.assertRaises(ValueError):
            evidence_frames(self.video, 5)

    def test_one_to_one_maximum_matching_preserves_two_canonical_markers(self):
        # Greedy nearest-first could steal annotation 2 from prediction 2.5.
        rows = match_events([1, 2.5], [0, 2], 1)
        self.assertEqual(sum(r["status"] == "matched" for r in rows), 2)
        rows = match_events([57.1], [57, 68], 12)
        self.assertEqual(sum(r["status"] == "matched" for r in rows), 1)
        self.assertEqual(sum(r["status"] == "unmatched_annotation" for r in rows), 1)
        rows = match_events([56, 57.1, 68.2], [57, 68], 1)
        self.assertEqual(sum(r["status"] == "matched" for r in rows), 2)
        self.assertEqual(next(r["predicted_sec"] for r in rows if r["status"] == "unmatched_prediction"), 56)
        self.assertEqual(len(match_events([], [57, 68])), 2)
        self.assertEqual(match_events([], []), [])

    def test_type_linking_requires_separate_supported_observation(self):
        result = link_action(ActionObservation("unsecured_accepted_cash", .7, "test-action-component", "visible cash on desk"))
        self.assertTrue(result["source_links"])
        self.assertTrue(result["applicability"])
        self.assertTrue(result["video_cannot_establish"])
        self.assertIsNone(result["legal_conclusion"])
        with self.assertRaises(ValueError):
            link_action(ActionObservation("violation", .9, "binary-r3d", "alarm"))

    def test_full_report_provenance_missing_events_and_zip_privacy(self):
        info = inspect_video(self.video)
        record = VideoRecord("X-4", self.video, "validation", 1, sha256(self.video), ("X-4", "X-47"),
                             (0.1, 2.8), info.fps, info.frame_count)
        scan = dict(clip_id=record.clip_id, **self.scanner.scan(self.video))
        # Long continuous alarm merges to one event; it cannot cover both markers.
        for window in scan["windows"]:
            window.update(probability=.9, prediction=1)
        scan["candidates"] = merge_candidates(scan["windows"], self.config)
        summary = summarize_scans([record], [scan], 3)
        self.assertEqual(summary["metrics"]["temporal"]["matched"], 1)
        self.assertEqual(summary["metrics"]["temporal"]["unmatched_annotations"], 1)
        narrow = summarize_scans([record], [scan], .01)
        self.assertEqual(narrow["metrics"]["temporal"]["unresolved_positive_candidates"], 1)
        self.assertIsNone(narrow["metrics"]["temporal"]["matched_mae_sec"])
        checkpoint, events, split = [self.root / name for name in ("best.pt", "events.csv", "split.csv")]
        for path in (checkpoint, events, split):
            path.write_bytes(path.name.encode())
        manifest = dict(events_sha256=sha256(events), split_sha256=sha256(split))
        p = dict(checkpoint_path=checkpoint, checkpoint_sha256=sha256(checkpoint), device=torch.device("cpu"),
                 config=dict(events_path=str(events), split_path=str(split)),
                 checkpoint=dict(config={}, epoch=1, data_manifest=manifest))
        prepared = dict(evaluation=p, scanner=self.scanner, records=[record], preparation_sec=.1)
        result = dict(scans=[scan], tolerance_sec=3, processing_sec=.5)
        output = self.root / "report"
        with patch.object(self.scanner, "scan", side_effect=AssertionError("Repeated inference")):
            report = save_full_report(prepared, result, output)
        self.assertEqual(report["checkpoint_sha256"], sha256(checkpoint))
        self.assertEqual(report["evaluated_videos"][0]["event_times"], [0.1, 2.8])
        for name in ("bank.png", ".env", "best.pt"):
            (output / name).write_bytes(b"private")
        archive = archive_full_report(output)
        with ZipFile(archive) as stream:
            self.assertEqual(len(stream.namelist()), 7)
            self.assertIsNone(stream.testzip())
            self.assertFalse(any(Path(n).name in {"bank.png", ".env", "best.pt"} for n in stream.namelist()))
        with self.assertRaises(FileExistsError):
            save_full_report(prepared, result, output)
        with self.assertRaises(FileExistsError):
            archive_full_report(output)
        scan["windows"].pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            summarize_scans([record], [scan], 3)
        events.write_bytes(b"drift")
        with self.assertRaisesRegex(ValueError, "events changed"):
            save_full_report(prepared, result, self.root / "new-report")

    def test_catalog_counts_duplicates_once_but_preserves_markers(self):
        root = Path(__file__).resolve().parents[1]
        audit = catalog_coverage(root / "data/annotations/events.csv", root / "data/processed/split_proposal.csv")
        cash = next(r for r in audit["rows"] if r["event_type"] == "unsecured_accepted_cash")
        self.assertEqual(cash["source_records"], 5)
        self.assertEqual(cash["canonical_videos"], 4)
        self.assertEqual(cash["distinct_markers"], 5)
        self.assertFalse(cash["verified_multiclass_targets"])
        self.assertTrue(cash['raw_catalog_hypotheses_only'])
        self.assertEqual(cash['usable_proposal_source_ids'],['X-4','X-47'])
        self.assertEqual(cash['usable_proposal_canonical_videos'],1)
        self.assertEqual(set(cash['needs_context_review_ids']),{'X-15','X-38','X-44'})


if __name__ == "__main__":
    unittest.main()
