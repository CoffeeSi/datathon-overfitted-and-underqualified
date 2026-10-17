"""Video aggregation must not overweight videos with multiple source events."""
from pathlib import Path
import sys
import unittest
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from overfitted_underqualified.ml.evaluate import (
    aggregate_video_predictions, write_evaluation_report, save_evaluation_report, archive_evaluation_report,
    resolve_checkpoint,
)
from overfitted_underqualified.ml.dataset import sha256


class VideoEvaluationTests(unittest.TestCase):
    def test_checkpoint_selection_requires_real_file_and_unambiguous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "Brev"):
                resolve_checkpoint(root=root)
            first = root / "runs/experiment_one/best.pt"
            first.parent.mkdir(parents=True)
            first.write_bytes(b"checkpoint")
            self.assertEqual(resolve_checkpoint(root=root), first.resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint("runs/r3d18_smoke/best.pt", root=root)
            second = root / "runs/experiment_two/best.pt"
            second.parent.mkdir(parents=True)
            second.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "Multiple checkpoints"):
                resolve_checkpoint(root=root)
            self.assertEqual(resolve_checkpoint("runs/experiment_two/best.pt", root=root), second.resolve())

    def test_multiwindow_video_counts_once_with_fixed_max_rule(self):
        rows = [dict(clip_id=clip_id, label=label, probability=probability)
                for clip_id, label, probability in (
                    ("X-4", 1, 0.1), ("X-4", 1, 0.8),
                    ("X-2", 0, 0.5), ("X-3", 1, 0.4), ("X-6", 0, 0.2))]
        metrics, videos = aggregate_video_predictions(rows)
        self.assertEqual(metrics["videos"], 4)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        for key in ("accuracy", "precision", "recall", "f1"):
            self.assertEqual(metrics[key], 0.5)
        self.assertEqual(videos[0]["windows"], 2)
        self.assertEqual(videos[0]["outcome"], "TP")
        self.assertEqual(videos[1]["outcome"], "FP")

    def test_rejects_empty_conflicting_and_nonfinite_predictions(self):
        for rows in ([], [dict(clip_id="X-1", label=1, probability=float("nan"))],
                     [dict(clip_id="X-1", label=1, probability=0.8),
                      dict(clip_id="X-1", label=0, probability=0.2)]):
            with self.assertRaises(ValueError):
                aggregate_video_predictions(rows)

    def test_report_binds_provenance_times_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, events, split = (root / name for name in ("best.pt", "events.csv", "split.csv"))
            for path in (checkpoint, events, split):
                path.write_bytes(path.name.encode())
            manifest = dict(events_sha256=sha256(events), split_sha256=sha256(split))
            config = dict(events_path=str(events), split_path=str(split), batch_size=2)
            prepared = dict(device=torch.device("cpu"), config=config,
                            dataset=SimpleNamespace(samples=[(SimpleNamespace(clip_id="X-4"), 57),
                                                             (SimpleNamespace(clip_id="X-4"), 68)]),
                            checkpoint=dict(epoch=1, config=config, data_manifest=manifest))
            rows = [dict(clip_id="X-4", label=1, probability=p, center_sec=t, prediction=int(p >= 0.5))
                    for p, t in ((0.1, 57), (0.8, 68))]
            output = root / "report"
            with patch("overfitted_underqualified.ml.evaluate.prepare_evaluation", return_value=prepared), \
                 patch("overfitted_underqualified.ml.evaluate.evaluate_prepared", return_value=({}, rows)):
                report = write_evaluation_report(checkpoint, output)
            self.assertEqual(report["checkpoint_sha256"], sha256(checkpoint))
            self.assertEqual(report["video_metrics"]["videos"], 1)
            self.assertGreater(report["timing"]["processing_sec"], 0)
            self.assertEqual(json.loads((output / "evaluation.json").read_text())["data_manifest"], manifest)
            for name in ("video_predictions.csv", "window_predictions.csv", "confusion_matrix.png", "report.md"):
                self.assertGreater((output / name).stat().st_size, 0)
            with self.assertRaises(FileExistsError):
                write_evaluation_report(checkpoint, output)
            # Missing the second event of the same video must not pass video-ID coverage.
            with patch("overfitted_underqualified.ml.evaluate.prepare_evaluation", return_value=prepared), \
                 patch("overfitted_underqualified.ml.evaluate.evaluate_prepared", return_value=({}, rows[:1])):
                with self.assertRaisesRegex(ValueError, "window coverage"):
                    write_evaluation_report(checkpoint, root / "incomplete")

    def test_export_cached_results_without_inference_and_archive_only_allowed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"weights-must-not-be-archived")
            events, split = root / "events.csv", root / "split.csv"
            events.write_text("events")
            split.write_text("split")
            manifest = dict(events_sha256=sha256(events), split_sha256=sha256(split))
            prepared = dict(device=torch.device("cpu"),
                            config=dict(events_path=str(events), split_path=str(split), batch_size=2),
                            checkpoint=dict(epoch=1, config={}, data_manifest=manifest),
                            checkpoint_path=checkpoint.resolve(), checkpoint_sha256=sha256(checkpoint),
                            dataset=SimpleNamespace(samples=[(SimpleNamespace(clip_id="X-1"), 2.0)]))
            rows = [dict(clip_id="X-1", center_sec=2.0, label=0, probability=0.8, prediction=1)]
            output = root / "evaluation_export"
            with patch("overfitted_underqualified.ml.evaluate.evaluate_prepared", side_effect=AssertionError("Repeated inference")), \
                 patch("overfitted_underqualified.ml.evaluate.load_checkpoint", side_effect=AssertionError("Reloaded model")):
                report = save_evaluation_report(prepared, checkpoint, output, {"windows": 1}, rows,
                                                 processing_sec=0.5, preparation_sec=1.0)
            self.assertEqual(report["video_metrics"]["confusion_matrix"], [[0, 1], [0, 0]])
            self.assertEqual(report["timing"]["processing_sec"], 0.5)
            for name in ("best.pt", ".env", "bank_frame.png"):
                (output / name).write_bytes(b"private")
            (output / "error_analysis.csv").write_text("clip_id,outcome\nX-1,FP\n")
            (root / "data_manifest.json").write_text(json.dumps(manifest))
            (root / "history.json").write_text('[{"epoch": 1}]')
            archive = archive_evaluation_report(output, checkpoint)
            with ZipFile(archive) as stream:
                names = {Path(name).name for name in stream.namelist()}
                self.assertTrue({"evaluation.json", "report.md", "confusion_matrix.png", "error_analysis.csv", "training_history.json"} <= names)
                self.assertFalse({"best.pt", ".env", "bank_frame.png"} & names)
                self.assertIsNone(stream.testzip())
            with self.assertRaises(FileExistsError):
                archive_evaluation_report(output, checkpoint)
