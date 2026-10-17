"""Video aggregation must not overweight videos with multiple source events."""
from pathlib import Path
import sys
import unittest
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from overfitted_underqualified.ml.evaluate import aggregate_video_predictions, write_evaluation_report
from overfitted_underqualified.ml.dataset import sha256


class VideoEvaluationTests(unittest.TestCase):
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
