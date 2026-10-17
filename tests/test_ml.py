import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import R3D_18_Weights, r3d_18

from overfitted_underqualified.ml.dataset import (
    VideoWindowDataset, decode_frames, load_records, sample_times, sha256,
)
from overfitted_underqualified.ml.train import (
    run_epoch, seed_everything, prepare_training, build_model, save_checkpoint,
)
from overfitted_underqualified.ml.evaluate import prepare_evaluation, evaluate_prepared


class TinyVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm3d(3)
        self.bn.requires_grad_(False)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(self.bn(x).mean(dim=(2, 3, 4)))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.events = self.root / "events.csv"
        self.split = self.root / "split.csv"
        source, assignments = [], []
        for i, (subset, label) in enumerate([
                ("train", "violation"), ("train", "no_violation"),
                ("validation", "violation"), ("validation", "no_violation")], start=1):
            clip_id = f"X-{i}"
            path = self.root / f"{clip_id}.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for frame in range(30):
                writer.write(np.full((48, 64, 3), (i * 20 + frame, i * 30, i * 40), dtype=np.uint8))
            writer.release()
            positive = label == "violation"
            source.append(dict(clip_id=clip_id, source_label=label, timecode="00:01" if positive else ""))
            assignments.append(dict(clip_id=clip_id, source_label=label, split=subset,
                                    sha256=sha256(path), source_clip_ids=json.dumps([clip_id]),
                                    event_times_sec=json.dumps([1.0] if positive else [])))
        # An exact duplicate has a second positive event, but only one canonical video.
        shutil.copyfile(self.root / "X-1.avi", self.root / "X-5.avi")
        source.append(dict(clip_id="X-5", source_label="violation", timecode="00:02"))
        assignments[0].update(source_clip_ids='["X-1", "X-5"]', event_times_sec='[1.0, 2.0]')
        self.write_csv(self.events, source)
        self.write_csv(self.split, assignments)
        self.source = source
        self.assignments = assignments

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def records(self):
        return load_records(self.events, self.split, self.root)

    def test_real_decode_preprocessing_and_alias_events(self):
        records = self.records()
        transform = R3D_18_Weights.KINETICS400_V1.transforms()
        train = VideoWindowDataset(records, "train", transform)
        validation = VideoWindowDataset(records, "validation", transform)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(validation), 2)
        self.assertEqual([t for r, t in train.samples if r.clip_id == "X-1"], [1.0, 2.0])
        self.assertTrue({r.sha256 for r, _ in train.samples}.isdisjoint(
            {r.sha256 for r, _ in validation.samples}))
        sample = train[0]
        self.assertEqual(tuple(sample["video"].shape), (3, 16, 112, 112))
        self.assertEqual(sample["video"].dtype, torch.float32)
        torch.testing.assert_close(sample["video"], train[0]["video"], rtol=0, atol=0)
        for center in (0, 1, 2.9):
            times = sample_times(center, 2.9, 4)
            self.assertEqual(len(times), 16)
            self.assertGreaterEqual(times.min(), 0)
            self.assertLessEqual(times.max(), 2.9)
            self.assertLessEqual(times.min(), center)
            self.assertGreaterEqual(times.max(), center)

    def test_failed_seek_recovers_sequential_frames(self):
        record = self.records()[0]
        times = sample_times(1, record.last_time, 4)
        expected = decode_frames(record, times)
        capture = cv2.VideoCapture

        class BrokenSeek:
            def __init__(self, path):
                self.cap = capture(path)

            def set(self, *_args):
                return False

            def __getattr__(self, name):
                return getattr(self.cap, name)

        with patch("overfitted_underqualified.ml.dataset.cv2.VideoCapture", BrokenSeek):
            torch.testing.assert_close(decode_frames(record, times), expected)

    def test_rejects_source_drift_and_hash_leakage(self):
        self.source[0]["timecode"] = "00:02"
        self.write_csv(self.events, self.source)
        with self.assertRaisesRegex(ValueError, "Stale event times"):
            self.records()
        self.source[0]["timecode"] = "00:01"
        self.write_csv(self.events, self.source)
        self.assignments[2]["sha256"] = self.assignments[0]["sha256"]
        self.write_csv(self.split, self.assignments)
        with self.assertRaisesRegex(ValueError, "possible leakage"):
            self.records()

    def test_rejects_modified_alias_and_missing_source_coverage(self):
        with (self.root / "X-5.avi").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.records()
        shutil.copyfile(self.root / "X-1.avi", self.root / "X-5.avi")
        self.source.append(dict(clip_id="X-6", source_label="no_violation", timecode=""))
        self.write_csv(self.events, self.source)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            self.records()

    def test_validation_updates_nothing_and_frozen_bn_stays_fixed(self):
        seed_everything(42, 2)
        model = TinyVideoModel()
        samples = [{"video": torch.randn(3, 16, 8, 8), "label": torch.tensor(i % 2),
                    "clip_id": f"X-{i}", "center_sec": 1.0} for i in range(4)]
        loader = DataLoader(samples, batch_size=2)
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.1)
        run_epoch(model, loader, torch.device("cpu"), optimizer, freeze_backbone=True)
        self.assertFalse(torch.equal(initial["fc.weight"], model.fc.weight))
        for key, value in model.bn.state_dict().items():
            torch.testing.assert_close(value, initial[f"bn.{key}"], rtol=0, atol=0)
        trained = {k: v.clone() for k, v in model.state_dict().items()}
        metrics, _ = run_epoch(model, loader, torch.device("cpu"))
        self.assertEqual(metrics["windows"], 4)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, trained[key], rtol=0, atol=0)

    def test_checkpoint_relocation_inference_and_provenance_without_download(self):
        config = dict(events_path=str(self.events), split_path=str(self.split), video_dir=str(self.root),
                      output_dir=str(self.root / "run"), weights_cache=str(self.root / "cache"),
                      seed=42, device="cpu", weights="KINETICS400_V1", freeze_backbone=True,
                      frames=16, window_sec=4.0, negative_windows=1, epochs=1, batch_size=2,
                      num_workers=0, cpu_threads=2, learning_rate=0.001, weight_decay=0.0001)
        prepared = prepare_training(config)
        model = build_model(None, True)
        optimizer = torch.optim.AdamW(model.fc.parameters())
        # Mimic a checkpoint copied from another computer, including Windows -> Linux relocation.
        saved_config = dict(config, events_path="Z:/old/events.csv", split_path="Z:/old/split.csv",
                            video_dir="Z:/old/videos", device="cuda")
        checkpoint_path = self.root / "checkpoint.pt"
        save_checkpoint(checkpoint_path, model, optimizer, 1, saved_config, prepared["manifest"], {})
        with patch("overfitted_underqualified.ml.train.r3d_18", wraps=r3d_18) as constructor:
            evaluation = prepare_evaluation(checkpoint_path, events_path=self.events,
                                            split_path=self.split, video_dir=self.root)
        constructor.assert_called_once_with(weights=None)
        self.assertEqual(evaluation["config"]["device"], "cpu")
        restored = evaluation["model"]
        for key, tensor in restored.state_dict().items():
            torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)
        metrics, rows = evaluate_prepared(evaluation)
        self.assertEqual(metrics["windows"], 2)
        self.assertTrue(all(row["prediction"] == int(row["probability"] >= 0.5) for row in rows))
        self.assertTrue(all(not module.training for module in restored.modules()))
        self.assertTrue(all(parameter.grad is None for parameter in restored.parameters()))
        for key, tensor in restored.state_dict().items():
            torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)
        self.events.write_bytes(self.events.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            prepare_evaluation(checkpoint_path, events_path=self.events,
                               split_path=self.split, video_dir=self.root)


if __name__ == "__main__":
    unittest.main()
