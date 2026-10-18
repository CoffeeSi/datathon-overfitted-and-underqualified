"""Synthetic contract tests, not evidence of action recognition quality."""
import asyncio
import csv
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2
import numpy as np
import torch
from torch import nn
from fastapi.testclient import TestClient

from overfitted_underqualified.ml.action_data import ActionSample, SCHEMA, mapped_types, prepare_action_data
from overfitted_underqualified.ml.action_train import action_epoch, action_metrics, load_action_checkpoint, read_config, run_action_training
from overfitted_underqualified.ml.action_inference import R3DActionRecognizer, TypedVideoPipeline, merge_typed_windows
from overfitted_underqualified.ml.action_evaluation import evaluate_actions, temporal_summary
from overfitted_underqualified.ml.train import save_checkpoint
from overfitted_underqualified.ml.dataset import sha256
from overfitted_underqualified.ml.inference import ActionObservation, ScanConfig, evidence_frames, inspect_video, link_action
from overfitted_underqualified.rules.catalog import CATALOG_PATH
from overfitted_underqualified.backend.local_analyzer import LocalActionAnalyzer, create_analyzer
from overfitted_underqualified.backend.api import create_app, _jury_job_response
from overfitted_underqualified.backend.schemas import TypedAnalysisResult, TypedFinding
from overfitted_underqualified.backend.settings import Settings
from overfitted_underqualified.backend.storage import current_time

A = "safe_key_left_in_lock"
B = "unsecured_accepted_cash"


def linked(name=A):
    return link_action(ActionObservation(name, 0.9, "synthetic-test", "Synthetic test observation"))


def window(index, types=()):
    return dict(window_index=index, start_sec=float(index), end_sec=float(index + 2),
                center_sec=index + 1.0, probability=0.8, actions=[linked(t) for t in types])


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm3d(3)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(self.bn(x).mean((2, 3, 4)))


class ActionTests(unittest.TestCase):
    def test_multilabel_merge_and_abstention_break_continuity(self):
        result = merge_typed_windows([window(0, (A, B)), window(1, (B,)), window(2), window(3, (A,))])
        self.assertEqual([(r["event_type"], r["window_indices"]) for r in result], [(A, [0]), (B, [0, 1]), (A, [3])])
        self.assertEqual(merge_typed_windows([window(0)]), [])
        bad = window(0)
        bad["actions"] = [{"event_type": "violation"}]
        with self.assertRaises(ValueError):
            merge_typed_windows([bad])

    def test_unknown_targets_are_not_negative_metrics(self):
        rows = [dict(targets=[1, 0], mask=[1, 0], probabilities=[0.9, 0.8]),
                dict(targets=[0, 1], mask=[0, 1], probabilities=[0.2, 0.1])]
        metrics = action_metrics(rows, [A, B], 0.7)
        self.assertIsNone(metrics["macro"]["f1"])
        self.assertIsNone(metrics["per_class"][0]["precision"])
        self.assertEqual(metrics["per_class"][1]["recall"], 0)
        self.assertEqual(metrics["per_class"][0]["confusion_matrix"], [[0, 0], [0, 1]])

    def test_training_masks_unknown_and_preserves_backbone_bn(self):
        torch.set_num_threads(2)
        model = Tiny()
        model.bn.requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        batch = dict(video=torch.rand(2, 3, 2, 8, 8), targets=torch.tensor([[1., 0.], [0., 1.]]),
                     mask=torch.tensor([[1., 0.], [1., 1.]]), clip_id=["test1", "test2"], center_sec=[1., 2.])
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.01)
        action_epoch(model, [batch], "cpu", [A, B], 0.7, optimizer)
        self.assertTrue(torch.equal(model.bn.running_mean, before["bn.running_mean"]))
        self.assertTrue(torch.equal(model.bn.weight, before["bn.weight"]))
        self.assertFalse(torch.equal(model.fc.weight, before["fc.weight"]))
        after = {k: v.clone() for k, v in model.state_dict().items()}
        action_epoch(model, [batch], "cpu", [A, B], 0.7)
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in after.items()))

    def test_positive_weighting_uses_only_known_train_labels(self):
        from overfitted_underqualified.ml.action_train import training_positive_weights
        from types import SimpleNamespace
        def sample(split, positive=(), negative=()):
            return SimpleNamespace(record=SimpleNamespace(split=split), positives=positive,
                                   negatives=negative, complete=False)
        samples=[sample('train',[A]),sample('train',negative=[A]),sample('train',negative=[A]),
                 sample('train'),sample('validation',[A])]
        self.assertEqual(training_positive_weights(samples,[A]).tolist(),[2.0])
        with self.assertRaises(ValueError):training_positive_weights(samples,[B])

    def test_review_does_not_imply_complete_labels_and_alias_points_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = [dict(clip_id="X-4", observed_event_type="prior_client_cash_left_during_next_client_entry", timecode="00:57"),
                    dict(clip_id="X-47", observed_event_type="accepted_cash_not_moved_to_lockable_storage", timecode="01:08")]
            with (path / "events.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            (path / "split.csv").write_text("synthetic split", encoding="utf-8")
            config = dict(events_path=str(path / "events.csv"), split_path=str(path / "split.csv"), video_dir=directory,
                          window_sec=4., review=dict(events_sha256=sha256(path / "events.csv"), listed_actions_verified=True,
                                                    complete_action_lists=False))
            record = SimpleNamespace(clip_id="X-4", aliases=("X-4", "X-47"), split="train", last_time=100, sha256="test")
            with patch("overfitted_underqualified.ml.action_data.load_records", return_value=[record]):
                _, samples, audit = prepare_action_data(config)
                self.assertEqual([s.center_sec for s in samples], [57, 68])
                self.assertEqual(audit["training_blockers"], [B])
                with self.assertRaisesRegex(ValueError, "confirmed negative"):
                    run_action_training(config)
            config["review"]["events_sha256"] = "stale"
            with self.assertRaises(ValueError):
                prepare_action_data(config)
        self.assertEqual(mapped_types("unattended_bag_contents_unknown"), [])

    def test_checkpoint_restoration_mapping_and_no_pretrained_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pt"
            checkpoint = dict(schema=SCHEMA, architecture="r3d_18_multilabel", class_names=[A, B],
                              class_to_idx={A: 0, B: 1}, config=dict(weights="KINETICS400_V1", frames=16),
                              data_manifest=dict(catalog_sha256=sha256(CATALOG_PATH), events_sha256="synthetic",
                                review=dict(events_sha256="synthetic", listed_actions_verified=True,
                                            catalog_mapping_verified=True, complete_action_lists=True)),
                              model_state_dict=Tiny().state_dict())
            torch.save(checkpoint, path)
            with patch("overfitted_underqualified.ml.train.r3d_18", return_value=Tiny()) as factory:
                model, _, _ = load_action_checkpoint(path)
                factory.assert_called_once_with(weights=None)
                self.assertFalse(model.training)
            checkpoint["class_to_idx"] = {B: 0, A: 1}
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_action_checkpoint(path)

    def test_concrete_recognizer_pixels_thresholds_and_bn_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            video = path / "unknown.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
            for _ in range(30): writer.write(np.zeros((32, 32, 3), np.uint8))
            writer.release()
            fixture = Tiny()
            with torch.no_grad():
                fixture.fc.weight.zero_(); fixture.fc.bias.copy_(torch.tensor([3., -3.]))
            checkpoint = dict(schema=SCHEMA, architecture="r3d_18_multilabel", class_names=[A, B],
                              class_to_idx={A: 0, B: 1}, validated_types=[A],
                              config=dict(weights="KINETICS400_V1", frames=16, window_sec=2,
                                          threshold=.7, background_ceiling=.2),
                              data_manifest=dict(catalog_sha256=sha256(CATALOG_PATH), events_sha256="synthetic",
                                review=dict(events_sha256="synthetic", listed_actions_verified=True,
                                            catalog_mapping_verified=True, complete_action_lists=True)),
                              model_state_dict=fixture.state_dict())
            torch.save(checkpoint, path / "fixture.pt")
            with patch("overfitted_underqualified.ml.train.r3d_18", side_effect=lambda **_: Tiny()):
                recognizer = R3DActionRecognizer(path / "fixture.pt", enabled_types=[A])
                with self.assertRaises(ValueError):
                    R3DActionRecognizer(path / "fixture.pt", enabled_types=[B])
            before = {k: v.clone() for k, v in recognizer.model.state_dict().items()}
            decision = recognizer.predict_window(inspect_video(video), window(0))
            self.assertEqual([a.event_type for a in decision["observations"]], [A])
            self.assertTrue(all(torch.equal(v, recognizer.model.state_dict()[k]) for k, v in before.items()))
            with torch.no_grad(): recognizer.model.fc.bias.fill_(0)
            self.assertEqual(recognizer.predict_window(inspect_video(video), window(0))["status"], "unknown")
            with torch.no_grad(): recognizer.model.fc.bias.fill_(-3)
            self.assertEqual(recognizer.predict_window(inspect_video(video), window(0))["status"], "background")

    def test_pipeline_uses_window_pixels_without_annotation_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unseen.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
            self.assertTrue(writer.isOpened())
            for _ in range(30): writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
            scanner = SimpleNamespace(config=ScanConfig(window_sec=2), provenance={})
            scanner.scan = lambda *args, **kwargs: dict(windows=[window(0), window(1)], sha256=sha256(path))
            recognizer = SimpleNamespace(window_sec=2, device="cpu", provenance={})
            recognizer.predict_window = lambda v, w: dict(status="unknown", observations=[], probabilities={A: 0.4})
            pipeline = TypedVideoPipeline(scanner, recognizer)
            with patch("overfitted_underqualified.ml.action_data.prepare_action_data", side_effect=AssertionError("No labels during inference")):
                result = pipeline.scan(path)
            self.assertEqual(result["verdict"], "insufficient_data")
            self.assertEqual(result["typed_events"], [])
            recognizer.predict_window = lambda v, w: dict(status="typed", observations=[ActionObservation(A, .9, "test", "test")], probabilities={A: .9})
            result = pipeline.scan(path)
            self.assertEqual(result["verdict"], "violation")
            self.assertEqual(result["typed_events"][0]["action"]["source_links"][0]["event_type"], A)
            scanner.scan = lambda *args, **kwargs: dict(windows=[{**window(0), "probability": .1}], sha256=sha256(path))
            self.assertEqual(pipeline.scan(path)["verdict"], "no_violation")

    def test_temporal_misses_include_binary_gate_and_untrained_types(self):
        sample = ActionSample(SimpleNamespace(clip_id="heldout"), 2., (A, B), False, ("source",))
        scans = {"heldout": dict(typed_events=[dict(event_type=A, timestamp_sec=2.)],
                                 scan_config=dict(threshold=.5), windows=[{**window(0), "probability": .1}])}
        result = temporal_summary([sample], scans, [A, B])
        self.assertEqual(result["per_class"][1]["missed"], 1)
        self.assertEqual(result["per_class"][1]["binary_gate_coverage"], 0)
        self.assertIsNone(result["per_class"][0]["precision"])

    def test_synthetic_training_checkpoint_and_evaluation_export(self):
        # Small substituted backbone; exercise real loaders, loops and report code.
        # No artifact or metric from this fixture is a banking-model experiment.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, assignments = [], []
            for i, (subset, positive) in enumerate((("train", True), ("train", False),
                                                   ("validation", True), ("validation", False)), 1):
                video = root / f"X-{i}.avi"
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
                for j in range(24): writer.write(np.full((32, 32, 3), i * 30 + j, np.uint8))
                writer.release()
                label = "violation" if positive else "no_violation"
                rows.append(dict(clip_id=f"X-{i}", source_label=label, timecode="00:01" if positive else "",
                                 observed_event_type="safe_key_left_in_lock" if positive else "not_reported",
                                 action_review=json.dumps(dict(schema="action_window_review_v1",
                                     catalog_sha256=sha256(CATALOG_PATH), video_sha256=sha256(video),
                                     center_sec=1. if positive else 1.15, window_sec=4., reviewer="SYNTHETIC TEST ONLY",
                                     reviewed_at="2026-10-18T00:00:00Z", evidence_note="Synthetic pixels, not banking labels",
                                     present_types=[A] if positive else [], absent_types=[] if positive else [A],
                                     whole_video_absent_types=[] if positive else [A]))))
                assignments.append(dict(clip_id=f"X-{i}", source_label=label, split=subset, sha256=sha256(video),
                                        source_clip_ids=json.dumps([f"X-{i}"]), event_times_sec="[1.0]" if positive else "[]"))
            for name, data in (("events.csv", rows), ("split.csv", assignments)):
                with (root / name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, data[0].keys()); writer.writeheader(); writer.writerows(data)
            config = read_config(Path(__file__).resolve().parents[1] / "configs/actions.yaml")
            config.update(events_path=str(root / "events.csv"), split_path=str(root / "split.csv"),
                          video_dir=str(root), output_dir=str(root / "run"), epochs=1,
                          weights_cache=str(root / "cache"), device="cpu", cpu_threads=2)
            config["review"] = dict(events_sha256=sha256(root / "events.csv"), listed_actions_verified=True,
                                    label_mode="per_type_v1", catalog_mapping_verified=False,
                                    complete_action_lists=False, provenance="SYNTHETIC TEST ONLY")
            with patch("overfitted_underqualified.ml.train.r3d_18", side_effect=lambda **_: Tiny()):
                checkpoint = run_action_training(config)
                self.assertTrue(checkpoint.is_file())
                binary = Tiny()
                save_checkpoint(root / "binary.pt", binary, torch.optim.AdamW(binary.parameters()), 1,
                                {**config, "freeze_backbone": True}, {}, {})
                output, report = evaluate_actions(config, root / "binary.pt", checkpoint, [A], output_parent=root / "reports")
                self.assertTrue(output.with_suffix(".zip").is_file())
                self.assertEqual(report["window_metrics"]["per_class"][0]["support"], 1)
                self.assertNotIn("image_base64", (output / "scans.json").read_text(encoding="utf-8"))
                with self.assertRaises(FileExistsError): run_action_training(config)

    def test_jury_schema_excludes_ui_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
            for _ in range(10): writer.write(np.zeros((32, 32, 3), np.uint8))
            writer.release()
            action = linked()
            finding = TypedFinding(type=A, timecode_sec=.5, description="test", evidence_frames=evidence_frames(path, .5),
                                   binary_score=.8, type_confidence=.9, source_links=action["source_links"],
                                   applicability=action["applicability"], video_cannot_establish=action["video_cannot_establish"])
            result = TypedAnalysisResult(verdict="violation", violations=[finding])
            wire = _jury_job_response(dict(job_id="test", status="done", result=result.model_dump()))
            self.assertEqual(set(wire["violations"][0]), {"type", "timecode_sec", "description", "evidence_frames"})
            self.assertNotIn("unresolved_candidates", wire)

    def test_missing_action_checkpoint_keeps_readiness_false(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"FORTEBANK_DATA_DIR": directory, "FORTEBANK_API_KEY": "test"}):
            with patch("overfitted_underqualified.ml.action_inference.prepare_typed_pipeline", side_effect=FileNotFoundError("test")):
                analyzer = create_analyzer(None)
            self.assertEqual(analyzer.name, "unconfigured")
            with TestClient(create_app(Settings.from_env(), analyzer)) as client:
                self.assertEqual(client.get("/readyz").status_code, 503)

    def test_local_provider_worker_and_ui_contract_with_synthetic_pipeline(self):
        # Actual adapter/thread/worker/SQLite/HTTP paths; decisions are a fixture,
        # explicitly not an integration pass with a trained action checkpoint.
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"FORTEBANK_DATA_DIR": directory, "FORTEBANK_API_KEY": "test"}):
            video = Path(directory) / "test.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
            for _ in range(10): writer.write(np.zeros((32, 32, 3), np.uint8))
            writer.release()
            pipeline = SimpleNamespace(recognizer=SimpleNamespace(enabled_types={A}))
            action = linked()
            results = [dict(verdict="violation", typed_events=[dict(event_type=A, timestamp_sec=.5,
                       type_confidence=.9, binary_score=.8, action=action)], unresolved_candidates=[]),
                       dict(verdict="insufficient_data", typed_events=[], unresolved_candidates=[
                           dict(timestamp_sec=.5, binary_score=.8, reason="unknown")]),
                       dict(verdict="no_violation", typed_events=[], unresolved_candidates=[]),
                       RuntimeError("private failure details")]
            def scan(*args, **kwargs):
                result = results.pop(0)
                if isinstance(result, Exception): raise result
                return result
            pipeline.scan = scan
            analyzer = LocalActionAnalyzer(pipeline)
            app = create_app(Settings.from_env(), analyzer)
            headers = {"X-API-Key": "test"}
            with TestClient(app) as client:
                self.assertEqual(client.get("/readyz").status_code, 200)
                for index in range(4):
                    job_id = f"synthetic-{index}"
                    app.state.store.admit_job(job_id=job_id, file_path=video, now=current_time(), max_active_jobs=2,
                                              job_deadline_seconds=30, analysis_provider=analyzer.name)
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        wire = client.get(f"/api/v1/ui/jobs/{job_id}", headers=headers).json()
                        if wire["status"] != "processing": break
                        time.sleep(.03)
                    self.assertEqual(wire["status"], "failed" if index == 3 else "done")
                    if index == 0:
                        self.assertEqual(wire["violations"][0]["source_links"][0]["event_type"], A)
                    if index == 1:
                        self.assertEqual(wire["verdict"], "insufficient_data")
                        self.assertEqual(wire["unresolved_candidates"][0]["reason"], "unknown")
                    if index == 2: self.assertEqual(wire["verdict"], "no_violation")
                    if index == 3: self.assertNotIn("private", json.dumps(wire))


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_retains_slot_until_worker_thread_finishes(self):
        entered, stop_seen, release = threading.Event(), threading.Event(), threading.Event()
        analyzer = LocalActionAnalyzer(None)
        def slow(_path, stop):
            entered.set()
            stop.wait(3)
            stop_seen.set()
            release.wait(3)
            raise InterruptedError("test")
        analyzer._analyze_sync = slow
        task = asyncio.create_task(analyzer.analyze(Path("test")))
        try:
            await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.to_thread(stop_seen.wait, 2)
            self.assertTrue(analyzer._lock.locked())
            self.assertFalse(task.done())
            task.cancel()  # repeated shutdown cancellation must also drain
            await asyncio.sleep(.01)
            self.assertTrue(analyzer._lock.locked())
        finally:
            release.set()
            with self.assertRaises(asyncio.CancelledError): await task
        self.assertFalse(analyzer._lock.locked())


if __name__ == "__main__":
    unittest.main()
