"""Geometry, review boundaries and checkpoint contracts for the banknote primitive."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from overfitted_underqualified.ml.banknote_model import (
    ARCHITECTURE, ARCHITECTURES, DEFINITION, PROTOCOL_V2, PROTOCOL_V3, SCHEMA, BanknoteDetector, BanknoteFrames, BanknoteSample,
    TinyBanknoteUNet, _epoch, _model_digest, collate_frames, extract_objects, initialize_model, load,
    prepare_banknote_data, prepare_frame, segmentation_loss, supplemental_sources, train, validate_boxes,
)
from overfitted_underqualified.ml.dataset import sha256


def config(directory="."):
    return dict(schema=SCHEMA, roi=[0., .3, 1., .6], resize_width=64, epochs=1, batch_size=2,
                cpu_threads=1, min_component_area=6, seed=42, threshold=.7, learning_rate=.001,
                events_sha256="a" * 64, split_sha256="b" * 64, device="cpu",
                output_dir=str(Path(directory) / "run"))


class BanknoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_review_masks_and_box_bounds_are_explicit(self):
        roi = [0., .3, 1., .6]
        point = dict(labels={"banknotes_on_counter": 1}, banknote_boxes=[[.2, .4, .3, .5]],
                     banknote_boxes_complete=True)
        self.assertEqual(validate_boxes(point, roi), point["banknote_boxes"])
        self.assertIsNone(validate_boxes(dict(point, banknote_boxes_complete=False), roi))
        for bad in (dict(point, banknote_boxes=[]), dict(point, labels={}),
                    dict(point, labels={"banknotes_on_counter": 0}),
                    dict(point, banknote_boxes=[[.2, .2, .3, .5]]),
                    dict(point, banknote_boxes=[[.3, .4, .2, .5]]),
                    dict(point, banknote_boxes=[[.2, .4, float("nan"), .5]])):
            with self.assertRaises(ValueError):
                validate_boxes(bad, roi)

    def test_resize_round_trip_and_padding_do_not_create_boxes(self):
        cfg = config()
        image = np.zeros((100, 200, 3), np.uint8)
        box = [.25, .4, .5, .5]
        frame = prepare_frame(image, cfg, [box])
        self.assertEqual(frame["image"].shape, (3, 12, 64))
        values = frame["mask"][0].numpy() * .9
        values[10:, :] = .99  # padding must be ignored by inverse geometry
        objects = extract_objects(values, frame["geometry"], .7, 6)
        self.assertEqual(len(objects), 1)
        np.testing.assert_allclose(objects[0]["box"], box, atol=.035)
        self.assertEqual(float(frame["valid"].sum()), 640)
        with self.assertRaises(ValueError):
            prepare_frame(image.astype(np.float32), cfg)

    def test_aliases_deduplicate_and_source_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path, split = Path(directory) / "events.csv", Path(directory) / "split.csv"
            split.write_text("split", encoding="utf-8")
            record = SimpleNamespace(clip_id="X-4", aliases=("X-4", "X-47"), sha256="c" * 64,
                                     last_time=80., split="train")
            review = dict(schema="visible_primitives_v1", video_sha256=record.sha256, window_sec=2.,
                          method="frame_sequence_review", reviewer="test", reviewed_at="2026-10-18",
                          points=[dict(time_sec=57., labels={"banknotes_on_counter": 1},
                                       evidence_seconds=[57.], note="box", banknote_boxes_complete=True,
                                       banknote_boxes=[[.2, .4, .3, .5]])])
            rows = [dict(clip_id=alias, primitive_review=json.dumps(review)) for alias in record.aliases]
            def write():
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, list(rows[0])); writer.writeheader(); writer.writerows(rows)
                return dict(config(directory), events_path=str(path), split_path=str(split), video_dir=directory,
                            events_sha256=sha256(path), split_sha256=sha256(split))
            with patch("overfitted_underqualified.ml.banknote_model.load_records", return_value=[record]):
                cfg = write()
                samples, manifest = prepare_banknote_data(cfg)
                self.assertEqual(len(samples), 1)
                self.assertEqual(manifest["train_parents"], ["X-4"])
                self.assertEqual(samples[0].source_ids, record.aliases)
                with self.assertRaisesRegex(ValueError, "hash"):
                    prepare_banknote_data(dict(cfg, events_sha256="f" * 64))
                review["points"][0]["banknote_boxes"] = [[.3, .4, .4, .5]]
                rows[1]["primitive_review"] = json.dumps(review)
                with self.assertRaisesRegex(ValueError, "Conflicting"):
                    prepare_banknote_data(write())

    def test_validation_has_no_augmentation_or_parameter_updates(self):
        sample = BanknoteSample(SimpleNamespace(split="validation"), 0., (), ())
        with self.assertRaisesRegex(ValueError, "Validation"):
            BanknoteFrames([sample], config(), augment=True)
        model = TinyBanknoteUNet()
        frame = prepare_frame(np.zeros((100, 200, 3), np.uint8), config(), [[.2, .4, .3, .5]])
        loader = [collate_frames([dict(frame, index=0)])]
        before = _model_digest(model)
        loss = _epoch(model, loader, "cpu", 10.)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(_model_digest(model), before)
        self.assertFalse(model.training)
        _epoch(model, loader, "cpu", 10., torch.optim.AdamW(model.parameters(), lr=.001))
        self.assertNotEqual(_model_digest(model), before)
        self.assertEqual(model(frame["image"][None]).shape, frame["mask"][None].shape)

    def test_checkpoint_restore_and_provenance_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            checkpoint = dict(schema=SCHEMA, architecture=ARCHITECTURE, definition=DEFINITION,
                              model_state_dict=TinyBanknoteUNet().state_dict(), config=cfg, epoch=1,
                              selection="fixed_last_epoch", data_manifest=dict(events_sha256=cfg["events_sha256"],
                              split_sha256=cfg["split_sha256"], train_parents=["X-1"], validation_parents=["X-2"]))
            path = Path(directory) / "last.pt"
            torch.save(checkpoint, path)
            detector = load(path)
            self.assertFalse(detector.model.training)
            result = detector.predict(np.zeros((100, 200, 3), np.uint8))
            self.assertFalse(result["absence_certified"])
            self.assertEqual(result["checkpoint_sha256"], sha256(path))
            checkpoint["data_manifest"]["events_sha256"] = "f" * 64
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "provenance"):
                load(path)
            Path(cfg["output_dir"]).mkdir()
            with self.assertRaises(FileExistsError):
                train(cfg)

    def test_supplemental_variants_preserve_parent_sampling_weight_and_coordinates(self):
        record = SimpleNamespace(clip_id="X-1", split="train", fps=1., frame_count=2)
        sample = BanknoteSample(record, 0., ((.2, .4, .3, .5),), ("X-1",))
        source = SimpleNamespace(parent_id="X-1", path=Path("mirror.mp4"),
                                 restore_orientation=lambda frames: frames.flip(-1))
        image = torch.zeros(1, 3, 100, 200, dtype=torch.uint8)
        image[:, :, :, :50] = 255
        with patch("overfitted_underqualified.ml.banknote_model.decode_frames", side_effect=[image, image.flip(-1)]):
            data = BanknoteFrames([sample], config(), augment=True, variants={"X-1": [source]})
        self.assertEqual(len(data), 1)  # variants do not multiply this parent's training weight
        self.assertTrue(torch.equal(data.frames[0]["image"], data.variants[0][0]["image"]))
        self.assertTrue(torch.equal(data.frames[0]["mask"], data.variants[0][0]["mask"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("{}", encoding="utf-8")
            cfg = dict(config(), supplemental_manifest=str(path), supplemental_manifest_sha256="f" * 64)
            with self.assertRaisesRegex(ValueError, "pinned"):
                supplemental_sources(cfg, [sample])

    def test_v2_rare_object_gradient_is_not_drowned_by_background_or_padding(self):
        logits = torch.zeros(1, 1, 64, 64, requires_grad=True)
        target = torch.zeros_like(logits); target[0, 0, 10, 10] = 1
        valid = torch.ones_like(logits); valid[:, :, 60:] = 0
        loss = segmentation_loss(logits, target, valid, 50., PROTOCOL_V2)
        loss.backward()
        positive_gradient = abs(float(logits.grad[0, 0, 10, 10]))
        negative_gradient = abs(float(logits.grad[0, 0, 0, 0]))
        self.assertGreater(positive_gradient, negative_gradient * 1000)
        self.assertTrue(torch.equal(logits.grad[:, :, 60:], torch.zeros_like(logits.grad[:, :, 60:])))
        self.assertLess(float(logits.grad[0, 0, 10, 10]), 0)
        self.assertGreater(float(logits.grad[0, 0, 0, 0]), 0)

    def test_v2_can_fit_a_tiny_synthetic_object_without_validation_updates(self):
        # Optimization sanity only: this does not claim real banknote quality.
        torch.manual_seed(42)
        model = TinyBanknoteUNet(PROTOCOL_V2)
        images = torch.zeros(2, 3, 24, 32)
        images[0, 2, 10:13, 15:19] = 1
        target = torch.zeros(2, 1, 24, 32); target[0, 0, 10:13, 15:19] = 1
        valid = torch.ones_like(target)
        initial = float(segmentation_loss(model(images), target, valid, 50., PROTOCOL_V2).detach())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        for _ in range(35):
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(images), target, valid, 50., PROTOCOL_V2)
            loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(images).sigmoid()
            final = float(segmentation_loss(model(images), target, valid, 50., PROTOCOL_V2))
        self.assertLess(final, initial * .5)
        self.assertGreater(float(prediction[0, 0, 10:13, 15:19].mean()), .7)
        self.assertLess(float(prediction[1].max()), .7)

    def test_v2_checkpoint_and_embedded_supplemental_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(config(directory), training_protocol=PROTOCOL_V2,
                       supplemental_manifest="unavailable-desktop-manifest.json", supplemental_manifest_sha256="c" * 64)
            manifest = dict(events_sha256=cfg["events_sha256"], split_sha256=cfg["split_sha256"],
                            train_parents=["X-1"], validation_parents=["X-2"],
                            supplemental=dict(manifest_sha256="c" * 64, sources=[dict(parent_id="X-1")]))
            checkpoint = dict(schema=SCHEMA, architecture=ARCHITECTURES[PROTOCOL_V2], definition=DEFINITION,
                              model_state_dict=TinyBanknoteUNet(PROTOCOL_V2).state_dict(), config=cfg, epoch=1,
                              selection="fixed_last_epoch", data_manifest=manifest)
            path = Path(directory) / "last.pt"
            torch.save(checkpoint, path)
            restored = load(path)
            self.assertTrue(any(isinstance(layer, torch.nn.LeakyReLU) for layer in restored.model.modules()))
            for mutation in ("hash", "parent", "presence", "architecture"):
                bad = copy.deepcopy(checkpoint)
                if mutation == "hash": bad["data_manifest"]["supplemental"]["manifest_sha256"] = "d" * 64
                if mutation == "parent": bad["data_manifest"]["supplemental"]["sources"][0]["parent_id"] = "X-2"
                if mutation == "presence": del bad["data_manifest"]["supplemental"]
                if mutation == "architecture": bad["architecture"] = ARCHITECTURE
                torch.save(bad, path)
                with self.assertRaises(ValueError, msg=mutation):
                    load(path)

    def test_v3_mines_confident_false_positives_instead_of_easy_background(self):
        # An explicitly negative frame has no Dice term: we can inspect exactly
        # which negative BCE pixels participate in the hard-negative gradient.
        values = torch.full((1, 1, 64, 64), -8.)
        values[0, 0, 20:22, 20:22] = 3.
        logits = values.requires_grad_()
        target, valid = torch.zeros_like(logits), torch.ones_like(logits)
        valid[:, :, 60:] = 0
        loss = segmentation_loss(logits, target, valid, 50., PROTOCOL_V3)
        loss.backward()
        wrong_gradient = logits.grad[0, 0, 20:22, 20:22]
        self.assertTrue((wrong_gradient > .01).all())
        self.assertEqual(int((logits.grad > 0).sum()), 39)  # ceil(.01 * 3840)
        self.assertEqual(float(logits.grad[:, :, 60:].abs().sum()), 0.)
        self.assertLess(float(logits.grad.abs().sum() - wrong_gradient.sum()), float(wrong_gradient.sum()) * .01)

    def test_v3_architecture_load_is_separate_and_keeps_leaky_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(config(directory), training_protocol=PROTOCOL_V3,
                       hard_negative_ratio=3, hard_negative_min_fraction=.01)
            checkpoint = dict(schema=SCHEMA, architecture=ARCHITECTURES[PROTOCOL_V3], definition=DEFINITION,
                              model_state_dict=TinyBanknoteUNet(PROTOCOL_V3).state_dict(), config=cfg, epoch=1,
                              selection="fixed_last_epoch", data_manifest=dict(events_sha256=cfg["events_sha256"],
                              split_sha256=cfg["split_sha256"], train_parents=["X-1"], validation_parents=["X-2"]))
            path = Path(directory) / "last.pt"; torch.save(checkpoint, path)
            restored = load(path)
            self.assertTrue(any(isinstance(layer, torch.nn.LeakyReLU) for layer in restored.model.modules()))
            checkpoint["config"]["hard_negative_min_fraction"] = .1
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "fixes"):
                load(path)

    def test_v2_to_v3_initialization_pins_data_and_restores_without_parent_file(self):
        with tempfile.TemporaryDirectory() as directory:
            parent_config = dict(config(directory), training_protocol=PROTOCOL_V2)
            manifest = dict(events_sha256=parent_config["events_sha256"], split_sha256=parent_config["split_sha256"],
                            train_parents=["X-1"], validation_parents=["X-2"], samples=[dict(clip_id="X-1", time_sec=2.)])
            parent_model = TinyBanknoteUNet(PROTOCOL_V2)
            checkpoint = dict(schema=SCHEMA, architecture=ARCHITECTURES[PROTOCOL_V2], definition=DEFINITION,
                              model_state_dict=parent_model.state_dict(), config=parent_config, epoch=1,
                              selection="fixed_last_epoch", data_manifest=manifest)
            parent_path = Path(directory) / "parent.pt"; torch.save(checkpoint, parent_path)
            child_config = dict(parent_config, training_protocol=PROTOCOL_V3, initialize_checkpoint=str(parent_path),
                                initialize_checkpoint_sha256=sha256(parent_path))
            child_model = TinyBanknoteUNet(PROTOCOL_V3)
            initialization = initialize_model(child_model, child_config, manifest)
            self.assertEqual(_model_digest(child_model), _model_digest(parent_model))
            bad_manifest = copy.deepcopy(manifest); bad_manifest["samples"][0]["time_sec"] = 3.
            with self.assertRaisesRegex(ValueError, "source manifest"):
                initialize_model(child_model, child_config, bad_manifest)
            with self.assertRaisesRegex(ValueError, "hash"):
                initialize_model(child_model, dict(child_config, initialize_checkpoint_sha256="f" * 64), manifest)
            child = dict(checkpoint, architecture=ARCHITECTURES[PROTOCOL_V3], config=child_config,
                         model_state_dict=child_model.state_dict(), initialization=initialization)
            child_path = Path(directory) / "child.pt"; torch.save(child, child_path)
            parent_path.unlink()  # deployment is self-contained, not dependent on the old run
            self.assertEqual(_model_digest(load(child_path).model), _model_digest(child_model))
            child["initialization"]["checkpoint_sha256"] = "d" * 64
            torch.save(child, child_path)
            with self.assertRaisesRegex(ValueError, "initialization provenance"):
                load(child_path)


if __name__ == "__main__":
    unittest.main()
