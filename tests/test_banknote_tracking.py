"""Synthetic image evidence checks; no model metrics or validation videos used."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from overfitted_underqualified.ml.banknote_tracking import BanknoteTracker


def scene(dx=0, *, present=True, background=40):
    rgb = np.full((100, 160, 3), background, np.uint8)
    if present:
        rng = np.random.default_rng(17)
        rgb[40:48, 60 + dx:72 + dx] = rng.integers(60, 220, size=(8, 12, 3), dtype=np.uint8)
    return rgb


def detection(dx=0, score=.95):
    return dict(box=[(60 + dx) / 160, .40, (72 + dx) / 160, .48], score=score)


class BanknoteTrackingTests(unittest.TestCase):
    def test_static_and_small_translation_require_fresh_image_matches(self):
        tracker = BanknoteTracker()
        first = tracker.update(scene(), 0., [detection()])
        self.assertEqual(first["objects"][0]["evidence_type"], "model_detection")
        for stamp, offset in ((1., 0), (2., 1), (3., 2)):
            result = tracker.update(scene(offset), stamp, [])
            self.assertEqual(len(result["objects"]), 1)
            item = result["objects"][0]
            self.assertEqual(item["evidence_type"], "tracked")
            self.assertGreater(item["tracking_score"], .99)
            self.assertEqual(item["originating_detection_sec"], 0.)
            self.assertAlmostEqual(item["box"][0], (60 + offset) / 160)
            self.assertFalse(result["absence_certified"])

    def test_removed_or_occluded_object_does_not_persist(self):
        for mode in ("removed", "occluded"):
            tracker = BanknoteTracker(); tracker.update(scene(), 0., [detection()])
            rgb = scene(present=False)
            if mode == "occluded":
                rgb[40:48, 60:72] = np.random.default_rng(88).integers(20, 250, size=(8, 12, 3), dtype=np.uint8)
            result = tracker.update(rgb, 1., [])
            self.assertEqual(result["objects"], [], mode)
            self.assertEqual(len(result["tracking_evidence"]), 1)
            self.assertEqual(tracker.update(scene(), 2., [])["objects"], [])

    def test_flat_patch_never_initializes_or_gets_inferred_from_score_alone(self):
        tracker = BanknoteTracker()
        result = tracker.update(scene(present=False), 0., [detection()])
        self.assertFalse(result["objects"][0]["trackable"])
        self.assertEqual(tracker.update(scene(present=False), 1., [])["objects"], [])
        tracker = BanknoteTracker()
        tracker.update(scene(), 0., [detection(score=.6)])
        self.assertEqual(tracker.update(scene(), 1., [])["objects"], [])

    def test_gap_cut_geometry_and_expired_origin_reset_history(self):
        for mode in ("gap", "cut", "geometry", "expiry"):
            tracker = BanknoteTracker(max_age_sec=1.)
            tracker.update(scene(), 0., [detection()])
            rgb, stamp = scene(), 1.
            if mode == "gap": stamp = 3.
            if mode == "cut": rgb = scene(background=230)
            if mode == "geometry": rgb = np.zeros((110, 160, 3), np.uint8)
            if mode == "expiry": stamp = 2.
            result = tracker.update(rgb, stamp, [])
            self.assertEqual(result["objects"], [], mode)

    def test_matching_texture_with_changed_rgb_is_rejected(self):
        tracker = BanknoteTracker()
        tracker.update(scene(), 0., [detection()])
        rgb = scene(); rgb[40:48, 60:72] += 25
        result = tracker.update(rgb, 1., [])
        self.assertEqual(result["objects"], [])
        self.assertEqual(result["tracking_evidence"][0]["reason"], "rgb_mismatch")

    def test_current_detection_replaces_track_and_bad_time_is_rejected(self):
        tracker = BanknoteTracker(max_age_sec=1.)
        tracker.update(scene(), 0., [detection()])
        result = tracker.update(scene(), 1., [detection()])
        self.assertEqual(len(result["objects"]), 1)
        self.assertEqual(result["objects"][0]["originating_detection_sec"], 0.)
        self.assertEqual(result["objects"][0]["last_detection_sec"], 1.)
        self.assertEqual(result["schema"], tracker.schema)
        with self.assertRaises(ValueError):
            tracker.update(scene(), 1., [])
        continued = tracker.update(scene(), 2., [])
        self.assertEqual(len(continued["objects"]), 1)
        self.assertEqual(continued["objects"][0]["originating_detection_sec"], 0.)
        self.assertEqual(continued["objects"][0]["last_detection_sec"], 1.)


if __name__ == "__main__":
    unittest.main()
