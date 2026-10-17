import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from overfitted_underqualified.eda import (
    build_groups, clip_id_from_name, inspect_video, parse_timecode,
    propose_split, validate_dataset, window_timestamps,
)


class EdaChecks(unittest.TestCase):
    def test_timecodes_reject_invalid_components(self):
        for value in ("00:60", "-1", "abc", "1:99:00", "inf"):
            self.assertTrue(np.isnan(parse_timecode(value)))
        self.assertEqual(parse_timecode("01:08"), 68)
        self.assertEqual(parse_timecode("1:02:03.5"), 3723.5)
        self.assertTrue(np.isnan(parse_timecode(pd.NA)))

    def test_windows_include_event_and_stay_inside_video(self):
        for center in (0, 0.3, 4.3, 9.95):
            stamps = window_timestamps(center, duration=10, fps=20)
            self.assertIn(center, stamps)
            self.assertGreaterEqual(min(stamps), 0)
            self.assertLessEqual(max(stamps), 9.95)
        with self.assertRaises(ValueError):
            window_timestamps(10, 10, 20)

    def test_decode_and_broken_container(self):
        # Real decoding rather than mocked container properties.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "data" / "processed") as folder:
            root = Path(folder)
            path = root / "X-1.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(20):
                writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
            writer.release()
            record, thumbnails = inspect_video(path, root)
            self.assertTrue(record["readable"])
            self.assertEqual(record["sampled_frames_ok"], 5)
            self.assertAlmostEqual(record["duration_sec"], 2)
            self.assertEqual(len(thumbnails), 5)
            broken = root / "X-2.mov"
            broken.write_bytes(b"not a video")
            record, _ = inspect_video(broken, root)
            self.assertFalse(record["readable"])

    def test_ids_normalize_cyrillic(self):
        self.assertEqual(clip_id_from_name("Х-01.mov"), "X-1")

    def test_validation_detects_conflicts_and_invalid_times(self):
        events = pd.DataFrame([
            dict(clip_id="X-1", source_label="violation", timecode="00:12"),
            dict(clip_id="X-2", source_label="no_violation", timecode=""),
            dict(clip_id="X-3", source_label="violation", timecode="bad"),
        ])
        inventory = pd.DataFrame([
            dict(clip_id="X-1", readable=True, error="", duration_sec=10, sha256="same", video_rel="a"),
            dict(clip_id="X-2", readable=True, error="", duration_sec=10, sha256="same", video_rel="b"),
        ])
        codes = set(validate_dataset(events, inventory)["code"])
        self.assertTrue({"time_out_of_bounds", "duplicate_label_conflict", "missing_video", "invalid_timecode"} <= codes)
        duplicate = pd.concat([inventory, inventory.iloc[:1]], ignore_index=True)
        self.assertIn("duplicate_video_id", set(validate_dataset(events, duplicate)["code"]))

    def test_group_split_keeps_duplicates_and_scenes_together(self):
        events = pd.DataFrame([dict(clip_id=f"X-{i}", source_label="violation" if i % 2 else "no_violation") for i in range(1, 17)])
        inventory = pd.DataFrame([dict(clip_id=f"X-{i}", sha256="same" if i in (1, 3) else str(i)) for i in range(1, 17)])
        pairs = pd.DataFrame([dict(clip_a="X-2", clip_b="X-4")])
        scenes = pd.DataFrame([dict(clip_id="X-5", camera_id="A"), dict(clip_id="X-6", camera_id="A")])
        groups = build_groups(events, inventory, pairs, scenes)
        index = groups.set_index("clip_id")["group_id"]
        for a, b in ((1, 3), (2, 4), (5, 6)):
            self.assertEqual(index[f"X-{a}"], index[f"X-{b}"])
        split, _ = propose_split(events, groups)
        self.assertFalse(split.empty)
        self.assertEqual(split.groupby("group_id")["split"].nunique().max(), 1)
        self.assertTrue(split.groupby("split")["source_label"].nunique().eq(2).all())
        again, _ = propose_split(events, groups)
        pd.testing.assert_frame_equal(split, again)

    def test_impossible_split_has_no_random_fallback(self):
        events = pd.DataFrame([dict(clip_id="X-1", source_label="violation"), dict(clip_id="X-2", source_label="no_violation")])
        groups = pd.DataFrame([dict(clip_id="X-1", group_id="same"), dict(clip_id="X-2", group_id="same")])
        result, _ = propose_split(events, groups)
        self.assertTrue(result.empty)
        result, _ = propose_split(events.iloc[:0], groups.iloc[:0])
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
