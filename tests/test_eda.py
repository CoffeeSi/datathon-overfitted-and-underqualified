import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

# Load imports/constants and the definition cells, without running the pipeline.
notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
definition_cells = []
for cell in notebook["cells"]:
    tags = cell.get("metadata", {}).get("tags", [])
    source = "".join(cell["source"])
    if "eda-imports" in tags:
        tree = ast.parse(source)
        tree.body = [node for node in tree.body
                     if isinstance(node, (ast.Import, ast.ImportFrom))
                     or (isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "LABELS" for t in node.targets))]
        exec(compile(tree, str(notebook_path), "exec"), globals())
    elif "eda-definitions" in tags:
        definition_cells.append(cell)
        exec(compile(source, str(notebook_path), "exec"), globals())
if not definition_cells:
    raise RuntimeError("No EDA definition cells found")



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
            capture = cv2.VideoCapture

            class FailedSeekCapture:
                def __init__(self, filename):
                    self.cap = capture(filename)
                    self.seek_failed = False

                def set(self, *_args):
                    self.seek_failed = True
                    return False

                def read(self):
                    return (False, None) if self.seek_failed else self.cap.read()

                def __getattr__(self, name):
                    return getattr(self.cap, name)

            # Seeking fails, but sequential decoding still recovers the final
            # frame without manufacturing a frame for a timestamp past EOF.
            with patch.object(cv2, "VideoCapture", FailedSeekCapture):
                recovered = read_frames(path, [0, 1.9, 10])
            self.assertIsNotNone(recovered[0])
            self.assertIsNotNone(recovered[1])
            self.assertIsNone(recovered[2])
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
        pairs = pd.DataFrame([dict(clip_a="X-2", clip_b="X-4"), dict(clip_a="X-5", clip_b="X-6")])
        groups = build_groups(events, inventory, pairs)
        conservative = build_groups(events, inventory, pairs, keep_together=True)
        self.assertEqual(conservative["group_id"].nunique(), 1)
        self.assertTrue(propose_split(events, conservative)[0].empty)
        index = groups.set_index("clip_id")["group_id"]
        for a, b in ((1, 3), (2, 4), (5, 6)):
            self.assertEqual(index[f"X-{a}"], index[f"X-{b}"])
        split, _ = propose_split(events, groups)
        self.assertFalse(split.empty)
        self.assertEqual(split.groupby("group_id")["split"].nunique().max(), 1)
        self.assertTrue(split.groupby("split")["source_label"].nunique().eq(2).all())
        again, _ = propose_split(events, groups)
        pd.testing.assert_frame_equal(split, again)
        with_notes = events.assign(observed_event="unverified note",
                                      candidate_violation_type="unverified category")
        with_notes_split, _ = propose_split(with_notes, groups)
        pd.testing.assert_frame_equal(split, with_notes_split)

    def test_impossible_split_has_no_random_fallback(self):
        events = pd.DataFrame([dict(clip_id="X-1", source_label="violation"), dict(clip_id="X-2", source_label="no_violation")])
        groups = pd.DataFrame([dict(clip_id="X-1", group_id="same"), dict(clip_id="X-2", group_id="same")])
        result, _ = propose_split(events, groups)
        self.assertTrue(result.empty)
        result, _ = propose_split(events.iloc[:0], groups.iloc[:0])
        self.assertTrue(result.empty)

    def test_deduplication_preserves_aliases_and_distinct_event_times(self):
        inventory = pd.DataFrame([
            dict(clip_id="X-4", sha256="same"),
            dict(clip_id="X-47", sha256="same"),
        ])
        mapping = duplicate_mapping(inventory)
        analysis = pd.DataFrame([
            dict(clip_id="X-4", source_label="violation", timecode_sec=57, video_file="a.mov"),
            dict(clip_id="X-47", source_label="violation", timecode_sec=68, video_file="b.mov"),
        ]).merge(mapping, on="clip_id", validate="one_to_one")
        videos = training_video_table(analysis)
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos.iloc[0]["clip_id"], "X-4")
        self.assertEqual(json.loads(videos.iloc[0]["source_clip_ids"]), ["X-4", "X-47"])
        self.assertEqual(json.loads(videos.iloc[0]["event_times_sec"]), [57, 68])
        analysis.loc[1, "source_label"] = "no_violation"
        with self.assertRaises(ValueError):
            training_video_table(analysis)


if __name__ == "__main__":
    unittest.main()
