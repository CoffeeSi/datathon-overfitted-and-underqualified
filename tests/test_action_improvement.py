"""Comparisons use the same reviewed denominator, not extra unknown true negatives."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from overfitted_underqualified.ml.action_improvement import checked_inputs, compare_cash_sequences, verify_unchanged
from overfitted_underqualified.ml.banknote_evaluation import EVENT_TYPE, GRID_SCHEMA, TARGET
from overfitted_underqualified.ml.dataset import sha256


class StudyIntegrityTests(unittest.TestCase):
    def test_pins_sources_and_detects_checkpoint_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {}
            for key in ("events_path", "split_path", "baseline_checkpoint", "binary_checkpoint"):
                path = root / key
                path.write_bytes(key.encode())
                config[key] = key
                config[key.removesuffix("_path") + "_sha256"] = sha256(path)
            with patch("overfitted_underqualified.ml.action_improvement.ROOT", root):
                values = checked_inputs(config)
            verify_unchanged(values)
            (root / "baseline_checkpoint").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "baseline_checkpoint"):
                verify_unchanged(values)

    def test_same_reviewed_videos_unknown_exclusion_and_timing(self):
        records = [SimpleNamespace(clip_id=f"X-{i}", sha256=str(i) * 64, last_time=2.,
                                   aliases=(f"X-{i}",), split="validation") for i in (1, 2, 3)]
        source = {r.clip_id: {} for r in records}
        for record, times in zip(records[:2], ([1.], [])):
            review = dict(schema=GRID_SCHEMA, visual_target=TARGET, mapped_candidate_type=EVENT_TYPE,
                          video_sha256=record.sha256, continuous_legal_review=False,
                          complete_for_sampled_grid=True, grid_step_sec=1.,
                          method="full_duration_sampled_frame_sequence_review", reviewer="test",
                          reviewed_at="2026-01-01", note="synthetic test", evidence_manifest="fixture",
                          sampled_seconds=[0., 1., 2.], reference_times_sec=times)
            source[record.clip_id] = {"primitive_review": json.dumps({"sequence_grid_review": review})}
        event = lambda time: dict(event_type=EVENT_TYPE, timestamp_sec=time)
        baseline = {r.clip_id: dict(sha256=r.sha256, typed_events=[]) for r in records}
        baseline["X-2"]["typed_events"] = [event(1.)]
        baseline["X-3"]["typed_events"] = [event(1.)]  # Unknown is not a confirmed false alarm.
        candidate = {r.clip_id: dict(clip_id=r.clip_id, video_sha256=r.sha256, events=[]) for r in records}
        candidate["X-1"]["events"] = [event(1.25)]
        result = compare_cash_sequences(records, source, baseline, candidate, .5)
        self.assertEqual(result["baseline"]["video"]["f1"], 0.)
        self.assertEqual(result["candidate"]["video"]["f1"], 1.)
        self.assertEqual(result["candidate"]["temporal"]["timestamp_mae_sec"], .25)
        self.assertEqual(result["candidate"]["excluded_unknown_videos"], ["X-3"])
        self.assertEqual(len(result["candidate"]["videos"]), 2)
        candidate["X-1"]["video_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hash"):
            compare_cash_sequences(records, source, baseline, candidate, .5)


if __name__ == "__main__":
    unittest.main()
