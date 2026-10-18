"""Typed API adapter preserves learned timing, abstention and cancellation."""
import base64
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from overfitted_underqualified.ml.sequence_service import CashSequencePipeline, EVENT_TYPE, prepare_cash_sequence_pipeline
from overfitted_underqualified.ml.dataset import sha256
from overfitted_underqualified.backend.local_analyzer import LocalActionAnalyzer


def event(stamp=29.):
    return dict(event_type=EVENT_TYPE, timestamp_sec=stamp, confidence=.71,
                receipt_sec=7., departure_sec=27.)


class SequenceServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "random-upload.mov"
        self.path.write_bytes(b"synthetic fixture, decoding mocked")
        self.digest = sha256(self.path)
        self.raw = dict(video_sha256=self.digest, events=[], unresolved=[])
        scanner = SimpleNamespace(banknote_detector=SimpleNamespace(checkpoint_sha256="b" * 64),
                                  checkpoint_sha256="a" * 64, scan=Mock(side_effect=lambda *a, **k: self.raw))
        self.pipeline = CashSequencePipeline(scanner, None)
        self.pipeline.binary_score = Mock(return_value=.01)
        self.video = SimpleNamespace(path=self.path, last_time=60.)
        patcher = patch("overfitted_underqualified.ml.sequence_service.inspect_video", return_value=self.video)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_adapter_returns_actual_multiple_times_and_low_binary_score_does_not_gate(self):
        self.raw["events"] = [event(), event(45.)]
        image = BytesIO()
        Image.new("RGB", (8, 8)).save(image, format="JPEG")
        frame = dict(timestamp_sec=29., image_base64=base64.b64encode(image.getvalue()).decode())
        analyzer = LocalActionAnalyzer(self.pipeline)
        self.assertEqual(analyzer.name, "local-cash-sequence")
        with patch("overfitted_underqualified.ml.inference.evidence_frames", return_value=[frame]):
            result = analyzer._analyze_sync(self.path, threading.Event())
        self.assertEqual(result.verdict, "violation")
        self.assertEqual([v.timecode_sec for v in result.violations], [29., 45.])
        self.assertTrue(all(v.binary_score == .01 for v in result.violations))
        self.assertEqual(result.recognition_scope, [EVENT_TYPE])
        self.assertIsNone(result.violations[0].legal_conclusion)

    def test_empty_chain_abstains_without_claiming_clean_video(self):
        result = self.pipeline.scan(self.path)
        self.assertEqual(result["verdict"], "insufficient_data")
        self.assertFalse(result["typed_events"])
        self.pipeline.binary_score.assert_not_called()

    def test_incomplete_chain_is_preserved_for_manual_review(self):
        self.raw["unresolved"] = [dict(timestamp_sec=10., reason="cash_visible_without_observed_receipt")]
        result = self.pipeline.scan(self.path)
        self.assertEqual(result["unresolved_candidates"], [dict(timestamp_sec=10., binary_score=.01, reason="unknown")])

    def test_rejects_unsupported_type_and_changed_upload(self):
        self.raw["events"] = [dict(event(), event_type="safe_key_left_in_lock")]
        with self.assertRaisesRegex(ValueError, "Unsupported type"):
            self.pipeline.scan(self.path)
        self.raw["events"] = []
        self.path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.pipeline.scan(self.path)

    def test_cancellation_is_cooperative(self):
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(InterruptedError):
            self.pipeline.scan(self.path, cancel_event=cancelled)

    def test_checkpoint_pin_checked_before_model_loading(self):
        config = dict(enabled_types=[EVENT_TYPE], banknote_tracking=True,
                      primitive_checkpoint=str(self.path), primitive_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "primitive checkpoint digest"):
            prepare_cash_sequence_pipeline(config, Path(self.directory.name))


if __name__ == "__main__":
    unittest.main()
