"""The hybrid scanner has no annotation input and resets its episode per call."""
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from overfitted_underqualified.ml.primitive_inference import PrimitiveScanner
from overfitted_underqualified.ml.banknote_tracking import BanknoteTracker


class BanknoteScannerTests(unittest.TestCase):
    def make_scanner(self, cash=.9):
        class Model:
            def eval(self):
                return self
            def __call__(self, batch):
                # Synthetic input carries timestamp in a pixel; no external labels.
                return torch.logit(torch.tensor([[.9 if t==0 else .1,.01,.9 if t==0 else .1]
                    for t in batch[:,0,0,0,0].tolist()]))
        scanner=PrimitiveScanner.__new__(PrimitiveScanner)
        scanner.config={'seed':42,'cpu_threads':2,'scan_step_sec':1.,'window_sec':2.,
            'sequence':{'positive_threshold':.7,'negative_threshold':.3,'dwell_sec':2.,'max_gap_sec':2.,'receipt_ttl_sec':120.}}
        scanner.model=Model();scanner.device=torch.device('cpu');scanner.checkpoint_sha256='primitive_hash'
        scanner.transform=lambda x:x.float().permute(1,0,2,3)
        scanner.banknote_detector=SimpleNamespace(checkpoint_sha256='banknote_hash',predict=lambda x:{'score':cash,'boxes':[], 'absence_certified':False})
        return scanner

    def test_detector_replaces_cash_only_and_episode_resets(self):
        scanner=self.make_scanner()
        record=SimpleNamespace(path=Path('unseen.mp4'),clip_id='unseen',fps=1.,frame_count=6,last_time=5.)
        def frames(record,times):
            return torch.full((16,3,8,8),int(times[-1]),dtype=torch.uint8)
        with patch('overfitted_underqualified.ml.primitive_inference.inspect_video',return_value=record),patch('overfitted_underqualified.ml.primitive_inference.sha256',return_value='video_hash'),patch('overfitted_underqualified.ml.primitive_inference.decode_frames',side_effect=frames):
            for _ in range(2):
                result=scanner.scan(record.path)
                self.assertEqual(len(result['events']),1)
                self.assertEqual(result['events'][0]['timestamp_sec'],3.)
                self.assertEqual(result['events'][0]['event_type'],'unsecured_accepted_cash')
                self.assertTrue(result['events'][0]['requires_review'])
                self.assertTrue(result['events'][0]['rule_candidates'])
                self.assertEqual(result['banknote_checkpoint_sha256'],'banknote_hash')
                self.assertAlmostEqual(result['observations'][0]['r3d_probabilities']['banknotes_on_counter'],.01,places=5)
                self.assertEqual(result['observations'][0]['probabilities']['banknotes_on_counter'],.9)

    def test_negative_detector_and_cancellation(self):
        scanner=self.make_scanner(0.)
        record=SimpleNamespace(path=Path('unseen.mp4'),clip_id='unseen',fps=1.,frame_count=6,last_time=5.)
        with patch('overfitted_underqualified.ml.primitive_inference.inspect_video',return_value=record),patch('overfitted_underqualified.ml.primitive_inference.sha256',return_value='video_hash'),patch('overfitted_underqualified.ml.primitive_inference.decode_frames',return_value=torch.zeros((16,3,8,8),dtype=torch.uint8)):
            self.assertEqual(scanner.scan(record.path)['status'],'insufficient_data')
            cancel=threading.Event();cancel.set()
            with self.assertRaises(InterruptedError):scanner.scan(record.path,cancel_event=cancel)

    def test_image_tracking_bridges_detector_flicker_and_resets_between_videos(self):
        scanner=self.make_scanner()
        scanner.banknote_tracking_config=BanknoteTracker().config
        def predict(rgb):
            objects=[{'box':[.25,.25,.75,.75],'score':.9,'area_px':16}] if rgb[0,0,0]==0 else []
            return {'objects':objects,'boxes':[o['box'] for o in objects],'score':.9 if objects else 0.,'absence_certified':False}
        scanner.banknote_detector.predict=predict
        def frames(record,times):
            frame=torch.zeros((16,3,8,8),dtype=torch.uint8)
            frame[:,:,0,0]=int(times[-1])
            frame[:,:,2:6,2:6]=torch.tensor([[20,80,160,220],[220,20,80,160],[80,160,220,20],[160,220,20,80]],dtype=torch.uint8)
            return frame
        record=SimpleNamespace(path=Path('unseen.mp4'),clip_id='unseen',fps=1.,frame_count=6,last_time=5.)
        with patch('overfitted_underqualified.ml.primitive_inference.inspect_video',return_value=record),patch('overfitted_underqualified.ml.primitive_inference.sha256',return_value='video_hash'),patch('overfitted_underqualified.ml.primitive_inference.decode_frames',side_effect=frames):
            for _ in range(2):
                result=scanner.scan(record.path)
                self.assertEqual(len(result['events']),1)
                self.assertEqual(result['events'][0]['timestamp_sec'],3.)
                self.assertEqual(result['observations'][1]['banknotes']['raw_detection_score'],0.)
                self.assertEqual(result['observations'][1]['banknotes']['objects'][0]['evidence_type'],'tracked')
                self.assertEqual(result['banknote_tracking_config'],BanknoteTracker().config)

    def test_tracking_requires_a_detector(self):
        with self.assertRaisesRegex(ValueError,'requires a banknote checkpoint'):
            PrimitiveScanner(Path('unused.pt'),banknote_tracking=True)


if __name__=='__main__':unittest.main()
