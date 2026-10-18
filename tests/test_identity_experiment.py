"""Guard action geometry parity and evaluation protocol against silent drift."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from overfitted_underqualified.ml.action_inference import R3DActionRecognizer
from overfitted_underqualified.ml.action_evaluation import evaluate_actions
from overfitted_underqualified.ml.preprocessing import video_transform

NAME='client_identity_check_not_visible'


class IdentityExperimentTests(unittest.TestCase):
    def test_inference_uses_the_exact_checkpoint_geometry(self):
        config=dict(weights='KINETICS400_V1',frames=16,threshold=.7,background_ceiling=.2,
                    window_sec=4.,preprocessing='aspect_letterbox_v1',roi=[0.,.1,1.,.65])
        checkpoint=dict(class_names=[NAME],validated_types=[NAME],config=config,schema='test')
        with patch('overfitted_underqualified.ml.action_inference.load_action_checkpoint',
                   return_value=(torch.nn.Identity(),checkpoint,'synthetic')):
            recognizer=R3DActionRecognizer('synthetic.pt',enabled_types=[NAME])
        frames=torch.randint(256,(16,3,160,80),dtype=torch.uint8)
        torch.testing.assert_close(recognizer.transform(frames),video_transform(config)(frames),rtol=0,atol=0)
        self.assertEqual(recognizer.provenance['roi'],config['roi'])

    def test_evaluation_rejects_different_geometry_before_decoding(self):
        recognizer=SimpleNamespace(classes=[NAME],checkpoint=dict(config=dict(preprocessing='aspect_letterbox_v1',roi=[0.,.1,1.,.65])))
        with patch('overfitted_underqualified.ml.action_evaluation.prepare_action_data',return_value=([],[],{'class_names':[NAME]})), \
             patch('overfitted_underqualified.ml.action_evaluation.prepare_typed_pipeline',return_value=SimpleNamespace(recognizer=recognizer)):
            with self.assertRaisesRegex(ValueError,'geometry'):
                evaluate_actions({'device':'cpu'},'binary.pt','action.pt',[NAME])


if __name__=='__main__':unittest.main()
