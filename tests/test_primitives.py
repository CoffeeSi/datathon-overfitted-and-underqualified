"""Causal semantics and leakage guards; synthetic tests are not model quality claims."""
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
from overfitted_underqualified.ml.cash_sequence import CashSequence
from overfitted_underqualified.ml.primitive_data import (
    CLASSES, SCHEMA, DEFINITIONS, PrimitiveSample, PrimitiveDataset, causal_times, validate_review, prepare_primitive_data,
)
from overfitted_underqualified.ml.primitive_model import load_primitive_checkpoint
from overfitted_underqualified.ml.dataset import annotation_fields_digest, sha256, verify_annotation_revision


def scores(client, cash, receipt):
    return dict(zip(CLASSES, (client, cash, receipt)))


class PrimitiveTests(unittest.TestCase):
    def test_causal_windows_never_see_future_even_at_start(self):
        for end in (0., .25, 1., 10.):
            times = causal_times(end, 10., 2.)
            self.assertEqual(len(times), 16)
            self.assertEqual(times[-1], end)
            self.assertGreaterEqual(times[0], 0)
            self.assertTrue(np.all(times <= end))
        for args in ((-1, 10), (11, 10), (float('nan'), 10)):
            with self.assertRaises(ValueError):
                causal_times(*args)

    def test_requires_receipt_and_absence_and_persistent_cash(self):
        s = CashSequence()
        for t in range(6):
            s.update(t, scores(.1, .9, .1))
        self.assertEqual(s.events, [])
        self.assertEqual(len(s.unresolved), 1)
        s = CashSequence()
        s.update(0, scores(.9, .9, .9))
        s.update(1, scores(.9, .1, .1))  # counting in hands/machine preserves receipt history
        s.update(2, scores(.1, .9, .1))
        s.update(3, scores(.1, .9, .1))
        self.assertEqual(s.events, [])
        s.update(4, scores(.1, .9, .1))
        self.assertEqual(s.events[0]['receipt_sec'], 0)
        self.assertEqual(s.events[0]['timestamp_sec'], 4)
        self.assertTrue(s.events[0]['requires_review'])
        self.assertEqual(s.events[0]['status'], 'research_candidate')

    def test_unknowns_gaps_and_expired_receipts_do_not_manufacture_departure(self):
        for mode in ('unknown_client', 'gap', 'ttl'):
            s = CashSequence(receipt_ttl_sec=3 if mode == 'ttl' else 120)
            s.update(0, scores(.9, .9, .9))
            stamps = range(5, 9) if mode == 'gap' else range(1, 7)
            for t in stamps:
                s.update(t, scores(.5 if mode == 'unknown_client' else .1, .9, .1))
            if mode == 'ttl':
                # A candidate at 3s is within TTL; after expiry it cannot generate another.
                self.assertLessEqual(len(s.events), 1)
                self.assertIsNone(s.receipt)
            else:
                self.assertEqual(s.events, [])
        s = CashSequence()
        s.update(0, scores(.9, .9, .9))
        s.update(1, scores(.1, .9, .1))
        s.update(2, scores(.1, .5, .1))
        s.update(3, scores(.1, .9, .1))
        self.assertEqual(s.events, [])

    def test_clear_counter_and_new_receipt_allow_multiple_events(self):
        s = CashSequence()
        for offset in (0, 7):
            s.update(offset, scores(.9, .9, .9))
            for t in (offset + 1, offset + 2, offset + 3, offset + 4):
                s.update(t, scores(.1, .9, .1))
            s.update(offset + 5, scores(.1, .1, .1))
        self.assertEqual(len(s.events), 2)
        self.assertEqual([e['receipt_sec'] for e in s.events], [0, 7])

    def test_no_event_for_active_service_or_clear_departure(self):
        s = CashSequence()
        s.update(0, scores(.9, .9, .9))
        for t in range(1, 8):
            s.update(t, scores(.9, .9, .1))
        for t in range(8, 12):
            s.update(t, scores(.1, .1, .1))
        self.assertEqual(s.events, [])
        with self.assertRaises(ValueError):
            s.update(11, scores(.1, .1, .1))
        with self.assertRaises(ValueError):
            s.update(12, scores(float('nan'), .1, .1))

    def test_primitive_review_does_not_accept_unsupported_or_guessed_targets(self):
        record = SimpleNamespace(sha256='a' * 64, last_time=10.)
        value = dict(schema=SCHEMA, video_sha256=record.sha256, window_sec=2.,
                     method='frame_sequence_review', reviewer='test', reviewed_at='2026-10-18',
                     points=[dict(time_sec=3., labels={CLASSES[2]: 1}, evidence_seconds=[1., 2., 3.], note='test')])
        validate_review(value, record, 2.)
        for mutation in ('video', 'window', 'type', 'target', 'evidence', 'duplicate'):
            bad = copy.deepcopy(value)
            if mutation == 'video': bad['video_sha256'] = 'b' * 64
            if mutation == 'window': bad['window_sec'] = 4.
            if mutation == 'type': bad['points'][0]['labels'] = {'violation': 1}
            if mutation == 'target': bad['points'][0]['labels'] = {CLASSES[0]: None}
            if mutation == 'evidence': bad['points'][0]['evidence_seconds'] = [3.]
            if mutation == 'duplicate': bad['points'] *= 2
            with self.assertRaises(ValueError, msg=mutation):
                validate_review(bad, record, 2.)

    def test_dataset_masks_unknown_and_uses_endpoint_sampling(self):
        record = SimpleNamespace(clip_id='X-test', last_time=10.)
        sample = PrimitiveSample(record, 1., (CLASSES[0],), (CLASSES[1],))
        with patch('overfitted_underqualified.ml.primitive_data.decode_frames',
                   return_value=torch.zeros(16, 3, 112, 112, dtype=torch.uint8)) as decode:
            data = PrimitiveDataset([sample], lambda x: x.permute(1, 0, 2, 3).float(), 2.)[0]
        self.assertEqual(data['targets'].tolist(), [1, 0, 0])
        self.assertEqual(data['mask'].tolist(), [1, 1, 0])
        self.assertEqual(decode.call_args.args[1][-1], 1.)
        self.assertEqual(decode.call_args.args[1][0], 0.)

    def test_two_views_preserve_order_and_frozen_batchnorm(self):
        from overfitted_underqualified.ml.primitive_model import MultiViewPrimitives
        from overfitted_underqualified.ml.action_train import action_epoch
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm3d(3)
                self.fc = torch.nn.Linear(3, 2)
            def forward(self, x):
                return self.fc(self.bn(x).mean((2, 3, 4)))
        model = MultiViewPrimitives(Backbone(), 2)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        batch = dict(video=torch.rand(2, 2, 3, 2, 8, 8), targets=torch.tensor([[1., 0., 1.], [0., 1., 0.]]),
                     mask=torch.ones(2, 3), clip_id=['a', 'b'], center_sec=[2., 3.])
        action_epoch(model, [batch], 'cpu', CLASSES, .7, torch.optim.AdamW(model.fc.parameters(), lr=.01))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items() if k.startswith('backbone.')))
        self.assertFalse(torch.equal(model.fc.weight, before['fc.weight']))
        after = {k: v.clone() for k, v in model.state_dict().items()}
        action_epoch(model, [batch], 'cpu', CLASSES, .7)
        self.assertTrue(all(torch.equal(v, after[k]) for k, v in model.state_dict().items()))

    def test_duplicate_alias_points_are_merged_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path, split = Path(d) / 'events.csv', Path(d) / 'split.csv'
            split.write_text('test', encoding='utf-8')
            record = SimpleNamespace(clip_id='X-4', aliases=('X-4', 'X-47'), sha256='a' * 64,
                                     last_time=80., split='train')
            review = dict(schema=SCHEMA, video_sha256=record.sha256, window_sec=2., method='frame_sequence_review',
                          reviewer='test', reviewed_at='2026-10-18',
                          points=[dict(time_sec=57., labels={CLASSES[1]: 1}, evidence_seconds=[57.], note='cash')])
            rows = [dict(clip_id=alias, primitive_review=json.dumps(review)) for alias in record.aliases]
            def write():
                with path.open('w', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
                return dict(events_path=str(path), split_path=str(split), video_dir=d, window_sec=2.,
                            events_sha256=sha256(path), split_sha256=sha256(split))
            config = write()
            with patch('overfitted_underqualified.ml.primitive_data.load_records', return_value=[record]):
                _, samples, _ = prepare_primitive_data(config)
                self.assertEqual(len(samples), 1)
                review['points'][0]['labels'][CLASSES[1]] = 0
                rows[1]['primitive_review'] = json.dumps(review)
                with self.assertRaisesRegex(ValueError, 'Conflicting'):
                    prepare_primitive_data(write())

    def test_metadata_addition_cannot_change_binary_labels(self):
        with tempfile.TemporaryDirectory() as d:
            path, registry = Path(d) / 'events.csv', Path(d) / 'revisions.json'
            rows = [dict(clip_id='X-1', source_label='violation', timecode='00:01')]
            fields = list(rows[0])
            def write():
                with path.open('w', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
            write(); before = sha256(path)
            rows[0]['primitive_review'] = '{}'; write()
            entry = dict(schema='binary_primitive_review_revision_v1', before_sha256=before,
                         after_sha256=sha256(path), allowed_changed_fields=['primitive_review'],
                         preserved_fields=fields, preserved_fields_sha256=annotation_fields_digest(rows, fields))
            registry.write_text(json.dumps([entry]), encoding='utf-8')
            self.assertEqual(verify_annotation_revision(path, before, registry_path=registry), entry)
            rows[0]['source_label'] = 'no_violation'; write(); entry['after_sha256'] = sha256(path)
            registry.write_text(json.dumps([entry]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'protected'):
                verify_annotation_revision(path, before, registry_path=registry)

    def test_checkpoint_restoration_never_requests_pretrained_weights(self):
        from overfitted_underqualified.ml.primitive_model import read_config
        config = read_config(Path(__file__).resolve().parents[1] / 'configs/primitives.yaml')
        model = torch.nn.Linear(2, 3)
        checkpoint = dict(schema=SCHEMA, architecture='r3d_18_visible_primitives', class_names=list(CLASSES),
                          definitions=DEFINITIONS, config=config, model_state_dict=model.state_dict(),
                          data_manifest={k: config[k] for k in ('events_sha256', 'split_sha256')})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'test.pt'; torch.save(checkpoint, path)
            with patch('overfitted_underqualified.ml.primitive_model.build_primitive_model', return_value=model) as build:
                restored, _, _ = load_primitive_checkpoint(path)
                build.assert_called_once_with(views=1)
                self.assertFalse(restored.training)


if __name__ == '__main__':
    unittest.main()
