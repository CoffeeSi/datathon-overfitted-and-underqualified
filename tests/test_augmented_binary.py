import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from overfitted_underqualified.ml.augmented_data import AugmentedBinaryWindows,read_audit,sources_for_record
from overfitted_underqualified.ml.augmented_evaluation import video_metrics
from overfitted_underqualified.ml.dataset import VideoRecord,sha256
from overfitted_underqualified.ml.features import WeightedWindows
from overfitted_underqualified.ml.supplemental_data import AUGMENTATIONS,SCHEMA


class AugmentedBinaryTests(unittest.TestCase):
    def fixture(self,directory,subset='train',clip='X-4',label=1):
        root=Path(directory);original=root/(clip+'.mov');original.write_bytes(b'original'+clip.encode())
        record=VideoRecord(clip,original,subset,label,sha256(original),(clip,), (57.,68.) if label else (),1.,100)
        parent=dict(sha256=record.sha256,split=subset);items=[]
        for variant in sorted(AUGMENTATIONS):
            path=root/(clip+'__'+variant+'.mov');path.write_bytes((clip+variant).encode())
            items.append(dict(source_id=clip,parent_id=clip,parent_sha256=record.sha256,inherited_split=subset,
                training_eligible=subset=='train',status='eligible_train_parent' if subset=='train' else 'quarantined_validation_descendant',
                augmentation=variant,path=str(path),sha256=sha256(path)))
        return record,dict(parents={clip:parent},augmentations=items)

    def test_partition_and_content_guards(self):
        with tempfile.TemporaryDirectory() as d:
            record,audit=self.fixture(d,'validation')
            self.assertEqual(len(sources_for_record(audit,record,'validation')),4)
            with self.assertRaisesRegex(ValueError,'partitions'):sources_for_record(audit,record,'train')
            audit['augmentations'][0]['training_eligible']=True
            with self.assertRaisesRegex(ValueError,'provenance'):sources_for_record(audit,record,'validation')
            audit['augmentations'][0]['training_eligible']=False
            Path(audit['augmentations'][0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'bytes changed'):sources_for_record(audit,record,'validation')

    def test_all_variants_preserve_alias_times_and_canonical_weights(self):
        with tempfile.TemporaryDirectory() as d:
            positive,a=self.fixture(d);negative,b=self.fixture(d,clip='X-5',label=0)
            audit=dict(parents={**a['parents'],**b['parents']},augmentations=a['augmentations']+b['augmentations'])
            base=SimpleNamespace(samples=[(positive,57.),(positive,68.),(negative,20.)])
            augmented=AugmentedBinaryWindows(base,audit,'train')
            self.assertEqual(len(augmented),15)
            self.assertEqual([t for r,t in augmented.samples if r.clip_id=='X-4'].count(68.),5)
            weighted=WeightedWindows(augmented)
            masses=[sum(w for w,(r,_) in zip(weighted.weights,augmented.samples) if r.label==label) for label in (0,1)]
            self.assertEqual(masses,[7.5,7.5])
            with self.assertRaisesRegex(ValueError,'partitions'):AugmentedBinaryWindows(base,audit,'validation')

    def test_mirror_restored_before_transform_and_no_external_label_used(self):
        with tempfile.TemporaryDirectory() as d:
            record,audit=self.fixture(d)
            observed=[]
            def transform(frames):
                observed.append(frames.clone());return torch.zeros(3,16,112,112)
            base=SimpleNamespace(samples=[(record,57.)],window_sec=4.,frames=16,transform=transform)
            dataset=AugmentedBinaryWindows(base,audit,'train',include_original=False)
            index=next(i for i,(_,source) in enumerate(dataset.items) if source.augmentation=='mirror')
            frames=torch.arange(4).reshape(1,1,1,4).expand(16,3,3,4).to(torch.uint8)
            with patch('overfitted_underqualified.ml.augmented_data.decode_frames',return_value=frames):result=dataset[index]
            self.assertTrue(torch.equal(observed[0],frames.flip(-1)))
            self.assertEqual((result['clip_id'],result['center_sec'],result['label'].item()),('X-4',57.,1))

    def test_audit_and_annotation_hash_pins(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);events=root/'events';split=root/'split';path=root/'audit.json'
            events.write_bytes(b'labels');split.write_bytes(b'partitions')
            path.write_text(json.dumps(dict(schema=SCHEMA,inputs=dict(events=dict(sha256=sha256(events)),split=dict(sha256=sha256(split))))))
            pinned=sha256(path);read_audit(path,pinned,events,split)
            events.write_bytes(b'changed labels')
            with self.assertRaisesRegex(ValueError,'changed'):read_audit(path,pinned,events,split)
            with self.assertRaisesRegex(ValueError,'audit hash'):read_audit(path,'a'*64,events,split)

    def test_notebook_preview_reads_selected_augmented_video(self):
        from overfitted_underqualified.ml.plots import show_windows
        with tempfile.TemporaryDirectory() as d:
            record,audit=self.fixture(d)
            base=SimpleNamespace(samples=[(record,57.)],window_sec=4.,frames=16)
            dataset=AugmentedBinaryWindows(base,audit,'train',include_original=False)
            frames=torch.zeros(16,3,3,4,dtype=torch.uint8)
            with patch('overfitted_underqualified.ml.augmented_data.decode_frames',return_value=frames) as decode,patch('overfitted_underqualified.ml.plots.decode_frames',side_effect=AssertionError('Must not show parent instead of variant')),patch('matplotlib.pyplot.show'):
                show_windows(dataset,[0])
            self.assertEqual(decode.call_args.args[0].path,dataset.items[0][1].path)

    def test_robustness_variants_are_not_pooled_into_one_success(self):
        rows=[]
        for index,variant in enumerate(sorted(AUGMENTATIONS)):
            rows.extend([dict(clip_id='X-38',variant=variant,label=1,probability=.9 if index==0 else .1),
                         dict(clip_id='X-2',variant=variant,label=0,probability=.1)])
        metrics,videos=video_metrics(rows)
        self.assertEqual(len(videos),8)
        self.assertEqual(metrics['confusion_matrix'],[[4,0],[3,1]])
        self.assertEqual(metrics['recall'],.25)


if __name__=='__main__':unittest.main()
