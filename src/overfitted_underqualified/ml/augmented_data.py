"""Binary windows from audited descendants; canonical parents retain split and weight."""
from dataclasses import replace
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset import decode_frames, sample_times, sha256
from .supplemental_data import AUGMENTATIONS, SCHEMA, AugmentedSource


def read_audit(path, expected_sha256, events_path, split_path):
    path=Path(path)
    if sha256(path)!=expected_sha256:
        raise ValueError('Supplemental audit hash mismatch')
    value=json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema')!=SCHEMA:
        raise ValueError('Unsupported supplemental audit')
    for key,source in (('events',events_path),('split',split_path)):
        if sha256(Path(source))!=value['inputs'][key]['sha256']:
            raise ValueError('Supplemental source labels/split changed')
    return value


def sources_for_record(audit, record, subset):
    """Validation descendants are permitted only for separate robustness evaluation."""
    if subset not in {'train','validation'} or record.split!=subset:
        raise ValueError('Augmentation requested across partitions')
    parent=audit['parents'].get(record.clip_id)
    if not parent or parent['sha256']!=record.sha256 or parent['split']!=subset:
        raise ValueError('Augmentation parent differs from canonical video')
    expected='eligible_train_parent' if subset=='train' else 'quarantined_validation_descendant'
    candidates=sorted(audit['augmentations'],key=lambda r:(r.get('source_id')!=record.clip_id,r.get('path','')))
    result=[];seen=set()
    for item in candidates:
        if item.get('parent_id')!=record.clip_id or item.get('status')!=expected:continue
        if (item.get('inherited_split')!=subset or item.get('parent_sha256')!=record.sha256
                or item.get('training_eligible')!=(subset=='train')):
            raise ValueError('Invalid supplemental partition/provenance')
        variant=item['augmentation']
        if variant not in AUGMENTATIONS:raise ValueError('Unsupported augmentation')
        if variant in seen:continue
        path=Path(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('Augmentation bytes changed')
        result.append(AugmentedSource(path,item['sha256'],record.clip_id,variant,variant=='mirror'));seen.add(variant)
    if seen!=AUGMENTATIONS:
        raise ValueError(f'Incomplete verified augmentation set for {record.clip_id}')
    return result


class AugmentedBinaryWindows(Dataset):
    """Use labels and times exclusively from the original VideoWindowDataset.

    All four audited versions are represented once per original window. Mirror is
    undone before the fixed work-area crop. There is no random variant selection
    inside a frozen feature cache. Validation wrappers never enter train loaders.
    """
    def __init__(self, base, audit, subset, *, include_original=True):
        self.base=base;self.items=[];self.samples=[];self.source_manifest=[]
        records={r.clip_id:r for r,_ in base.samples}
        sources={key:sources_for_record(audit,r,subset) for key,r in records.items()}
        for index,(record,center) in enumerate(base.samples):
            for source in ([None] if include_original else [])+sources[record.clip_id]:
                self.items.append((index,source));self.samples.append((record,center))
                self.source_manifest.append(dict(clip_id=record.clip_id,center_sec=center,label=record.label,
                    variant=source.augmentation if source else 'original',
                    sha256=source.sha256 if source else record.sha256,
                    path=str(source.path if source else record.path),split=subset))

    def __len__(self):return len(self.items)

    @property
    def window_sec(self):return self.base.window_sec

    @property
    def frames(self):return self.base.frames

    def decode_window(self,index):
        base_index,source=self.items[index]
        record,center=self.base.samples[base_index]
        alternate=replace(record,path=source.path,sha256=source.sha256) if source else record
        frames=decode_frames(alternate,sample_times(center,record.last_time,self.window_sec,self.frames))
        return source.restore_orientation(frames) if source else frames

    def __getitem__(self,index):
        base_index,source=self.items[index]
        if source is None:return self.base[base_index]
        record,center=self.base.samples[base_index]
        video=self.base.transform(self.decode_window(index))
        if video.shape!=(3,16,112,112) or not torch.isfinite(video).all():
            raise ValueError('Invalid augmented window')
        return dict(video=video,label=torch.tensor(record.label,dtype=torch.long),clip_id=record.clip_id,center_sec=center)
