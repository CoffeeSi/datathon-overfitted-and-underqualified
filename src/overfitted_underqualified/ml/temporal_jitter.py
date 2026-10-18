"""Anchor-preserving positive-window jitter; reuse verified train-only features."""
import argparse
from collections import Counter
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from .augmented_data import AugmentedBinaryWindows,read_audit
from .augmented_evaluation import evaluate
from .dataset import VideoWindowDataset,sample_times,sha256
from .features import cache_frozen_loaders
from .matched_augmentation import backbone_digest,fit_head,training_probabilities,write_json
from .regularized_heads import diagnose
from .train import ROOT,build_model,load_checkpoint,prepare_training,save_checkpoint,seed_everything


def positive_jitter(base,offsets):
    """Derive train windows only when the agreed point remains inside input support."""
    if (not offsets or len(set(offsets))!=len(offsets)
            or any(not math.isfinite(x) or x==0 or abs(x)>base.window_sec/2 for x in offsets)
            or any(record.split!='train' for record,_ in base.samples)):
        raise ValueError('Jitter requires train samples and finite nonzero bounded offsets')
    extra=copy.copy(base);extra.samples=[];lineage=[]
    seen={(r.clip_id,t) for r,t in base.samples}
    for record,anchor in base.samples:
        if record.label!=1:continue
        for offset in offsets:
            center=float(min(max(anchor+offset,0),record.last_time))
            times=sample_times(center,record.last_time,base.window_sec,base.frames)
            if not times[0]-1e-9<=anchor<=times[-1]+1e-9:
                raise ValueError('Jittered support would lose its labelled event point')
            if (record.clip_id,center) in seen:continue
            seen.add((record.clip_id,center));extra.samples.append((record,center))
            lineage.append(dict(clip_id=record.clip_id,center_sec=center,event_anchor_sec=anchor,
                requested_offset_sec=offset,source_ids=list(record.aliases),label=1,
                support_start_sec=float(times[0]),support_end_sec=float(times[-1])))
    return extra,lineage


def balanced_probabilities(sources,fraction):
    counts=Counter(r['clip_id'] for r in sources)
    classes={c:len({r['clip_id'] for r in sources if r['label']==c}) for c in (0,1)}
    if not all(classes.values()):raise ValueError('Both train classes required')
    weights=[len(sources)/(2*classes[r['label']]*counts[r['clip_id']]) for r in sources]
    return training_probabilities(sources,torch.tensor(weights,dtype=torch.float32),fraction)


def run(settings):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=ROOT/'runs'/f'r3d18_temporal_jitter_{stamp}'
    report=ROOT/'reports'/f'evaluation_temporal_jitter_{stamp}'
    summary_path=ROOT/'reports'/f'temporal_jitter_study_{stamp}.json'
    checkpoint=ROOT/settings['checkpoint'];baseline_digest=sha256(checkpoint)
    feature_path=ROOT/settings['feature_bank'];meta_path=ROOT/settings['feature_manifest']
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    if sha256(feature_path)!=meta['cache_sha256']:raise ValueError('Changed train feature cache')
    bank=torch.load(feature_path,map_location='cpu',weights_only=True)
    old=bank['plan']
    if baseline_digest!=old['baseline_sha256']:raise ValueError('Cache from another model')
    for file,key in (('data/annotations/events.csv','events_sha256'),('data/processed/split_proposal.csv','split_sha256')):
        if sha256(ROOT/file)!=old[key]:raise ValueError('Cache from another annotation/split')
    model,checkpoint_data=load_checkpoint(checkpoint,settings['device'])
    if backbone_digest(model)!=old['backbone_sha256']:raise ValueError('Cache backbone changed')
    cfg=dict(checkpoint_data['config'],output_dir=str(output),seed=settings['seed'],cpu_threads=settings['cpu_threads'])
    prepared=prepare_training(cfg);cfg=prepared['config'];device=prepared['device']
    source=prepared['datasets']['train'].source_manifest
    if source!=meta['source_manifest']:raise ValueError('Training cache/source mismatch')
    source_prob=balanced_probabilities(source,settings['original_fraction'])
    torch.testing.assert_close(source_prob,bank['probabilities'],rtol=1e-6,atol=1e-9)
    output.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    plan=dict(schema='positive_temporal_jitter_plan_v1',settings=settings,threshold=.5,
        baseline_sha256=baseline_digest,feature_cache_sha256=meta['cache_sha256'],
        events_sha256=old['events_sha256'],split_sha256=old['split_sha256'],
        rationale='Additional temporal views retain the labelled point; no new independent examples claimed.',
        selection='Fixed offsets and 120 updates; no validation-driven epoch or threshold search.',
        previous_failed_phase=str(ROOT/settings['baseline_report']),independent_evaluation=False)
    write_json(report/'plan.json',plan)
    seed_everything(settings['seed'],settings['cpu_threads'])
    initial_model=build_model(None,True)
    initial=copy.deepcopy(initial_model.fc.state_dict());del initial_model
    head_sha=hashlib.sha256(b''.join(t.numpy().tobytes() for t in initial.values())).hexdigest()
    old_checks=json.loads((checkpoint.parent/'checks.json').read_text(encoding='utf-8'))
    if head_sha!=old_checks['initial_head_sha256']:raise ValueError('Initial head differs from matched baseline')
    fit_args={k:settings[k] for k in ('steps','batch_size','learning_rate','weight_decay','seed')}
    control,_,_=fit_head(bank['features'],bank['labels'],bank['probabilities'],initial,device=device,**fit_args)
    maxdiff=max(float((control.state_dict()[k].cpu()-model.fc.state_dict()[k].cpu()).abs().max()) for k in initial)
    if maxdiff>1e-6:raise RuntimeError('Paired control failed to reproduce baseline head')
    write_json(report/'control_checks.json',dict(initial_head_sha256=head_sha,max_head_diff_vs_baseline=maxdiff,
        training_only=True,baseline_control_reproduced=True))
    del control
    extra_base,lineage=positive_jitter(prepared['datasets']['train'].base,settings['positive_offsets_sec'])
    audit=read_audit(Path(cfg['supplemental_manifest']),cfg['supplemental_manifest_sha256'],cfg['events_path'],cfg['split_path'])
    extra=AugmentedBinaryWindows(extra_base,audit,'train')
    print('EXTRA_WINDOWS '+str(len(extra)),flush=True)
    extra_loader=DataLoader(extra,batch_size=2,shuffle=False,num_workers=0)
    cached=cache_frozen_loaders(model,{'train_extra':extra_loader},device,2,settings['seed'])['train_extra'].dataset
    for value,s in zip(cached,extra.source_manifest):
        if (value['clip_id'],float(value['center_sec']),int(value['label']))!=(s['clip_id'],s['center_sec'],s['label']):
            raise ValueError('Extra source/feature order mismatch')
    if len(cached)!=len(extra):raise ValueError('Extra feature coverage mismatch')
    features=torch.cat([bank['features'],torch.stack([r['features'] for r in cached])])
    labels=torch.cat([bank['labels'],torch.tensor([int(r['label']) for r in cached])])
    sources=source+extra.source_manifest
    probabilities=balanced_probabilities(sources,settings['original_fraction'])
    head,optimizer,history=fit_head(features,labels,probabilities,initial,device=device,**fit_args)
    model.fc=head
    if backbone_digest(model)!=old['backbone_sha256']:raise RuntimeError('Frozen backbone/BN changed')
    manifest=copy.deepcopy(prepared['manifest'])
    manifest['windows']['train']=[dict(clip_id=s['clip_id'],center_sec=s['center_sec'],label=s['label']) for s in sources]
    manifest['supplemental']['training_sources']=sources
    manifest['positive_window_lineage']=lineage
    manifest['supplemental']['sampling_policy']=dict(original_fraction=settings['original_fraction'],steps=settings['steps'])
    config=dict(cfg,output_dir=str(output),positive_offsets_sec=settings['positive_offsets_sec'],
        checkpoint_selection='fixed_final_step_no_validation',temporal_jitter_settings=settings)
    save_checkpoint(output/'last.pt',model,optimizer,1,config,manifest,{'optimization_steps':settings['steps']})
    (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    write_json(output/'history.json',history);write_json(output/'data_manifest.json',manifest)
    write_json(report/'training_lineage.json',lineage)
    print('CHECKPOINT_SAVED '+str(output/'last.pt'),flush=True)
    original_eval=prepared['datasets']['validation']
    diagnostic_data={s:VideoWindowDataset(prepared['records'],s,original_eval.transform,cfg['window_sec'],16,1,
        negative_sampling=cfg.get('negative_sampling','interior')) for s in ('train','validation')}
    del features,labels,cached,bank
    diagnostics=diagnose(model,diagnostic_data,device);del model
    robustness=evaluate(output/'last.pt',report/'jitter',device=str(device))
    previous_path=ROOT/settings['baseline_report']
    previous=json.loads(previous_path.read_text(encoding='utf-8'))
    if previous['plan']['baseline_sha256']!=baseline_digest:raise ValueError('Different baseline report')
    result=dict(schema='positive_temporal_jitter_study_v1',plan=plan,checkpoint=str(output/'last.pt'),
        checkpoint_sha256=sha256(output/'last.pt'),diagnostics=diagnostics,robustness=robustness,
        baseline=dict(diagnostics=previous['diagnostics']['baseline'],robustness=previous['robustness']['baseline'],
            source_report=str(previous_path),source_sha256=sha256(previous_path)),
        training_windows=len(sources),additional_windows=len(extra),canonical_training_videos=len({s['clip_id'] for s in sources}),
        initial_head_sha256=head_sha,control_head_max_difference=maxdiff,
        backbone_unchanged=True,active_model_replaced=False,independent_evaluation=False)
    for key,expected in (('events_path',old['events_sha256']),('split_path',old['split_sha256']),('supplemental_manifest',cfg['supplemental_manifest_sha256'])):
        if sha256(Path(cfg[key]))!=expected:raise RuntimeError('Changed data')
    for s in {s['path']:s for s in sources}.values():
        if sha256(Path(s['path']))!=s['sha256']:raise RuntimeError('Changed training video')
    if sha256(checkpoint)!=baseline_digest:raise RuntimeError('Changed original checkpoint')
    write_json(summary_path,result)
    lines=['# Положительные окна со сдвигом времени','',
        'Фиксированы ±1 s, 4 s окно, 16 кадров, 120 обновлений головы и порог 0.5. '
        'Каждое дополнительное окно содержит исходную отметку; метки и split сохранены. '
        'Сдвиги не являются новыми независимыми видео. Контроль воспроизвёл исходную голову.','']
    for name,data in (('baseline',result['baseline']['diagnostics']),('temporal_jitter',diagnostics)):
        lines += [f'## {name}','']
        for subset,v in data.items():
            m=v['video_metrics'];lines += [f"- {subset}: accuracy={m['accuracy']:.3f}, precision={m['precision']:.3f}, "
                f"recall={m['recall']:.3f}, F1={m['f1']:.3f}; confusion={m['confusion_matrix']}."]
        lines += ['']
    lines += ['Метрики относятся к окнам вокруг известных отметок; это не независимый test '
        'или подтверждение качества детектора всего видео. Старый checkpoint и backend не переключены.', '',
        'Предыдущая фаза с регуляризованными головами сохранена отдельно; её неудачные результаты не скрыты.']
    summary_path.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('STUDY_SAVED '+str(summary_path),flush=True)
    return summary_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/r3d18_temporal_jitter.yaml')
    args=parser.parse_args();run(yaml.safe_load(args.config.read_text(encoding='utf-8')))


if __name__=='__main__':main()
