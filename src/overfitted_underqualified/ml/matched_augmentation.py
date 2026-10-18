"""Matched fixed-step binary experiment; validation is evaluated after both fits.

No threshold/epoch search and no promise of perfect accuracy. All descendants
remain in their canonical partition. Current production settings are not edited.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import yaml

from .augmented_evaluation import evaluate
from .dataset import sha256
from .features import cache_frozen_loaders
from .supplemental_data import AUGMENTATIONS
from .train import ROOT, build_model, prepare_training, save_checkpoint


def training_probabilities(source_manifest, sample_weights, original_fraction):
    if not isinstance(original_fraction, (int,float)) or not 0 < original_fraction <= 1:
        raise ValueError('Original fraction must be in (0,1]')
    weights=torch.as_tensor(sample_weights,dtype=torch.float64).clone()
    if (weights.ndim!=1 or len(weights)!=len(source_manifest) or not len(weights)
            or not torch.isfinite(weights).all() or (weights<=0).any()):
        raise ValueError('Invalid sample weights')
    groups={}
    for index,row in enumerate(source_manifest):
        if row.get('split')!='train':raise ValueError('Validation cannot enter training probabilities')
        key=(row['clip_id'],row['center_sec'])
        group=groups.setdefault(key,{})
        variant=row['variant']
        if variant in group:raise ValueError('Duplicate source variant')
        group[variant]=index
    expected=AUGMENTATIONS|{'original'}
    for group in groups.values():
        if set(group)!=expected:raise ValueError('Each training window needs original and four variants')
        values=weights[list(group.values())]
        if not torch.allclose(values,values[0].expand_as(values)):raise ValueError('Variant weights differ within parent window')
    for index,row in enumerate(source_manifest):
        weights[index]*=original_fraction if row['variant']=='original' else (1-original_fraction)/len(AUGMENTATIONS)
    return weights/weights.sum()


def fit_head(features, labels, probabilities, initial_state, *, steps, batch_size,
             learning_rate, weight_decay, seed, device):
    """Only train tensors are accepted; exactly ``steps`` optimizer updates."""
    features=features.detach().cpu();labels=labels.detach().cpu()
    probabilities=torch.as_tensor(probabilities,dtype=torch.float64).cpu()
    if (features.ndim!=2 or not torch.isfinite(features).all() or labels.shape!=(len(features),)
            or probabilities.shape!=(len(features),) or not torch.isfinite(probabilities).all()
            or (probabilities<0).any() or not torch.isclose(probabilities.sum(),torch.tensor(1.,dtype=torch.float64))
            or set(labels.tolist())!={0,1}):
        raise ValueError('Invalid training feature/label/probability inputs')
    if (type(steps) is not int or steps<1 or type(batch_size) is not int or batch_size<1
            or learning_rate<=0 or weight_decay<0):raise ValueError('Invalid fixed-step optimizer settings')
    head=nn.Linear(features.shape[1],2)
    head.load_state_dict(initial_state,strict=True);head.to(device).train()
    optimizer=torch.optim.AdamW(head.parameters(),lr=learning_rate,weight_decay=weight_decay)
    generator=torch.Generator().manual_seed(seed);history=[]
    for step in range(1,steps+1):
        indices=torch.multinomial(probabilities,batch_size,replacement=True,generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(head(features[indices].to(device)),labels[indices].to(device))
        if not torch.isfinite(loss):raise RuntimeError('Non-finite matched training loss')
        loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        history.append(dict(step=step,train_batch_loss=float(loss.detach())))
    return head,optimizer,history


def backbone_digest(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if not name.startswith('fc.'):
            digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run(config):
    config=dict(config)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=ROOT/'runs'/('r3d18_matched_'+stamp)
    report_root=ROOT/'reports'/('evaluation_matched_'+stamp)
    if output.exists() or report_root.exists():raise FileExistsError('Use fresh study directories')
    config.update(output_dir=str(output),epochs=1)
    prepared=prepare_training(config);config=prepared['config']
    if not config['freeze_backbone'] or not config.get('supplemental_manifest'):
        raise ValueError('Matched study requires frozen backbone and audited augmentations')
    plan=dict(schema='matched_augmentation_plan_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        config=config,steps=config['matched_steps'],original_fractions={'control':1.,'mild_augmented':.8},
        selection='Fixed final step; no validation-based epoch, threshold or model selection.',
        limitation='Previously inspected development validation, not a new independent test.',
        events_sha256=prepared['manifest']['events_sha256'],split_sha256=prepared['manifest']['split_sha256'])
    write_json(ROOT/'reports'/('matched_augmentation_plan_'+stamp+'.json'),plan)
    model=build_model(config['weights'],True).to(prepared['device']).eval()
    frozen_before=backbone_digest(model);initial=copy.deepcopy(model.fc.state_dict())
    print('Extracting train-only frozen features; validation not decoded during fitting.',flush=True)
    cache=cache_frozen_loaders(model,{'train':prepared['loaders']['train']},prepared['device'],config['batch_size'],config['seed'])['train'].dataset
    source=prepared['datasets']['train'].source_manifest
    if len(cache)!=len(source):raise ValueError('Feature/source coverage mismatch')
    for row,provenance in zip(cache,source):
        if (row['clip_id'],float(row['center_sec']),int(row['label']))!=(provenance['clip_id'],provenance['center_sec'],provenance['label']):
            raise ValueError('Feature source ordering changed')
    features=torch.stack([r['features'] for r in cache]);labels=torch.tensor([int(r['label']) for r in cache])
    weights=torch.tensor([float(r['sample_weight']) for r in cache]);runs={}
    for name,fraction in plan['original_fractions'].items():
        probabilities=training_probabilities(source,weights,fraction)
        head,optimizer,history=fit_head(features,labels,probabilities,initial,steps=config['matched_steps'],
            batch_size=config['batch_size'],learning_rate=config['learning_rate'],weight_decay=config['weight_decay'],
            seed=config['seed'],device=prepared['device'])
        model.fc=head
        if backbone_digest(model)!=frozen_before:raise ValueError('Frozen backbone/BatchNorm changed')
        for key,expected in (('events_path',plan['events_sha256']),('split_path',plan['split_sha256']),('supplemental_manifest',config['supplemental_manifest_sha256'])):
            if sha256(Path(config[key]))!=expected:raise ValueError('Source changed during training')
        run_dir=output/name;run_dir.mkdir(parents=True,exist_ok=False)
        run_config=dict(config,output_dir=str(run_dir),original_fraction=fraction,checkpoint_selection='fixed_final_step_no_validation')
        manifest=copy.deepcopy(prepared['manifest'])
        manifest['supplemental']['sampling_policy']=dict(original_fraction=fraction,steps=config['matched_steps'],threshold=.5)
        save_checkpoint(run_dir/'last.pt',model,optimizer,1,run_config,manifest,{'optimization_steps':config['matched_steps']})
        (run_dir/'config.yaml').write_text(yaml.safe_dump(run_config,sort_keys=False),encoding='utf-8')
        write_json(run_dir/'history.json',history);write_json(run_dir/'data_manifest.json',manifest)
        write_json(run_dir/'checks.json',dict(backbone_before=frozen_before,backbone_after=backbone_digest(model),
             optimizer_steps=len(history),validation_used_during_training=False,threshold=.5,
             initial_head_sha256=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in initial.values())).hexdigest()))
        runs[name]=str(run_dir)
        print(name,'finished',len(history),'updates',flush=True)
    del cache,features,labels,weights
    # Both models are frozen/saved before the first validation prediction.
    results={}
    for name,run_dir in runs.items():
        results[name]=evaluate(Path(run_dir)/'last.pt',report_root/name,device=str(prepared['device']))
    result=dict(schema='matched_augmentation_study_v1',run=str(output),reports=str(report_root),plan=plan,
        results=results,independent_evaluation=False,production_model_replaced=False,
        reached_100_percent_all_scopes=all(r['original_video_metrics']['accuracy']==1 and r['transformed_video_metrics']['accuracy']==1 for r in results.values()))
    path=ROOT/'reports'/('matched_augmentation_study_'+stamp+'.json');write_json(path,result)
    print(json.dumps(dict(study=str(path),metrics={k:{m:v[m] for m in ('original_video_metrics','transformed_video_metrics')} for k,v in results.items()}),indent=2),flush=True)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();run(yaml.safe_load(args.config.read_text(encoding='utf-8')))


if __name__=='__main__':main()
