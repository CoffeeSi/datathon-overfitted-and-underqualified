"""Fixed-threshold binary robustness evaluation on validation descendants only."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from zipfile import ZipFile,ZIP_DEFLATED

from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support
from torch.utils.data import DataLoader

from .augmented_data import read_audit,AugmentedBinaryWindows
from .dataset import sha256
from .evaluate import prepare_evaluation,verify_evaluation_inputs
from .train import run_epoch


def video_metrics(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[(row['clip_id'],row['variant'])].append(row)
    videos=[]
    for (clip,variant),windows in grouped.items():
        if len({r['label'] for r in windows})!=1:raise ValueError('Conflicting inherited labels')
        score=max(r['probability'] for r in windows)
        videos.append(dict(clip_id=clip,variant=variant,label=windows[0]['label'],probability=score,prediction=int(score>=.5)))
    truth=[r['label'] for r in videos];pred=[r['prediction'] for r in videos]
    p,r,f,_=precision_recall_fscore_support(truth,pred,average='binary',zero_division=0)
    return dict(videos=len(videos),accuracy=float(accuracy_score(truth,pred)),precision=float(p),recall=float(r),f1=float(f),
                confusion_matrix=confusion_matrix(truth,pred,labels=[0,1]).tolist()),videos


def evaluate(checkpoint,output_dir,*,device='cpu',supplemental_manifest=None,supplemental_manifest_sha256=None):
    output=Path(output_dir);archive=output.with_suffix('.zip')
    if output.exists() or archive.exists():raise FileExistsError('Use a fresh robustness report')
    start=time.perf_counter();prepared=prepare_evaluation(checkpoint,device=device)
    cfg=prepared['config']
    if supplemental_manifest is not None:
        trained_pin=cfg.get('supplemental_manifest_sha256')
        if trained_pin and trained_pin!=supplemental_manifest_sha256:raise ValueError('Different audit from augmented training provenance')
        cfg=dict(cfg,supplemental_manifest=str(supplemental_manifest),supplemental_manifest_sha256=supplemental_manifest_sha256)
    audit=read_audit(cfg['supplemental_manifest'],cfg['supplemental_manifest_sha256'],cfg['events_path'],cfg['split_path'])
    original_metrics,original_rows=run_epoch(prepared['model'],prepared['loader'],prepared['device'])
    for row in original_rows:row['variant']='original'
    original_video,original_video_rows=video_metrics(original_rows)
    dataset=AugmentedBinaryWindows(prepared['dataset'],audit,'validation',include_original=False)
    loader=DataLoader(dataset,batch_size=min(2,cfg['batch_size']),shuffle=False,num_workers=0)
    metrics,rows=run_epoch(prepared['model'],loader,prepared['device'])
    if len(rows)!=len(dataset.source_manifest):raise ValueError('Incomplete augmented prediction coverage')
    for row,source in zip(rows,dataset.source_manifest):
        if (row['clip_id'],row['center_sec'])!=(source['clip_id'],source['center_sec']):raise ValueError('Coverage mismatch')
        row.update(variant=source['variant'],source_sha256=source['sha256'])
    combined,videos=video_metrics(rows)
    per_variant={variant:video_metrics([r for r in rows if r['variant']==variant])[0] for variant in sorted({r['variant'] for r in rows})}
    verify_evaluation_inputs(prepared)
    if sha256(Path(checkpoint))!=prepared['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    for source in {s['path']:s for s in dataset.source_manifest}.values():
        if sha256(Path(source['path']))!=source['sha256']:raise ValueError('Augmentation changed during evaluation')
    result=dict(schema='binary_augmentation_robustness_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=prepared['checkpoint_sha256'],
        config=cfg,events_sha256=prepared['evaluation_events_sha256'],split_sha256=prepared['checkpoint']['data_manifest']['split_sha256'],
        audit_sha256=cfg['supplemental_manifest_sha256'],original_window_metrics=original_metrics,
        original_video_metrics=original_video,augmented_window_metrics=metrics,transformed_video_metrics=combined,
        per_variant_video_metrics=per_variant,canonical_validation_parents=len({r['clip_id'] for r in rows}),
        threshold=.5,processing_sec=time.perf_counter()-start,independent_evaluation=False,
        limitations=['Annotation-centred binary windows, not full-video localization or action-type recognition.',
          'Four descendants of each validation parent are correlated robustness cases, not new independent videos.',
          ('Train descendants only fit weights; fixed final checkpoint selected without validation; validation descendants never optimize weights or thresholds.'
           if cfg.get('checkpoint_selection')=='fixed_final_step_no_validation' else
           'Train descendants only fit weights; original validation selects checkpoint; validation descendants never optimize weights or thresholds.'),
          'Mirror orientation restored before fixed ROI; all other preprocessing matches training.',
          'Same room/people and repeatedly inspected validation; no independent test.'])
    output.mkdir(parents=True,exist_ok=False)
    artifacts={'summary.json':result,'window_predictions.json':rows,'video_predictions.json':videos,
        'original_window_predictions.json':original_rows,'original_video_predictions.json':original_video_rows,
        'evaluation_sources.json':dataset.source_manifest,
        'errors.json':[r for r in videos if r['label']!=r['prediction']]}
    for name,value in artifacts.items():(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with ZipFile(archive,'x',ZIP_DEFLATED) as stream:
        for name in artifacts:stream.write(output/name,name)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--supplemental-manifest',type=Path)
    parser.add_argument('--supplemental-manifest-sha256')
    args=parser.parse_args();result=evaluate(args.checkpoint,args.output_dir,device=args.device,
        supplemental_manifest=args.supplemental_manifest,supplemental_manifest_sha256=args.supplemental_manifest_sha256)
    print(json.dumps({k:result[k] for k in ('original_video_metrics','transformed_video_metrics','per_variant_video_metrics','processing_sec')},indent=2))


if __name__=='__main__':main()
