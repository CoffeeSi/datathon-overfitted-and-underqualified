"""Train-only regularized heads, folded into the native R3D-18 linear fc layer."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from .augmented_evaluation import evaluate as evaluate_robustness
from .dataset import LABELS, VideoWindowDataset, sha256
from .evaluate import aggregate_video_predictions, prepare_evaluation
from .features import cache_frozen_loaders
from .matched_augmentation import backbone_digest, training_probabilities, write_json
from .train import ROOT, load_checkpoint, prepare_training, run_epoch, seed_everything


def weighted_standardize(features, labels, probabilities, variance_floor):
    """Fit all statistics on train. A zero-variance feature remains finite."""
    x=features.detach().to(device='cpu',dtype=torch.float64)
    y=labels.detach().to(device='cpu',dtype=torch.long)
    p=torch.as_tensor(probabilities,dtype=torch.float64,device='cpu')
    if (x.ndim!=2 or not len(x) or y.shape!=(len(x),) or p.shape!=(len(x),)
            or not torch.isfinite(x).all() or not torch.isfinite(p).all() or (p<0).any()
            or not torch.isclose(p.sum(),torch.tensor(1.,dtype=p.dtype))
            or set(y.tolist())!={0,1} or not math.isfinite(variance_floor) or variance_floor<=0
            or any(p[y==label].sum()<=0 for label in (0,1))):
        raise ValueError('Invalid train features, binary targets or probabilities')
    mean=(x*p[:,None]).sum(0)
    variance=((x-mean).square()*p[:,None]).sum(0)
    scale=variance.sqrt().clamp_min(variance_floor)
    return (x-mean)/scale,y,p,mean,scale


def fold_standardization(head, mean, scale):
    """Exactly absorb affine normalization; inference still uses ordinary nn.Linear."""
    result=nn.Linear(head.in_features,2,dtype=torch.float64)
    with torch.no_grad():
        result.weight.copy_(head.weight.detach()/scale[None,:])
        result.bias.copy_(head.bias.detach()-result.weight@mean)
    return result


def fit_logistic(features,labels,probabilities,*,l2,max_iter,variance_floor):
    if not math.isfinite(l2) or l2<=0 or type(max_iter) is not int or max_iter<1:raise ValueError('Positive L2 and iterations required')
    z,y,p,mean,scale=weighted_standardize(features,labels,probabilities,variance_floor)
    head=nn.Linear(z.shape[1],2,dtype=torch.float64)
    with torch.no_grad():head.weight.zero_();head.bias.zero_()
    optimizer=torch.optim.LBFGS(head.parameters(),lr=1.,max_iter=max_iter,history_size=20,
        tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    history=[]
    def closure():
        optimizer.zero_grad(set_to_none=True)
        data_loss=(nn.functional.cross_entropy(head(z),y,reduction='none')*p).sum()
        loss=data_loss+.5*l2*head.weight.square().sum()
        if not torch.isfinite(loss):raise RuntimeError('Non-finite regularized loss')
        loss.backward()
        history.append(dict(closure_evaluation=len(history)+1,weighted_cross_entropy=float(data_loss.detach()),
                            regularized_objective=float(loss.detach())))
        return loss
    optimizer.step(closure)
    folded=fold_standardization(head,mean,scale)
    with torch.no_grad():
        error=float((folded(features.double())-head(z)).abs().max())
    if error>1e-8:raise RuntimeError('Normalization folding changed logits')
    return folded.float(),dict(algorithm='weighted_standardized_l2_logistic',l2=l2,
        closure_evaluations=len(history),history=history,fold_error_float64=error,
        normalization_mean=mean,normalization_scale=scale,
        selection='Fixed L2 and train-only convergence criterion; no validation inputs')


def fit_shrinkage_lda(features,labels,probabilities,*,shrinkage,variance_floor):
    if not 0<shrinkage<=1:raise ValueError('LDA shrinkage must be in (0,1]')
    z,y,p,mean,scale=weighted_standardize(features,labels,probabilities,variance_floor)
    masses=torch.stack([p[y==c].sum() for c in (0,1)])
    means=torch.stack([(z[y==c]*p[y==c,None]).sum(0)/masses[c] for c in (0,1)])
    residual=z-means[y]
    covariance=residual.T@(residual*p[:,None])
    regularized=(1-shrinkage)*covariance+shrinkage*torch.diag(covariance.diag())
    regularized+=torch.eye(z.shape[1],dtype=z.dtype)*variance_floor
    direction=torch.linalg.solve(regularized,means[1]-means[0])
    intercept=-.5*(means[1]+means[0])@direction+torch.log(masses[1]/masses[0])
    head=nn.Linear(z.shape[1],2,dtype=torch.float64)
    with torch.no_grad():
        head.weight.copy_(torch.stack([-.5*direction,.5*direction]))
        head.bias.copy_(torch.stack([-.5*intercept,.5*intercept]))
    folded=fold_standardization(head,mean,scale)
    with torch.no_grad():error=float((folded(features.double())-head(z)).abs().max())
    if error>1e-8:raise RuntimeError('LDA folding changed logits')
    return folded.float(),dict(algorithm='weighted_shrinkage_lda',shrinkage=shrinkage,
        normalization_mean=mean,normalization_scale=scale,fold_error_float64=error,
        selection='Fixed covariance shrinkage; all statistics from train only')


def json_solver_state(state):
    return {k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in state.items()}


def diagnose(model,datasets,device):
    result={}
    for subset,dataset in datasets.items():
        metrics,rows=run_epoch(model,DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0),device)
        videos,video_rows=aggregate_video_predictions(rows)
        result[subset]=dict(window_metrics=metrics,video_metrics=videos,window_predictions=rows,video_predictions=video_rows)
        print(json.dumps(dict(subset=subset,**videos)),flush=True)
    return result


def run(settings):
    settings=dict(settings)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=ROOT/'runs'/f'r3d18_regularized_{stamp}'
    report=ROOT/'reports'/f'evaluation_regularized_{stamp}'
    summary_path=ROOT/'reports'/f'regularized_study_{stamp}.json'
    checkpoint=(ROOT/settings['checkpoint']).resolve()
    prepared_eval=prepare_evaluation(checkpoint,device=settings['device'])
    cfg=dict(prepared_eval['config'],output_dir=str(output),freeze_backbone=True,epochs=1,
             cache_frozen_features=True,cpu_threads=settings['cpu_threads'],seed=settings['seed'])
    prepared=prepare_training(cfg);cfg=prepared['config']
    output.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    baseline=prepared_eval['model'];device=prepared['device']
    frozen=backbone_digest(baseline)
    plan=dict(schema='regularized_heads_plan_v1',settings=settings,
        baseline_checkpoint=str(checkpoint),baseline_sha256=sha256(checkpoint),backbone_sha256=frozen,
        events_sha256=prepared['manifest']['events_sha256'],split_sha256=prepared['manifest']['split_sha256'],
        candidates=['l2_logistic','shrinkage_lda'],threshold=.5,preprocessing='Identical to baseline',
        selection='Both candidates fixed before validation; no threshold/grid/epoch search.',
        limitation='Repeatedly inspected within-room development validation; not an independent test.')
    write_json(report/'plan.json',plan)
    print('PLAN_SAVED '+str(report/'plan.json'),flush=True)
    print('Caching frozen TRAIN features only; labels and variants inherit canonical split.',flush=True)
    start=time.perf_counter()
    cache=cache_frozen_loaders(baseline,{'train':prepared['loaders']['train']},device,cfg['batch_size'],cfg['seed'])['train'].dataset
    sources=prepared['datasets']['train'].source_manifest
    if len(cache)!=len(sources) or any(s['split']!='train' for s in sources):raise ValueError('Train cache coverage/partition mismatch')
    for value,source in zip(cache,sources):
        if (value['clip_id'],float(value['center_sec']),int(value['label']))!=(source['clip_id'],source['center_sec'],source['label']):
            raise ValueError('Cache/source order changed')
    features=torch.stack([r['features'] for r in cache]);labels=torch.tensor([int(r['label']) for r in cache])
    weights=torch.tensor([float(r['sample_weight']) for r in cache])
    probabilities=training_probabilities(sources,weights,settings['original_fraction'])
    torch.save(dict(features=features,labels=labels,probabilities=probabilities,plan=plan),output/'train_features.pt')
    write_json(report/'feature_bank.json',dict(windows=len(cache),feature_dimensions=features.shape[1],
        processing_sec=time.perf_counter()-start,source_manifest=sources,validation_cached=False,
        cache_sha256=sha256(output/'train_features.pt')))
    heads={
        'l2_logistic':fit_logistic(features,labels,probabilities,l2=settings['logistic_l2'],
            max_iter=settings['logistic_max_iter'],variance_floor=settings['variance_floor']),
        'shrinkage_lda':fit_shrinkage_lda(features,labels,probabilities,shrinkage=settings['lda_shrinkage'],
            variance_floor=settings['variance_floor'])}
    checkpoints={'baseline':checkpoint};checks={}
    for name,(head,solver) in heads.items():
        run_dir=output/name;run_dir.mkdir()
        baseline.fc=head.to(device).eval()
        if backbone_digest(baseline)!=frozen:raise RuntimeError('Frozen backbone or BatchNorm changed')
        run_config=dict(cfg,output_dir=str(run_dir),regularized_head=name,
            regularized_head_settings=settings,normalization_folded_into_fc=True,
            checkpoint_selection='fixed_final_step_no_validation',initial_backbone_checkpoint=str(checkpoint))
        payload=dict(format_version=1,architecture='r3d_18',class_to_idx=LABELS,epoch=1,
            model_state_dict=baseline.state_dict(),config=run_config,data_manifest=copy.deepcopy(prepared['manifest']),
            optimizer_state_dict={},optimizer_resume_supported=False,solver_state=solver,
            metrics={'protocol':'Fixed train-only regularized fitting; no validation selection'})
        torch.save(payload,run_dir/'last.pt')
        write_json(run_dir/'solver.json',json_solver_state(solver))
        (run_dir/'config.yaml').write_text(yaml.safe_dump(run_config,sort_keys=False),encoding='utf-8')
        write_json(run_dir/'data_manifest.json',prepared['manifest'])
        checks[name]=dict(checkpoint_sha256=sha256(run_dir/'last.pt'),backbone_before=frozen,
            backbone_after=backbone_digest(baseline),validation_used_for_fitting=False,threshold=.5,
            original_checkpoint_unchanged=sha256(checkpoint)==plan['baseline_sha256'])
        write_json(run_dir/'checks.json',checks[name]);checkpoints[name]=run_dir/'last.pt'
        print('SAVED '+str(run_dir/'last.pt'),flush=True)
    del cache,features,labels,weights,heads,baseline,prepared_eval
    datasets={s:VideoWindowDataset(prepared['records'],s,prepared['datasets']['validation'].transform,
        cfg['window_sec'],cfg['frames'],1,negative_sampling=cfg.get('negative_sampling','interior')) for s in ('train','validation')}
    diagnostics={};robustness={}
    # Every new model is already saved. Decode validation only for this final comparison.
    for name,path in checkpoints.items():
        print('EVALUATING '+name,flush=True)
        model,_=load_checkpoint(path,device)
        diagnostics[name]=diagnose(model,datasets,device)
        write_json(report/f'{name}_diagnostics.json',diagnostics[name]);del model
        robustness[name]=evaluate_robustness(path,report/name,device=str(device))
    for key,expected in (('events_path',plan['events_sha256']),('split_path',plan['split_sha256']),
                         ('supplemental_manifest',cfg['supplemental_manifest_sha256'])):
        if sha256(Path(cfg[key]))!=expected:raise RuntimeError('Dataset changed during experiment')
    for source in {s['path']:s for s in sources}.values():
        if sha256(Path(source['path']))!=source['sha256']:raise RuntimeError('Training video changed')
    if sha256(checkpoint)!=plan['baseline_sha256']:raise RuntimeError('Original checkpoint changed')
    summary=dict(schema='regularized_heads_study_v1',plan=plan,diagnostics=diagnostics,robustness=robustness,
        runs={name:str(path) for name,path in checkpoints.items()},checks=checks,
        active_model_replaced=False,independent_evaluation=False,
        environment=dict(torch=str(torch.__version__),device=str(device),cuda=torch.version.cuda))
    write_json(summary_path,summary);write_report(summary_path.with_suffix('.md'),summary)
    print('STUDY_SAVED '+str(summary_path),flush=True)
    return summary_path


def write_report(path,summary):
    lines=['# Регуляризованные головы на замороженных признаках R3D-18','',
        'Проверены два заранее заданных метода. Все параметры нормализации и классификаторов '
        'вычислены только по train. В validation не подбирались порог, регуляризация или число шагов. '
        'Split, исходные метки, кадры, ROI и backbone сохранены.','',
        'Основные метрики: окна вокруг известных отметок и одно центральное окно отрицательного видео '
        'одинаково в train и validation. Это не оценка сканирования неизвестного полного видео.','']
    for name,result in summary['diagnostics'].items():
        lines += [f'## {name}','']
        for subset,values in result.items():
            m=values['video_metrics']
            lines += [f"- {subset}: accuracy={m['accuracy']:.3f}, precision={m['precision']:.3f}, recall={m['recall']:.3f}, "
                f"F1={m['f1']:.3f}; confusion={m['confusion_matrix']}."]
        m=summary['robustness'][name]['transformed_video_metrics']
        lines += [f"- 40 преобразований validation: accuracy={m['accuracy']:.3f}, precision={m['precision']:.3f}, "
            f"recall={m['recall']:.3f}, F1={m['f1']:.3f}; confusion={m['confusion_matrix']}.",'']
        errors=[f"{r['clip_id']} ({r['outcome']})" for r in result['validation']['video_predictions'] if r['outcome'] in {'FP','FN'}]
        lines += ['Ошибки: '+(', '.join(errors) or 'нет на этой выборке')+'.','']
    base=summary['diagnostics']['baseline']['validation']['video_metrics']
    lines += ['## Сравнение с прежними весами','']
    for name in ('l2_logistic','shrinkage_lda'):
        values=summary['diagnostics'][name]
        train,validation=(values[s]['video_metrics'] for s in ('train','validation'))
        lines += [f"- {name}: validation accuracy {(validation['accuracy']-base['accuracy'])*100:+.1f} п.п.; "
            f"F1 {validation['f1']-base['f1']:+.3f}; разрыв train/validation accuracy "
            f"{(train['accuracy']-validation['accuracy'])*100:.1f} п.п."]
    improved=[name for name in ('l2_logistic','shrinkage_lda')
        if summary['diagnostics'][name]['validation']['video_metrics']['accuracy']>base['accuracy']
        and summary['diagnostics'][name]['validation']['video_metrics']['f1']>=base['f1']]
    lines += ['', ('На известных validation-оригиналах accuracy выросла без снижения F1: '+', '.join(improved)+'.'
        if improved else 'Прирост accuracy без ухудшения F1 на исходной validation в этом эксперименте не подтверждён.'),
        'Это итоговое сравнение заранее заданных кандидатов на development-выборке; '
        'оно не является независимой оценкой выбранного кандидата.','']
    lines += ['## Ограничения','',
        'Validation многократно просмотрена; один кабинет и неизвестные сеансы. '
        '40 преобразований — связанные версии 10 исходников. Независимого test нет. '
        'Ни одна метрика не является оценкой юридических типов. Прежний checkpoint '
        'сохранён; backend и .env автоматически не переключаются. Нормализация свёрнута '
        'в обычный fc, восстановление совместимо с существующим load_checkpoint.','',
        'Калибровка scores не проверялась; уверенность LDA/логистической головы не является '
        'вероятностью юридического нарушения. Порог остаётся 0.5, без подбора по validation.','',
        'Гиперпараметры и SHA-256: в одноимённом JSON-отчёте.']
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot_comparison(summary,output):
    """Numeric chart only; no video pixels or annotations rendered."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(summary['diagnostics'])
    width=.8/len(names)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for i,name in enumerate(names):
        d=summary['diagnostics'][name]
        accuracy=[d['train']['video_metrics']['accuracy'],d['validation']['video_metrics']['accuracy'],
                  summary['robustness'][name]['transformed_video_metrics']['accuracy']]
        axes[0].bar([x+(i-(len(names)-1)/2)*width for x in range(3)],[100*v for v in accuracy],width=width,label=name)
        val=d['validation']['video_metrics']
        axes[1].bar([x+(i-(len(names)-1)/2)*width for x in range(3)],[100*val[k] for k in ('precision','recall','f1')],width=width,label=name)
    axes[0].set_xticks(range(3),['Train originals (30)','Validation originals (10)','Correlated variants (40)'],rotation=12)
    axes[0].set(title='Accuracy by evaluation scope',ylabel='Accuracy (%)',xlabel='Video group')
    axes[1].set_xticks(range(3),['Precision','Recall','F1'])
    axes[1].set(title='Original validation videos (10)',ylabel='Metric (%)',xlabel='Binary classification metric')
    for ax in axes:
        ax.set_ylim(0,105);ax.legend(fontsize=8)
    fig.text(.5,.01,'Source: saved frozen-R3D studies, 18 Oct 2026; threshold 0.5; not an independent test.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(Path(output)/'comparison.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/r3d18_regularized.yaml')
    args=parser.parse_args()
    run(yaml.safe_load(args.config.read_text(encoding='utf-8')))


if __name__=='__main__':main()
