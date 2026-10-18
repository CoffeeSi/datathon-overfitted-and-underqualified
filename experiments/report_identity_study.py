"""Report a fixed matched geometry comparison; never selects an epoch or threshold."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from overfitted_underqualified.ml.dataset import sha256,load_records,decode_frames,sample_times
from overfitted_underqualified.ml.action_train import load_action_checkpoint,write_json
from overfitted_underqualified.ml.preprocessing import video_transform
NAME='client_identity_check_not_visible'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--control-evaluation',type=Path,required=True)
    parser.add_argument('--roi-evaluation',type=Path,required=True)
    parser.add_argument('--diagnostics',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError(args.output)
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    diagnostic=json.loads((args.diagnostics/'diagnostics.json').read_text(encoding='utf-8'))
    assert sha256(ROOT/'data/annotations/events.csv')==plan['events_sha256']
    assert sha256(ROOT/'data/processed/split_proposal.csv')==plan['split_sha256']
    torch.set_num_threads(4)
    records=load_records(ROOT/'data/annotations/events.csv',ROOT/'data/processed/split_proposal.csv',ROOT/'data/raw/videos')
    train_record=next(r for r in records if r.clip_id=='X-18')
    pretrained=torch.load(ROOT/'models/pretrained/checkpoints/r3d_18-b3b3357e.pth',map_location='cpu',weights_only=True)
    results=[]
    for variant,folder in zip(plan['variants'],(args.control_evaluation,args.roi_evaluation)):
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        provenance=report['provenance']['action'];run=Path(provenance['checkpoint_path']).parent
        model,checkpoint,digest=load_action_checkpoint(run/'last.pt','cpu')
        config=checkpoint['config'];history=checkpoint['history']
        # Executed paths are absolute, planned paths portable; compare all substantive settings.
        for key,value in variant['config'].items():
            if key not in ('events_path','split_path','video_dir','output_dir','weights_cache'):assert config[key]==value,key
        keys=[k for k in checkpoint['model_state_dict'] if not k.startswith('fc.')]
        assert all(torch.equal(checkpoint['model_state_dict'][k],pretrained[k]) for k in keys)
        predictions=json.loads((folder/'window_predictions.json').read_text(encoding='utf-8'))
        metrics=next(c for c in report['window_metrics']['per_class'] if c['event_type']==NAME)
        with torch.inference_mode():
            stamps=sample_times(train_record.event_times[0],train_record.last_time,config['window_sec'],16)
            tensor=video_transform(config)(decode_frames(train_record,stamps))
            train_score=model(tensor.unsqueeze(0)).sigmoid()[0,0].item()
        results.append(dict(name=variant['name'],run=run.relative_to(ROOT).as_posix(),checkpoint_sha256=digest,
            evaluation=folder.resolve().relative_to(ROOT).as_posix(),epochs=config['epochs'],
            configuration=config,train=history[-1]['train']['per_class'][0],validation=metrics,
            x2_score=next(r['probabilities'][0] for r in predictions if r['clip_id']=='X-2'),
            final_x18_score=train_score,
            temporal=next(c for c in report['temporal']['per_class'] if c['event_type']==NAME),
            typed_windows=report['typed_windows'],candidate_windows=report['candidate_windows'],
            full_video_processing_sec=report['full_video_processing_sec'],
            frozen_tensors_verified=len(keys),history=history))
        del model,checkpoint,tensor
    payload=dict(schema='identity_geometry_results_v1',plan=str(args.plan),diagnostics=diagnostic,
                 results=results,annotation_and_split_unchanged=True,
                 object_clarification='X-2 blue rectangular item is a banknote, per dataset owner.',
                 independent_evaluation=False,deployment_changed=False,
                 additional_independent_positive_training_videos=0)
    write_json(args.output.with_suffix('.json'),payload)
    figures=ROOT/'reports/figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for result in results:
        h=result['history'];epochs=[e['epoch'] for e in h]
        axes[0].plot(epochs,[e['train']['loss'] for e in h],label=result['name'])
        axes[1].plot(epochs,[e['validation']['loss'] for e in h],label=result['name'])
    for ax,title in zip(axes,('Train weighted BCE','Validation unweighted BCE')):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/'identity_geometry_learning.png',dpi=160);plt.close(fig)
    lines=['# Разбор X-18 / X-2 и эксперимент с геометрией\n',
        'Разметка и split не менялись. X-18 остаётся в train, X-2 — в validation. '
        'Стол, руки и лица проверены по полному обзору обоих роликов с шагом 1 секунда '
        'и исходным кадрам у отметок. Такой обзор не равен независимой проверке всех действий. '
        'В X-2 голубой предмет — банкнота по уточнению владельца, не удостоверение личности.\n',
        'Источник 464×848 принудительно приводился к 171×128 и затем обрезался до 112×112. '
        'Это искажает пропорции и тратит часть входа на переднюю перегородку и пол. '
        'В train-окне X-18 присутствует ещё один человек: это отличие сцены от X-2, '
        'которое может стать ложным признаком. Его фактическое использование сетью не доказано.\n',
        'Диагностика прежних весов при контексте 4/8/12 с дала для X-2 '
        '0.2471/0.2481/0.2571. Это чувствительность к входу, не улучшенное качество новой модели. '
        'Для обучения сохранены 4 секунды: отрицательные метки не распространялись на более длинный участок.\n',
        'До оценки зафиксировано сравнение: один тип, train-only positive weight=23, '
        '30 эпох, seed=42, batch=8, lr=0.001, порог=0.7, frozen backbone. '
        'Единственное различие вариантов — геометрия. ROI=[0,0.1,1,0.65] выбран по train-кадрам; '
        'он сохраняет пропорции и область клиента/кассира. Frozen-признаки вычислены один раз '
        'внутри каждого запуска; backbone не обучается, признаки validation используются только для оценки.\n']
    for r in results:
        v=r['validation'];t=r['temporal']
        lines += ['## '+r['name']+'\n',
            f"Веса `{r['run']}/last.pt`, SHA-256 `{r['checkpoint_sha256']}`.\n",
            f"Train F1={r['train']['f1']:.3f}; validation precision={v['precision']:.3f}, recall={v['recall']:.3f}, F1={v['f1']:.3f}. "
            f"Матрица [[TN,FP],[FN,TP]]={v['confusion_matrix']}; final X-18 score={r['final_x18_score']:.4f}, X-2 score={r['x2_score']:.4f}.\n",
            f"Полные 10 validation-видео: matched={t['matched']}, missed={t['missed']}, "
            f"binary gate coverage={t['binary_gate_coverage']}; typed windows={r['typed_windows']}/{r['candidate_windows']}. "
            f"Время {r['full_video_processing_sec']:.2f} с, исследовательский замер на этом устройстве.\n",
            f"Полный JSON и ZIP: `{r['evaluation']}` и `{r['evaluation']}.zip`. "
            f"С исходными pretrained-весами сверены {r['frozen_tensors_verified']} тензоров backbone, включая BatchNorm.\n"]
    improved=results[1]['validation']['f1']>results[0]['validation']['f1']
    lines += ['## Вывод\n',
        ('ROI повысил F1 на этой маленькой проверке.' if improved else 'ROI исправил геометрию входа, но прирост validation F1 не подтверждён.')+' '
        'Оба результата и фиксированные последние эпохи сохранены; порог не подбирался. '
        'Train F1=1 при пропуске X-2 указывает на отсутствие продемонстрированного переноса '
        'с единственного положительного train-видео, а не просто на незавершённую оптимизацию.\n',
        'Новых независимых подтверждённых положительных видео не появилось. Повторные кадры, '
        'сдвиги окна X-18 и X-2 не объявлялись дополнительными обучающими видео. '
        'Следующий источник улучшения — дополнительные записи приёма наличных с понятным '
        'контекстом идентификации и отрицательные примеры с видимой проверкой, '
        'а также проверка полноты существующих action-отсутствий.\n',
        'Validation уже использовалась для диагностики; кабинет и люди пересекаются. '
        'Это исследовательское сравнение, не независимый тест. Основной сервис не переключён. '
        'Кадры лежат только в ignored data/local_artifacts/; отчёты/ZIP не содержат кадров.\n',
        'Графики loss: `reports/figures/identity_geometry_learning.png`.\n']
    args.output.write_text('\n'.join(lines),encoding='utf-8');print(args.output)


if __name__=='__main__':main()
