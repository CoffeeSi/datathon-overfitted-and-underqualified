"""Summarize immutable real action runs without selecting a validation threshold."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from overfitted_underqualified.ml.dataset import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError(args.output)
    results=[]
    for folder in args.evaluation:
        folder=folder.resolve()
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        predictions=json.loads((folder/'window_predictions.json').read_text(encoding='utf-8'))
        action=report['provenance']['action']
        run=Path(action['checkpoint_path']).parent
        config=json.loads((run/'config.json').read_text(encoding='utf-8'))
        history=json.loads((run/'history.json').read_text(encoding='utf-8'))
        errors=[]
        for row in predictions:
            for index,name in enumerate(action['class_names']):
                if not row['mask'][index]:continue
                actual=bool(row['targets'][index]); predicted=row['probabilities'][index]>=action['threshold']
                if actual!=predicted:
                    errors.append(dict(clip_id=row['clip_id'],center_sec=row['center_sec'],event_type=name,
                        kind='FN' if actual else 'FP',probability=row['probabilities'][index]))
        results.append(dict(run=run.relative_to(ROOT).as_posix(),checkpoint_sha256=sha256(run/'last.pt'),
            config_sha256=sha256(run/'config.json'),evaluation=folder.relative_to(ROOT).as_posix(),
            positive_weighting=config.get('positive_weighting','none'),epochs=config['epochs'],
            train_loss=[h['train']['loss'] for h in history],validation_loss=[h['validation']['loss'] for h in history],
            window_metrics=report['window_metrics'],window_errors=errors,temporal=report['temporal'],
            full_video_processing_sec=report['full_video_processing_sec'],candidate_windows=report['candidate_windows'],
            typed_windows=report['typed_windows'],recognition_coverage=report['recognition_coverage']))
    payload=dict(protocol='exploratory_within_room',independent_evaluation=False,
                 events_sha256=sha256(ROOT/'data/annotations/events.csv'),
                 split_sha256=sha256(ROOT/'data/processed/split_proposal.csv'),
                 catalog_sha256=sha256(ROOT/'src/overfitted_underqualified/rules/catalog.json'),
                 runs=results,default_deployment_enabled=False)
    args.output.with_suffix('.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# Action-модель: обучение и полный инференс, 18 октября 2026\n',
        'Два реальных запуска по пять эпох на CUDA. 31 подтверждённое окно: 24 train и 7 validation. '
        'Обучались восемь типов; только client_identity_check_not_visible имеет положительный и отрицательные validation-примеры. '
        'Подтверждение владельца записано в action_review того же events.csv. Неоднозначные строки masked, '
        'все прежние поля, времена и split сохранены. Никакое отсутствие не распространено на полное видео.\n',
        'Первый запуск использует обычную masked BCE. Второй — балансировку positive loss отношением '
        'известных отрицательных к положительным train-примерам; validation для расчёта весов не используется. '
        'Порог 0.7, фон 0.2, пять эпох и временной допуск 5 секунд зафиксированы; порог не подбирался. '
        'Второй запуск — исследовательская итерация после первого; независимого теста нет.\n']
    for r in results:
        lines += ['## '+r['positive_weighting']+'\n',
            f"Checkpoint: `{r['run']}/last.pt`; SHA-256 `{r['checkpoint_sha256']}`.\n",
            f"Отчёт и ZIP: `{r['evaluation']}` / `{r['evaluation']}.zip`.\n"]
        for item in r['window_metrics']['per_class']:
            if item['evaluable']:
                lines.append(f"{item['event_type']}: precision={item['precision']:.3f}, recall={item['recall']:.3f}, F1={item['f1']:.3f}; "
                    f"confusion [[TN,FP],[FN,TP]] = {item['confusion_matrix']}; support={item['support']}.\n")
        lines.append(f"Полные 10 validation-видео: {r['full_video_processing_sec']:.2f} с обработки на CUDA; "
            f"бинарный фильтр пропустил {r['candidate_windows']} окон, тип выдан в {r['typed_windows']}. "
            "Это wall time данного запуска, а не нормированный benchmark.\n")
        for error in r['window_errors']:
            lines.append(f"Ошибка {error['kind']}: {error['clip_id']} @ {error['center_sec']:.2f} с, "
                         f"{error['event_type']}, score={error['probability']:.4f}.\n")
        for item in r['temporal']['per_class']:
            if item['support']:
                lines.append(f"Полный инференс {item['event_type']}: matched={item['matched']}, missed={item['missed']}, "
                    f"binary gate coverage={item['binary_gate_coverage']}, MAE={item['matched_timestamp_mae_sec']}.\n")
    lines += ['## Вывод и ограничения\n',
        'Обучение и восстановление реальных весов выполнены. 120 тензоров backbone, включая BatchNorm, '
        'сверены с исходными pretrained-весами и не изменились. Тесты проверяют отсутствие обновления на validation, '
        'несколько событий, пустой результат, неизвестные типы и привязку правил. Синтетические проверки '
        'не доказывают качество распознавания банковских событий.\n',
        'Измеренное качество недостаточно: единственный оцениваемый положительный пример пропущен. '
        'Другие семь train-типов не имеют положительных validation-примеров. Ещё один тип есть только '
        'в validation и считается пропуском, а не используется для обучения. '
        'Пропуски и binary gate coverage разделены в JSON. При отсутствии совпадений MAE=null. '
        'Лишние тревоги на неполной временной разметке не объявляются подтверждёнными FP.\n',
        'Общий кабинет, люди и неизвестные сеансы могут пересекать выборки. Разметка подтверждена '
        'владельцем пакетно, не прошла независимую проверку. В отчёты и ZIP не включены кадры или веса.\n',
        'В основном сервисе enabled_types остаётся пустым. Для воспроизведения используется '
        'configs/action_service_research.yaml; это экспериментальный режим с известными пропусками. '
        'Для закрытия P0 по распознаванию нужны дополнительные положительные видео в обеих выборках '
        'и разбор X-2/контекста идентификации: одна обучающая точка не подтверждает обобщение.\n']
    args.output.write_text('\n'.join(lines),encoding='utf-8')
    print(args.output)


if __name__=='__main__':main()
