"""Summarize the preregistered cash-storage experiment, including failed acceptance."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from overfitted_underqualified.ml.dataset import sha256
from overfitted_underqualified.ml.action_train import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError(args.output)
    plan=json.loads((ROOT/'reports/unsecured_cash_20261018_plan.json').read_text(encoding='utf-8'))
    folder=args.evaluation.resolve();report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    predictions=json.loads((folder/'window_predictions.json').read_text(encoding='utf-8'))
    name='unsecured_accepted_cash';item=next(c for c in report['window_metrics']['per_class'] if c['event_type']==name)
    temporal=next(c for c in report['temporal']['per_class'] if c['event_type']==name)
    known=[dict(clip_id=r['clip_id'],center_sec=r['center_sec'],target=int(r['targets'][0]),
                score=r['probabilities'][0],predicted=int(r['probabilities'][0]>=.7)) for r in predictions if r['mask'][0]]
    positive=next(r for r in known if r['target']);negatives=[r for r in known if not r['target']]
    run=Path(report['provenance']['action']['checkpoint_path']).parent
    assert report['provenance']['data_manifest']['events_sha256']==plan['events_sha256']
    assert sha256(ROOT/'data/processed/split_proposal.csv')==plan['split_sha256']
    result=dict(schema='unsecured_cash_quality_v1',plan='reports/unsecured_cash_20261018_plan.json',
        checkpoint=run.relative_to(ROOT).as_posix()+'/last.pt',checkpoint_sha256=sha256(run/'last.pt'),
        config_sha256=sha256(run/'config.json'),events_sha256=plan['events_sha256'],split_sha256=plan['split_sha256'],
        catalog_sha256=plan['catalog_sha256'],evaluation=folder.relative_to(ROOT).as_posix(),
        window_metrics=item,known_windows=known,temporal=temporal,
        full_video_processing_sec=report['full_video_processing_sec'],typed_windows=report['typed_windows'],
        candidate_windows=report['candidate_windows'],
        useful_window_recognition_demonstrated=bool(item['confusion_matrix'][1][1] and item['confusion_matrix'][0][0]),
        independent_evaluation=False,deployment_enabled=False,
        score_ordering_separates_all_known_targets=positive['score']>max(r['score'] for r in negatives))
    scans=json.loads((folder/'scans.json').read_text(encoding='utf-8'))
    alarms=[dict(clip_id=clip_id,timestamp_sec=e['timestamp_sec'],confidence=e['type_confidence'])
            for clip_id,scan in scans.items() for e in scan['typed_events'] if e['event_type']==name]
    result['typed_alarms']=alarms
    # The observed X-2 alarm is inspected after prediction for error analysis only;
    # it is not added to train, threshold selection, or the predefined four windows.
    result['posthoc_error_review']=[dict(**a,status='false_alarm',
        reason='Source-resolution X-2 sequence: at 2-6s the desk has no banknotes, the first customer is only entering. No preceding cash receipt is visible.')
        for a in alarms if a['clip_id']=='X-2' and abs(a['timestamp_sec']-4.)<1e-6]
    write_json(args.output.with_suffix('.json'),result)
    lines=['# Проверка unsecured_accepted_cash на positive/negative validation\n',
        'Тип выбран по наблюдаемому контексту, до получения scores нового эксперимента. '
        'X-38 остаётся в validation: получение денег, уход первого клиента и оставшаяся на столе '
        'банкнота при входе следующего. В action_review добавлено present только для этого типа; '
        'другие действия X-38 остаются неизвестными. Это наблюдение, не доказательство правового нарушения.\n',
        'Три проверенных отрицательных окна: X-2 (текущий взнос первого клиента), '
        'X-31 (на столе нет банкнот), X-39 (предыдущие деньги убраны, стол свободен). '
        'В X-10/X-14/X-16/X-34 отсутствие этого типа не установлено: прежние предположительные '
        'отрицательные метки заменены unknown. X-44 остаётся неизвестным: происхождение денег не видно. '
        'Оригинальные бинарные метки, описания, правовые ссылки, таймстампы и split сохранены.\n',
        'Train: две точки X-4/X-47 одного канонического видео и 22 отрицательных окна. '
        'Validation: один положительный и три отрицательных окна; четыре неизвестных исключены из метрик. '
        'Это не четыре независимых сеанса: общий кабинет и люди пересекаются.\n',
        'Фиксированы 30 эпох, seed=42, порог=0.7, train-only балансировка loss и ROI рабочего стола '
        '[0,0.30,1,0.58] с сохранением пропорций. Frozen-признаки кешируются внутри запуска; '
        'validation не обновляет веса, порог или выбор эпохи. Все настройки записаны в плане до оценки.\n',
        '## Результаты\n',
        f"Precision={item['precision']:.3f}, Recall={item['recall']:.3f}, F1={item['f1']:.3f}. "
        f"Матрица [[TN,FP],[FN,TP]]={item['confusion_matrix']}; unknown={item['unknown']}.\n"]
    for row in known:
        lines.append(f"- {row['clip_id']} @ {row['center_sec']:.2f}s: target={row['target']}, score={row['score']:.4f}, prediction={row['predicted']}.")
    lines += ['\nПолные 10 validation-видео: '
        f"matched={temporal['matched']}, missed={temporal['missed']}, binary gate coverage={temporal['binary_gate_coverage']}, "
        f"timestamp MAE={temporal['matched_timestamp_mae_sec']}; typed windows={report['typed_windows']}/{report['candidate_windows']}. "
        f"Время {report['full_video_processing_sec']:.2f}s на устройстве {report['device']}.\n",
        'Полная временная разметка отсутствует. Лишние предсказания не считаются автоматически '
        'подтверждёнными FP, отсутствие всего типа во всём отрицательном ролике не заявлено. '
        'Полные video-level precision/F1 поэтому не подменяются метриками четырёх известных окон.\n',
        '## Приёмка\n']
    for alarm in result['posthoc_error_review']:
        lines.append(f"Визуальный разбор единственной тревоги: X-2 @ {alarm['timestamp_sec']:.2f}s, confidence={alarm['confidence']:.4f}. "
                     'На столе ещё нет банкнот, первый клиент только входит. Это ложное срабатывание. '
                     'Такой разбор не меняет исходные метрики фиксированных окон и не используется для дообучения.\n')
    if result['useful_window_recognition_demonstrated']:
        lines.append('Есть правильно распознанный положительный и отрицательный примеры в пределах этого малого исследовательского протокола. Независимое качество не установлено.\n')
    else:
        lines.append('Измерение выполнено, но полезное распознавание на известных окнах не подтверждено: положительный X-38 пропущен. Положительную приёмку P0 объявлять нельзя.\n')
    if not result['score_ordering_separates_all_known_targets']:
        lines.append('Score отрицательного X-2 выше положительного X-38. Одним снижением порога нельзя получить правильное разделение всех известных случаев. Порог не изменён.\n')
    lines += ['Следующее ограничение — только одно независимое положительное train-видео и неоднозначность условий хранения. '
        'Нужны дополнительные видео с прослеживаемым приёмом/хранением денег и разные отрицательные '
        'операции. Смежные окна того же ролика не объявляются дополнительными независимыми примерами. '
        'Основной сервис не переключён. Предыдущие результаты сохранены как исторические.\n',
        f"Веса: `{result['checkpoint']}`; SHA-256 `{result['checkpoint_sha256']}`.\n",
        f"Полный отчёт/ZIP: `{result['evaluation']}` / `{result['evaluation']}.zip`. Кадры находятся только в ignored data/local_artifacts/.\n"]
    args.output.write_text('\n'.join(lines),encoding='utf-8');print(args.output)


if __name__=='__main__':main()
