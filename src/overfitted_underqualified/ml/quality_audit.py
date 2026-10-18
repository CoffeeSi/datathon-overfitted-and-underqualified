"""Read-only coverage/provenance audit and data-collection queue; no model fitting."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .action_data import reviewed_samples
from .dataset import load_records, sha256
from ..rules.catalog import CATALOG_PATH, load_catalog, supported_events
from ..rules.evidence import VISUAL_REQUIREMENTS, grounded_rule_match, proposed_mapping

ROOT = Path(__file__).resolve().parents[3]


def assess_holdout_candidate(candidate, *, used_hashes, used_source_groups):
    """Check future holdout eligibility; never assign or change a partition.

    Group provenance must be reviewed: a different hash or filename does not
    establish a new session. Labels continue to belong to events.csv.
    """
    reasons=[]
    digest=candidate.get('sha256','')
    group=candidate.get('source_group')
    if not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest):reasons.append('missing_valid_sha256')
    if digest in used_hashes:reasons.append('already_used_video_bytes')
    if candidate.get('origin')!='real_recording':reasons.append('not_confirmed_real_recording')
    if candidate.get('derivative_of') is not None:reasons.append('derived_from_existing_source')
    if not group or candidate.get('source_group_verified') is not True:reasons.append('unverified_source_group')
    if group in used_source_groups:reasons.append('already_used_source_group')
    if candidate.get('previously_used') is not False:reasons.append('prior_use_unknown_or_true')
    if candidate.get('labels_reviewed') is not True:reasons.append('unreviewed_labels')
    return dict(eligible=not reasons,reasons=reasons,source_group=group,
                label_source='data/annotations/events.csv',assigned_partition=None)


def review_basis(review):
    """Report the declared provenance; do not turn batch labels into frame reviews."""
    note = review.get('evidence_note','').lower()
    if 'batch' in note or 'not a new frame-by-frame' in note or 'no independent frame-by-frame' in note:
        return 'batch_supplied_labels'
    if review.get('reviewer') == 'frame_sequence_review':
        return 'declared_frame_sequence_review'
    return 'other_declared_review'


def summarize_type(name, samples):
    entry = dict(event_type=name, required_visual_facts=list(VISUAL_REQUIREMENTS.get(name,())),
                 catalog_membership_is_model_support=False)
    for subset in ('train','validation'):
        selected = [s for s in samples if s.record.split==subset]
        positives = [s for s in selected if name in s.positives]
        negatives = [s for s in selected if name in s.negatives]
        whole = [s for s in selected if name in s.whole_video_absent_types]
        bases = Counter(review_basis(r) for s in positives for r in s.reviews if name in r['present_types'])
        entry[subset] = dict(positive_windows=len(positives),
            positive_ids=sorted({s.record.clip_id for s in positives}),
            positive_source_ids=sorted({a for s in positives for a in s.source_ids}),
            negative_windows=len(negatives), negative_window_ids=sorted({s.record.clip_id for s in negatives}),
            whole_video_negative_ids=sorted({s.record.clip_id for s in whole}),
            positive_review_basis=dict(bases),
            non_batch_positive_ids=sorted({s.record.clip_id for s in positives
                if any(review_basis(r)!='batch_supplied_labels' and name in r['present_types'] for r in s.reviews)}),
            points=[dict(clip_id=s.record.clip_id,source_ids=list(s.source_ids),center_sec=s.center_sec,
                target='present' if name in s.positives else 'absent' if name in s.negatives else 'unknown')
                for s in selected])
    tr,va=entry['train'],entry['validation']
    entry['can_fit_with_existing_declared_labels']=bool(tr['positive_ids'] and tr['negative_windows'])
    entry['window_validation_evaluable']=bool(va['positive_ids'] and va['negative_windows'])
    entry['has_whole_video_negative_support']=bool(va['whole_video_negative_ids'])
    entry['full_video_validation_evaluable']=False  # No complete positive-event ground truth.
    reasons=[]
    if not tr['positive_ids']:reasons.append('no_train_positive_video')
    elif len(tr['positive_ids'])<2:reasons.append('single_canonical_train_positive')
    if not tr['negative_windows']:reasons.append('no_reviewed_train_negative_window')
    if not va['positive_ids']:reasons.append('no_validation_positive_video')
    if not va['negative_windows']:reasons.append('no_reviewed_validation_negative_window')
    if not va['whole_video_negative_ids']:reasons.append('no_whole_video_negative_attestation')
    if len(tr['non_batch_positive_ids'])<len(tr['positive_ids']):reasons.append('train_positives_include_batch_labels')
    if len(va['non_batch_positive_ids'])<len(va['positive_ids']):reasons.append('validation_positives_include_batch_labels')
    entry['quality_limitations']=reasons
    entry['deployment_approved']=False
    # Collection goals are explicit planning targets, not statistical guarantees.
    entry['collection_goals']=dict(target_train_positive_videos=5,target_validation_positive_videos=2,
        additional_train_positive_videos=max(0,5-len(tr['positive_ids'])),
        additional_validation_positive_videos=max(0,2-len(va['positive_ids'])),
        evidence_required='New source videos with provenance and reviewed visible facts; not shifted windows or augmentations.',
        priority='missing_train_type' if not tr['positive_ids'] and va['positive_ids'] else
            'both_partitions_present_but_sparse' if tr['positive_ids'] and va['positive_ids'] else 'missing_partition_support')
    return entry


def build_audit(*,root=ROOT,synthetic_dir=None):
    root=Path(root)
    events=root/'data/annotations/events.csv';split=root/'data/processed/split_proposal.csv'
    pins=dict(events_sha256=sha256(events),split_sha256=sha256(split),catalog_sha256=sha256(CATALOG_PATH))
    records=load_records(events,split,root/'data/raw/videos')
    with events.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    samples=reviewed_samples(records,{r['clip_id']:r for r in rows},4.)
    by_alias={alias:r for r in records for alias in r.aliases}
    mappings=[]
    for row in rows:
        mapping=proposed_mapping(row['observed_event_type'])
        record=by_alias[row['clip_id']]
        mapping.update(clip_id=row['clip_id'],canonical_id=record.clip_id,split=record.split,
            source_label=row['source_label'],annotation_timecode=row['timecode'],
            legal_basis_is_action_target=False,
            evidence_check={name:grounded_rule_match(name,{}) for name in mapping['catalog_candidates']})
        if row.get('action_review'):
            review=json.loads(row['action_review'])
            mapping['declared_review_positive_types']=review['present_types']
            mapping['declared_review_basis']=review_basis(review)
        mappings.append(mapping)
    coverage=[summarize_type(e['event_type'],samples) for e in supported_events()]
    # Reuse the pinned supplemental audit, but verify that every referenced file
    # is still the same before claiming it is accounted for.
    audit_path=root/'data/raw/supplemental/audit.json'
    supplemental=json.loads(audit_path.read_text(encoding='utf-8'))
    for key,path in (('events',events),('split',split)):
        if supplemental['inputs'][key]['sha256']!=sha256(path):raise ValueError('Stale supplemental audit')
    available={};missing=[]
    for item in supplemental['augmentations']+supplemental['synthetic']:
        path=Path(item['path'])
        if not path.is_file() and item in supplemental['synthetic'] and synthetic_dir is not None:
            path=Path(synthetic_dir)/path.name
        if not path.is_file():
            missing.append(Path(item['path']).name)
            continue
        if sha256(path)!=item['sha256']:raise ValueError('Changed supplemental video')
        available[item['sha256']]=str(path.resolve())
    synthetic= supplemental['synthetic']
    supplemental_summary=dict(audit_sha256=sha256(audit_path),counts=supplemental['counts'],
        augmentation_files=len(supplemental['augmentations']),
        synthetic_files=len(synthetic),synthetic_parent_scenarios=len({r.get('source_id') for r in synthetic if r.get('source_id')}),
        missing_files=missing,verified_available_files=len(available),
        independent_test_eligible_files=0,
        reasons=['X-* descendants inherit the already used canonical partitions.',
                 'S01-S04 are synthetic scenarios; mirrors are not new source cases.',
                 'The unlabelled scenario has no approved action targets.'],
        synthetic_screening=[dict(filename=Path(r['path']).name,sha256=r['sha256'],source_id=r.get('source_id'),
            status=r['status'],origin=r.get('origin'),existing_training_eligible=r['training_eligible'],
            verified_available_path=available.get(r['sha256']),
            test_eligible=False,scenario_claim=r.get('source_claim',{}).get('Сценарий'),
            claim_is_confirmed_label=False) for r in synthetic])
    result=dict(schema='action_quality_coverage_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=pins,window_sec=4.,source_of_editable_labels='data/annotations/events.csv',
        protocol='Same fixed canonical split; explicit per-type window reviews; unknown stays unknown.',
        canonical_videos=len(records),source_ids=len(rows),
        partition_counts=dict(Counter(r.split for r in records)),reviewed_windows=len(samples),
        review_basis=dict(Counter(review_basis(v) for s in samples for v in s.reviews)),
        coverage=coverage,mappings=mappings,supplemental=supplemental_summary,
        independently_confirmed_quality=False,
        independent_test=dict(ready=False,eligible_real_video_ids=[],
            reason='No new real sessions available; current validation already inspected repeatedly.',
            preserve_current_split=True,required_before_intake=['source/recording provenance','session and canonical group',
                'original vs derivative relation','unseen real footage','reviewed labels and explicit unknowns'],
            do_not_relabel_existing_validation_as_test=True),
        next_actions=['Collect train positives for types present only in validation.',
            'Add distinct reviewed positives/near-miss negatives for sparse types, keeping relatives together.',
            'Review batch-labelled windows against actual footage; do not silently change existing labels.',
            'Use grounded_rule_match to bind only evidence-supported candidates to conditional rule references.',
            'Freeze a new real holdout before any future tuning; synthetic videos cannot supply it.'])
    if pins!=dict(events_sha256=sha256(events),split_sha256=sha256(split),catalog_sha256=sha256(CATALOG_PATH)):
        raise ValueError('Inputs changed while auditing')
    return result


def write_report(audit,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    queue=[dict(event_type=c['event_type'],train_positive_ids=c['train']['positive_ids'],
        validation_positive_ids=c['validation']['positive_ids'],quality_limitations=c['quality_limitations'],
        collection_goals=c['collection_goals'],required_visual_facts=c['required_visual_facts'],status='pending_new_sources_or_review')
        for c in audit['coverage']]
    (output/'collection_queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Покрытие action-типов и готовность проверки качества','',
        f"Источники: {audit['source_ids']} ID, {audit['canonical_videos']} канонических видео; "
        f"train/validation {audit['partition_counts']['train']}/{audit['partition_counts']['validation']}. Split не изменён.",'',
        'Новых реальных сеансов нет. Независимый test не создан и качество на нём не заявляется. '
        'Прежние метрики не пересчитаны и не заменены. Это аудит данных, а не обучение модели.','',
        'Перечисленные ниже положительные метки взяты из существующих action_review. '
        'Пакетные метки не объявляются независимым просмотром; отрицательное окно не доказывает '
        'отсутствие действия во всём видео. Каноническое видео считается один раз.','']
    for c in audit['coverage']:
        lines += [f"## {c['event_type']}",'',
            f"- Train положительные: {', '.join(c['train']['positive_ids']) or 'нет'}; "
            f"окон: {c['train']['positive_windows']}; отрицательных окон: {c['train']['negative_windows']}.",
            f"- Validation положительные: {', '.join(c['validation']['positive_ids']) or 'нет'}; "
            f"отрицательных окон: {c['validation']['negative_windows']}; "
            f"подтверждённых отрицательных видео целиком: {len(c['validation']['whole_video_negative_ids'])}.",
            '- Ограничения: '+', '.join(c['quality_limitations'])+'.',
            f"- План сбора: ещё {c['collection_goals']['additional_train_positive_videos']} train-положительных "
            f"и {c['collection_goals']['additional_validation_positive_videos']} validation-положительных исходных видео. "
            'Это ориентир для сбора, не обещание достаточной статистики.','']
    gaps=[m['clip_id'] for m in audit['mappings'] if m['status']=='needs_context_review']
    lines += ['## Каталог и визуальное основание','',
        'Неоднозначные теги требуют контекста: '+', '.join(gaps)+'.',
        'Ни один тег CSV сам по себе не подтверждает все визуальные условия. '
        'grounded_rule_match возвращает event_type и ссылки только при явных фактах '
        'и проверяемом происхождении доказательств. При неизвестном/ложном обязательном '
        'факте — event_type=null и source_links=[]. Даже разрешённая ссылка условна, '
        'legal_conclusion=null и requires_review=true.', '',
        'Текущую редакцию официальных текстов повторно подтвердить не удалось: '
        'страница Әділет требует JavaScript. Сохранённый каталог и его хеш не менялись; '
        '№14/№53 не становятся новыми распознаваемыми типами.','',
        '## Дополнительные данные','',
        f"Аугментаций: {audit['supplemental']['augmentation_files']}; синтетических файлов: "
        f"{audit['supplemental']['synthetic_files']}. Независимых test-кандидатов: 0. "
        'Синтетические сценарии остаются отдельным потенциальным вспомогательным источником '
        'после проверки, без автоматического импорта их заявлений как фактов.', '',
        '## Что остаётся открытым','',
        'Реальное добавление независимых исходных примеров и итоговый test требуют новых записей. '
        'Старый split не переразмечается ради получения метрики; модель с accuracy 60% не заменяется. '
        'Backend/frontend и конфигурация активной модели не изменялись.', '',
        'Хеши входов: `'+json.dumps(audit['inputs'],sort_keys=True)+'`.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return output/'report.md'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--synthetic-dir',type=Path,help='Verify relocated synthetic originals by pinned SHA-256; never imports their claims as labels')
    args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path=write_report(build_audit(synthetic_dir=args.synthetic_dir),args.output_dir or ROOT/'reports'/f'quality_coverage_{stamp}')
    print(path)


if __name__=='__main__':main()
