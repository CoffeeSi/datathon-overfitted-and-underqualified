"""One-off, explicit legal_basis-only revision; preserve historical ML provenance."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from overfitted_underqualified.ml.dataset import sha256, annotation_fields_digest as preserved_digest


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('proposed',type=Path);args=parser.parse_args()
    target=ROOT/'data/annotations/events.csv'
    def read(path):
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream);return reader.fieldnames,list(reader)
    fields,old=read(target);new_fields,new=read(args.proposed)
    if fields!=new_fields or [r['clip_id'] for r in old]!=[r['clip_id'] for r in new]:raise ValueError('Schema or IDs/order differ')
    protected=[k for k in fields if k!='legal_basis']
    if preserved_digest(old,protected)!=preserved_digest(new,protected):raise ValueError('Non-legal annotation changes')
    if any(r['legal_basis'] and not r['legal_basis'].startswith('№231,') for r in new):raise ValueError('Only rule 231 references permitted')
    before=sha256(target);after=sha256(args.proposed)
    if before==after:raise ValueError('Already applied')
    snapshot=ROOT/'data/local_artifacts/annotation_snapshots'/before/'events.csv';snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists() and sha256(snapshot)!=before:raise ValueError('Snapshot conflict')
    snapshot.write_bytes(target.read_bytes())
    changes=[dict(clip_id=a['clip_id'],before=a['legal_basis'],after=b['legal_basis']) for a,b in zip(old,new) if a['legal_basis']!=b['legal_basis']]
    revision=dict(schema='legal_basis_only_revision_v1',before_sha256=before,after_sha256=after,
                  allowed_changed_fields=['legal_basis'],preserved_fields=protected,
                  preserved_fields_sha256=preserved_digest(new,protected),created_utc=datetime.now(timezone.utc).isoformat(),
                  reason='Owner requested the supplied 231-only annotations; labels, timestamps and observations unchanged.',
                  legal_verification='Not newly verified; official Adilet text requires JavaScript. Applicability remains conditional.',changes=changes)
    registry=ROOT/'data/annotation_revisions.json'
    if registry.exists():raise FileExistsError('Review existing revision registry before another migration')
    registry.write_text(json.dumps([revision],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    target.write_bytes(args.proposed.read_bytes())
    config=ROOT/'configs/actions.yaml';config.write_text(config.read_text(encoding='utf-8').replace(before,after),encoding='utf-8')
    catalog_path=ROOT/'src/overfitted_underqualified/rules/catalog.json'
    catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    catalog['version']='1.2';catalog['legal_scope']='nb_231_only'
    catalog['scope_revision']='Owner scope restriction; no new legal verification claimed.'
    catalog['sources']=[s for s in catalog['sources'] if s['id']=='nb_231']
    catalog['provisions']=[p for p in catalog['provisions'] if p['source_id']=='nb_231']
    for event in catalog['events']:
        event['provision_ids']=[p for p in event['provision_ids'] if p.startswith('nb_231_')]
        if event['event_type']=='cash_bag_received_without_visible_identity_authority_check':
            event['visible_action']='Принята сумка с наличными; в кадре не видно визуальной идентификации передавшего лица.'
            event['applicability']=['Подтвердить, что операция и передающее лицо подпадают под №231, п. 19 пп. 1.',
                                    'Отсутствие видимой идентификации не означает, что проверка не проводилась вне кадра.']
            event['video_cannot_establish'].append('Идентификатор типа сохранён для совместимости; вывод о проверке полномочий по одному п. 19 пп. 1 не делается.')
    by_id={r['clip_id']:r for r in new}
    for item in catalog.get('excluded',[]):
        if 'cited_in_events_csv' in item:item['cited_in_events_csv']=by_id[item['clip_id']]['legal_basis'] if 'clip_id' in item else ''
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    snapshot_doc=ROOT/'docs/video-labels-split.md';text=snapshot_doc.read_text(encoding='utf-8');prefix=text.split('| `clip_id`')[0]
    def cell(s):return (s or '—').replace('|','\\|').replace('\n',' ')
    table='| '+' | '.join('`'+f+'`' for f in fields)+' |\n| '+' | '.join('---' for _ in fields)+' |\n'
    table+=''.join('| '+' | '.join(cell(r[f]) for f in fields)+' |\n' for r in new)
    snapshot_doc.write_text(prefix+table,encoding='utf-8')
    print(json.dumps(dict(before=before,after=after,changes=len(changes),snapshot=str(snapshot)),ensure_ascii=False))


if __name__=='__main__':main()
