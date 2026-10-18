"""Create private visual evidence and a source-grounded action/catalog review queue.

No labels are changed, no review is auto-approved, no model predictions are used.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from overfitted_underqualified.ml.dataset import load_records, decode_frames, sha256, parse_time
from overfitted_underqualified.ml.train import ROOT
from overfitted_underqualified.ml.action_review import SEMANTIC_GAPS
from overfitted_underqualified.rules.catalog import CATALOG_PATH, load_catalog, source_links_for


def write_review_page(report, catalog, by_alias, out):
    for row in report["rows"]:
        record = by_alias[row["clip_id"]]
        row["original_video"] = record.path.resolve().as_uri()
        row["review_center_sec"] = row["timecode_sec"] if row["timecode_sec"] is not None else record.last_time/2
        row["semantic_gap"] = SEMANTIC_GAPS.get(row["observed_event_type"])
    payload = json.dumps(dict(report=report, types=[{**e, "source_links": source_links_for(e["event_type"])}
                                                  for e in catalog["events"] if e.get("api_supported")]),
                         ensure_ascii=False).replace("<", "\\u003c")
    html = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка типов действий</title>
<style>body{font:16px/1.5 system-ui,sans-serif;margin:24px auto;max-width:1200px;padding:0 20px;background:#f5f4f0;color:#202c32}header{position:sticky;top:0;background:#f5f4f0;padding:12px 0;border-bottom:1px solid #aaa;z-index:1}button,input,select,textarea{font:inherit;padding:8px}button{cursor:pointer}img{width:100%;height:auto}video{max-height:400px;max-width:100%}article{padding:16px;background:white;margin:10px 0;border:1px solid #ccc}label{display:block;font-weight:600}textarea{width:95%;min-height:65px}.warning{color:#8c300c;font-weight:600}small{display:block}#count{margin-left:15px}h2{margin-bottom:4px}a{color:#064c9b}</style>
<h1>Проверка типов действий</h1><p>Просмотрите оригинал. Кадры ниже помогают найти момент, но не подтверждают отсутствие действия во всём ролике.
Оценивайте условия типа, включая принадлежность денег, завершение операции и роль человека. Если условие неизвестно, оставьте «Неизвестно».</p>
<p>«Нет в окне» относится к 4 секундам вокруг указанного центра с учётом границ видео.
«Нет во всём видео» выбирайте только после просмотра всего оригинала. Это дополнительно подтверждает отсутствие в окне.
Ответы не предзаполнены по бинарным меткам. Каталог содержит условные связи с правилами, а не доказательства юридического нарушения.</p>
<header><label for="reviewer">Кто проверил</label><input id="reviewer" autocomplete="off" placeholder="Имя или идентификатор проверяющего">
<button id="prev">Назад</button><select id="clip" aria-label="Ролик"></select><button id="next">Далее</button>
<button id="export">Скачать решения</button><span id="count" role="status"></span></header>
<main><h2 id="heading"></h2><p id="source"></p><p id="gap" class="warning"></p>
<a id="original" target="_blank">Открыть оригинал отдельно</a><p>Если MOV не воспроизводится в браузере, откройте исходный файл в видеоплеере.</p>
<video id="video" controls preload="none"></video><img id="sheet" alt="Кадры всего ролика и контекста отметки">
<label for="note">Что видно и почему выбран тип или его отсутствие</label><textarea id="note"></textarea><section id="types"></section></main>
<p>После скачивания решения нужно проверить командой импорта. Только явный импорт с --apply обновляет events.csv.
Закрытие страницы удаляет несохранённые ответы. Скачайте JSON перед закрытием; можно загрузить его обратно:</p><input id="restore" type="file" accept="application/json">
<script>const DATA=__PAYLOAD__;
const report=DATA.report, rows=report.rows, drafts={}; let index=0;
const $=id=>document.getElementById(id);
function save(){const states={}; document.querySelectorAll('[data-type]').forEach(s=>{if(s.value)states[s.dataset.type]=s.value});
  drafts[rows[index].clip_id]={states,note:$('note').value};
  $('count').textContent=Object.values(drafts).filter(d=>Object.keys(d.states).length).length+' / '+rows.length+' роликов с решениями';}
function show(i){index=Math.max(0,Math.min(rows.length-1,i));const r=rows[index],d=drafts[r.clip_id]||{states:{},note:''};
  $('clip').value=index; $('heading').textContent=r.clip_id+' · '+r.split+' · центр окна '+r.review_center_sec.toFixed(2)+' с';
  $('source').textContent=r.observed_event+' ['+r.observed_event_type+']'; $('gap').textContent=r.semantic_gap||'';
  $('original').href=r.original_video;$('video').src=r.original_video;$('sheet').src=r.sheet;$('note').value=d.note;
  $('types').replaceChildren();
  for(const t of DATA.types){const box=document.createElement('article'), label=document.createElement('label'),sel=document.createElement('select');
    sel.dataset.type=t.event_type;sel.id='type-'+t.event_type;label.htmlFor=sel.id;
    label.textContent=t.event_type+(r.candidate_types.includes(t.event_type)?' (кандидат из каталога)':'');box.append(label);
    for(const [v,n] of [['','Неизвестно'],['present','Есть в окне'],['absent','Нет в окне'],['whole','Нет во всём видео']]){const o=document.createElement('option');o.value=v;o.textContent=n;sel.append(o)}
    sel.value=d.states[t.event_type]||'';sel.onchange=save;box.append(sel);
    for(const text of [t.visible_action, 'Условия: '+JSON.stringify(t.applicability), 'Нельзя установить: '+JSON.stringify(t.video_cannot_establish)]){const p=document.createElement('p');p.textContent=text;box.append(p)}
    for(const link of t.source_links){const a=document.createElement('a');a.href=link.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=link.source+', '+link.locator;const p=document.createElement('p');p.append(a);box.append(p)}
    $('types').append(box)} }
rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.clip_id+' · '+r.split;$('clip').append(o)});
$('clip').onchange=e=>{save();show(Number(e.target.value))};$('prev').onclick=()=>{save();show(index-1)};$('next').onclick=()=>{save();show(index+1)};$('note').oninput=save;
function envelope(){save();const reviewer=$('reviewer').value.trim();if(!reviewer)throw Error('Укажите проверяющего');
 const decisions=[];
 for(const r of rows){const d=drafts[r.clip_id];if(!d||!Object.keys(d.states).length)continue;
  if(!d.note.trim())throw Error('Нужно обоснование для '+r.clip_id);
  const names=v=>Object.keys(d.states).filter(k=>d.states[k]===v);
  decisions.push({clip_id:r.clip_id,review:{schema:'action_window_review_v1',catalog_sha256:report.catalog_sha256,
   video_sha256:r.video_sha256,center_sec:r.review_center_sec,window_sec:4,reviewer,evidence_note:d.note.trim(),reviewed_at:new Date().toISOString(),
   present_types:names('present'),absent_types:[...names('absent'),...names('whole')],whole_video_absent_types:names('whole')}})}
 if(!decisions.length)throw Error('Нет решений для экспорта');
 return {schema:'action_review_decisions_v1',events_sha256:report.events_sha256,split_sha256:report.split_sha256,catalog_sha256:report.catalog_sha256,rows:decisions};}
$('export').onclick=()=>{try{const data=envelope(),url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='action_review_decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}catch(e){alert(e.message)}};
$('restore').onchange=async e=>{try{const d=JSON.parse(await e.target.files[0].text());
 for(const k of ['events_sha256','split_sha256','catalog_sha256'])if(d[k]!==report[k])throw Error('Хеши источников изменились');
 for(const item of d.rows){if(!rows.some(r=>r.clip_id===item.clip_id))throw Error('Неизвестный ролик');const v=item.review,states={};
  v.present_types.forEach(t=>states[t]='present');v.absent_types.forEach(t=>states[t]='absent');v.whole_video_absent_types.forEach(t=>states[t]='whole');
  drafts[item.clip_id]={states,note:v.evidence_note};$('reviewer').value=v.reviewer}show(index);save();
 }catch(e){alert(e.message)}};
show(0);window.addEventListener('beforeunload',e=>{if(Object.values(drafts).some(d=>Object.keys(d.states).length)){e.preventDefault();e.returnValue=''}});
</script></html>'''.replace("__PAYLOAD__", payload)
    (out/"review.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", type=Path, help="Reuse already decoded private sheets")
    parser.add_argument("--reuse-sheets", type=Path, help="Build a fresh queue, copying frames only when video hash, point and tag match")
    args = parser.parse_args()
    records = load_records(ROOT/"data/annotations/events.csv", ROOT/"data/processed/split_proposal.csv", ROOT/"data/raw/videos")
    with (ROOT/"data/annotations/events.csv").open(encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    by_alias = {a:r for r in records for a in r.aliases}
    catalog = load_catalog()
    if args.reuse:
        report=json.loads((args.reuse/"review.json").read_text(encoding="utf-8"))
        for key,path in (("events_sha256",ROOT/"data/annotations/events.csv"),("split_sha256",ROOT/"data/processed/split_proposal.csv"),("catalog_sha256",CATALOG_PATH)):
            if report[key] != sha256(path):
                raise ValueError("Cannot reuse stale evidence: "+key)
        write_review_page(report,catalog,by_alias,args.reuse)
        print(args.reuse/"review.html")
        return
    out = ROOT/"data/local_artifacts"/("action_review_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    old_rows = {}
    if args.reuse_sheets:
        old_rows = {r["clip_id"]: r for r in json.loads((args.reuse_sheets/"review.json").read_text(encoding="utf-8"))["rows"]}
    rows = []
    for row in source:
        record = by_alias[row["clip_id"]]
        stamp = parse_time(row["timecode"])
        candidates = [e for e in catalog["events"] if row["observed_event_type"] in e.get("maps_from_observed_event_type", [])]
        entry = dict(clip_id=row["clip_id"], canonical_id=record.clip_id, split=record.split,
                     source_label=row["source_label"], timecode_sec=stamp, observed_event=row["observed_event"],
                     observed_event_type=row["observed_event_type"], video_sha256=record.sha256,
                     candidate_types=[e["event_type"] for e in candidates],
                     semantic_gap=SEMANTIC_GAPS.get(row["observed_event_type"]),
                     review_status="pending", candidate_details=[dict(event_type=e["event_type"], visible_action=e["visible_action"],
                       applicability=e["applicability"], video_cannot_establish=e.get("video_cannot_establish", []),
                       source_links=source_links_for(e["event_type"])) for e in candidates])
        previous = old_rows.get(row["clip_id"])
        if (previous and previous["video_sha256"] == record.sha256 and previous["timecode_sec"] == stamp
                and previous["observed_event_type"] == row["observed_event_type"]):
            sheet = out/f"{row['clip_id']}.jpg"
            shutil.copyfile(args.reuse_sheets/sheet.name, sheet)
            entry["sheet"] = sheet.name
            rows.append(entry)
            continue
        if stamp is not None:
            spans = [("whole-clip samples (not continuous review)", np.linspace(0, record.last_time, 8)),
                     ("around source timestamp", np.linspace(max(0,stamp-6), min(record.last_time,stamp+6),8))]
        else:
            spans = [("whole-clip samples (not continuous review)", np.linspace(0, record.last_time,8)),
                     ("midpoint 4-second window", np.linspace(max(0, record.last_time/2-2), min(record.last_time,record.last_time/2+2),8))]
        fig, axes = plt.subplots(2,8,figsize=(18,7),squeeze=False)
        for axes_row, (label,times) in zip(axes, spans):
            # Decoder accepts an arbitrary number of requested positions; training
            # keeps its separate strict 16-frame sampler.
            frames = decode_frames(record,times)
            for ax,frame,t in zip(axes_row,frames,times):
                ax.imshow(frame.permute(1,2,0).numpy())
                ax.set_title(f"{t:.2f}s",fontsize=9)
                ax.axis("off")
            axes_row[0].set_ylabel(label)
        fig.suptitle(f"{row['clip_id']} | canonical {record.clip_id} | {record.split} | source t={stamp}\n{row['observed_event_type']}",fontsize=12)
        fig.tight_layout()
        sheet=out/f"{row['clip_id']}.jpg"
        fig.savefig(sheet,dpi=100)
        plt.close(fig)
        entry["sheet"] = sheet.name
        rows.append(entry)
        print(row["clip_id"], flush=True)
    report=dict(schema="action_review_queue_v1", events_sha256=sha256(ROOT/"data/annotations/events.csv"),
                split_sha256=sha256(ROOT/"data/processed/split_proposal.csv"), catalog_sha256=sha256(CATALOG_PATH),
                note="Proposal queue only. Sparse frames do not verify absence of actions throughout a video.", rows=rows)
    (out/"review.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    write_review_page(report,catalog,by_alias,out)
    print("OUTPUT",out,flush=True)


if __name__ == "__main__": main()
