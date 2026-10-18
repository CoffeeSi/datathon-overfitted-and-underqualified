"""Decode complete videos and make private 1 Hz sequences for descriptive review."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from overfitted_underqualified.ml.dataset import load_records,sha256


def main():
    events=ROOT/'data/annotations/events.csv';split=ROOT/'data/processed/split_proposal.csv'
    with events.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    ids={r['clip_id'] for r in rows if r['observed_event_type']=='not_reported' or r['observed_event']=='Событие отдельно не описано в исходной разметке'}
    records=load_records(events,split,ROOT/'data/raw/videos')
    out=ROOT/'data/local_artifacts'/('undescribed_review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:
        font=ImageFont.load_default()
    reviewed=[]
    for record in records:
        aliases=[a for a in record.aliases if a in ids]
        if not aliases:continue
        directory=out/record.clip_id;directory.mkdir()
        cap=cv2.VideoCapture(str(record.path));count=0;samples=[];next_second=0;last=None
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                t=count/record.fps
                if t>=next_second:
                    img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    img.save(directory/f'{count:06d}.jpg',quality=93)
                    samples.append((count,t))
                    next_second+=1
                last=frame;count+=1
        finally:cap.release()
        if last is None:raise ValueError('No decoded frames: '+record.clip_id)
        if samples[-1][0]!=count-1:
            Image.fromarray(cv2.cvtColor(last,cv2.COLOR_BGR2RGB)).save(directory/f'{count-1:06d}.jpg',quality=93)
            samples.append((count-1,(count-1)/record.fps))
        pages=[]
        for offset in range(0,len(samples),36):
            batch=samples[offset:offset+36];w,h=180,330
            sheet=Image.new('RGB',(6*w,52+((len(batch)+5)//6)*(h+25)),'white');draw=ImageDraw.Draw(sheet)
            draw.text((8,6),f'{record.clip_id} ({", ".join(aliases)}) | 1 Hz | {batch[0][1]:.1f}-{batch[-1][1]:.1f}s',fill='black',font=font)
            for j,(index,t) in enumerate(batch):
                image=Image.open(directory/f'{index:06d}.jpg');image.thumbnail((w,h))
                x=(j%6)*w;y=52+(j//6)*(h+25)
                draw.text((x+3,y),f'{t:.2f}s',fill='black',font=font)
                sheet.paste(image,(x+(w-image.width)//2,y+25))
            name=f'{record.clip_id}_{offset//36+1:02d}.jpg';sheet.save(out/name,quality=94);pages.append(name)
        reviewed.append(dict(clip_id=record.clip_id,source_ids=aliases,video_sha256=record.sha256,
                             nominal_frames=record.frame_count,decoded_frames=count,
                             decoded_last_sec=(count-1)/record.fps,sampled_frames=len(samples),pages=pages))
        print(record.clip_id,count,'frames',len(samples),'samples',len(pages),'pages',flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(events_sha256=sha256(events),split_sha256=sha256(split),
        method='Sequential decoding to EOF; visual review at 1 Hz plus final frame. Short actions may fall between samples.',videos=reviewed),indent=2)+'\n',encoding='utf-8')
    print('OUTPUT',out,flush=True)


if __name__=='__main__':main()
