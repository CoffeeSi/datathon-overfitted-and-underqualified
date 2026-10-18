"""Private full-clip frame sequences for resolving action-label uncertainty.

No predictions, annotation edits, or legal conclusions. Generated images stay in data/local_artifacts/.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import cv2,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from overfitted_underqualified.ml.dataset import load_records,decode_frames,sha256
from overfitted_underqualified.ml.action_train import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip',action='append',required=True)
    parser.add_argument('--step',type=float,default=1.)
    args=parser.parse_args()
    if not np.isfinite(args.step) or args.step<=0:raise ValueError('Invalid step')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    events=ROOT/'data/annotations/events.csv';split=ROOT/'data/processed/split_proposal.csv'
    records=load_records(events,split,ROOT/'data/raw/videos')
    selected=[r for r in records if r.clip_id in args.clip]
    if {r.clip_id for r in selected}!=set(args.clip):raise ValueError('Use canonical IDs')
    out=ROOT/'data/local_artifacts'/('action_candidates_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'));out.mkdir()
    results=[]
    for record in selected:
        times=np.unique(np.r_[np.arange(0,record.last_time,args.step),record.last_time])
        for page,start in enumerate(range(0,len(times),24),1):
            stamps=times[start:start+24];frames=decode_frames(record,stamps)
            fig,axes=plt.subplots(4,6,figsize=(15,15))
            for ax in axes.flat:ax.axis('off')
            for ax,frame,t in zip(axes.flat,frames,stamps):
                ax.imshow(frame.permute(1,2,0).numpy());ax.set_title(f'{t:.2f}s',fontsize=10)
            fig.suptitle(f'{record.clip_id} | {record.split} | step={args.step}s; images are not a legal finding')
            fig.tight_layout();fig.savefig(out/f'{record.clip_id}_page_{page}.jpg',dpi=120);plt.close(fig)
        for center in record.event_times or (record.last_time/2,):
            frame=decode_frames(record,np.array([center]))[0]
            Image.fromarray(frame.permute(1,2,0).numpy()).save(out/f'{record.clip_id}_{center:g}.png')
        results.append(dict(clip_id=record.clip_id,split=record.split,video_sha256=record.sha256,
            last_time_sec=record.last_time,source_times=list(record.event_times),sample_count=len(times)))
        print(record.clip_id,'pages',(len(times)+23)//24,flush=True)
    write_json(out/'manifest.json',dict(events_sha256=sha256(events),split_sha256=sha256(split),
        method=f'{args.step}s frame sequences plus exact source reference frames; labels not edited',videos=results))
    print(out,flush=True)


if __name__=='__main__':main()
