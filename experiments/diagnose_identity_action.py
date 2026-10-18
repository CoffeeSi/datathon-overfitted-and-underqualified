"""Private train/validation evidence and model-input diagnostics; never edits labels."""
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from overfitted_underqualified.ml.action_train import load_action_checkpoint,write_json
from overfitted_underqualified.ml.dataset import load_records,decode_frames,sample_times,sha256
from overfitted_underqualified.ml.preprocessing import video_transform


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    out=ROOT/'data/local_artifacts'/('identity_diagnostics_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'));out.mkdir()
    events=ROOT/'data/annotations/events.csv';split=ROOT/'data/processed/split_proposal.csv'
    records=load_records(events,split,ROOT/'data/raw/videos')
    selected=[r for r in records if r.clip_id in ('X-18','X-2')]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    model,ckpt,digest=load_action_checkpoint(args.checkpoint,device)
    transform=video_transform(ckpt['config'])
    index=ckpt['class_names'].index('client_identity_check_not_visible')
    mean=torch.tensor(transform.mean).view(3,1,1,1);std=torch.tensor(transform.std).view(3,1,1,1)
    results=[]
    for record in selected:
        center=record.event_times[0]
        times=list(np.arange(0,record.last_time,1.))+[record.last_time]
        # Every 1 Hz sample is shown, rather than a selected favourable frame.
        for page,start in enumerate(range(0,len(times),24),1):
            stamps=np.asarray(times[start:start+24]);frames=decode_frames(record,stamps)
            fig,axes=plt.subplots(4,6,figsize=(15,15));
            for ax in axes.flat:ax.axis('off')
            for ax,frame,t in zip(axes.flat,frames,stamps):
                ax.imshow(frame.permute(1,2,0).numpy());ax.set_title(f'{t:.2f}s')
            fig.suptitle(f'{record.clip_id} | {record.split} | source point {center}s | 1Hz overview')
            fig.tight_layout();fig.savefig(out/f'{record.clip_id}_overview_{page}.jpg',dpi=120);plt.close(fig)
        probes=[]
        for span in (4.,8.,12.):
            stamps=sample_times(center,record.last_time,span,16);frames=decode_frames(record,stamps)
            tensor=transform(frames)
            with torch.inference_mode():score=model(tensor.unsqueeze(0).to(device)).sigmoid()[0,index].item()
            visible=(tensor*std+mean).clamp(0,1)
            fig,axes=plt.subplots(2,4,figsize=(14,10))
            for j,i in enumerate((0,5,10,15)):
                h,w=frames.shape[-2:];axes[0,j].imshow(frames[i].permute(1,2,0).numpy())
                axes[0,j].add_patch(Rectangle(((171-112)/2/171*w,(128-112)/2/128*h),112/171*w,112/128*h,fill=False,color='yellow'))
                axes[0,j].set_title(f'raw {stamps[i]:.2f}s');axes[0,j].axis('off')
                axes[1,j].imshow(visible[:,i].permute(1,2,0).numpy(),interpolation='nearest');axes[1,j].axis('off')
                axes[1,j].set_title('actual 112 x 112 input')
            fig.suptitle(f'{record.clip_id} | context={span}s | score={score:.4f} | altered context is sensitivity only')
            fig.tight_layout();fig.savefig(out/f'{record.clip_id}_context_{span:g}.jpg',dpi=120);plt.close(fig)
            probes.append(dict(requested_span_sec=span,start_sec=float(stamps[0]),end_sec=float(stamps[-1]),probability=score))
        # Full-resolution reference for examination of the face/document/desk area.
        frame=decode_frames(record,np.asarray([center]))[0]
        from PIL import Image
        Image.fromarray(frame.permute(1,2,0).numpy()).save(out/f'{record.clip_id}_reference.png')
        results.append(dict(clip_id=record.clip_id,split=record.split,video_sha256=record.sha256,
            center_sec=center,last_time_sec=record.last_time,dimensions=[int(frame.shape[-1]),int(frame.shape[-2])],probes=probes))
        print(record.clip_id,'complete',flush=True)
    write_json(out/'diagnostics.json',dict(events_sha256=sha256(events),split_sha256=sha256(split),checkpoint_sha256=digest,
        device=device,records=results,note='Sensitivity on already inspected validation; not an independent experiment or label expansion.'))
    print(out,flush=True)


if __name__=='__main__':main()
