"use client";
/* Evidence is displayed locally: never pass sensitive data through an image optimizer. */
/* eslint-disable @next/next/no-img-element */
import { useState } from "react";
import { Dialog } from "@/components/ui/dialog";
import { Icon } from "@/components/ui/icon";
import { timecode } from "@/lib/format";
import type { EvidenceFrame } from "@/lib/contracts";

export function EvidenceImage({ frame, alt, className = "" }: { frame: EvidenceFrame; alt: string; className?: string }) {
  const [failed, setFailed] = useState(false);
  return failed ? <div className={`image-error ${className}`} role="img" aria-label={`${alt}: кадр недоступен`}><Icon name="film" /><span>Не удалось открыть кадр</span></div>
    : <img className={className} src={`data:image/jpeg;base64,${frame.image_base64}`} alt={alt} onError={() => setFailed(true)} />;
}

export function EvidenceGallery({ frames, title, index, onIndex, onClose }: {
  frames: EvidenceFrame[]; title: string; index: number | null; onIndex: (index: number) => void; onClose: () => void;
}) {
  const current = index === null ? null : frames[index];
  return <Dialog open={!!current} onClose={onClose} title={title} className="evidence-dialog" onKeyDown={(event) => {
      if (event.key === "ArrowLeft" && index! > 0) { event.preventDefault(); onIndex(index! - 1); }
      if (event.key === "ArrowRight" && index! < frames.length - 1) { event.preventDefault(); onIndex(index! + 1); }
    }}>
    {current && <div tabIndex={0} className="gallery-body" aria-label="Просмотр кадров. Стрелки влево и вправо переключают кадры.">
      <EvidenceImage key={current.image_base64} frame={current} alt={`${title}, ${timecode(current.timestamp_sec, true)}`} />
      <div className="gallery-controls"><span className="mono">{timecode(current.timestamp_sec, true)}</span><div><button className="icon-button" aria-label="Предыдущий кадр" disabled={index === 0} onClick={() => onIndex(index! - 1)}><Icon name="chevron-left" /></button><span>{index! + 1} / {frames.length}</span><button className="icon-button" aria-label="Следующий кадр" disabled={index === frames.length - 1} onClick={() => onIndex(index! + 1)}><Icon name="chevron-right" /></button></div><span className="gallery-hint">← → кадры · Esc закрыть</span></div>
    </div>}
  </Dialog>;
}
