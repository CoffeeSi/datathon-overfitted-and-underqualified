"use client";
import { timecode } from "@/lib/format";
import type { ReviewEvent } from "./presentation";

export function Timeline({ events, selectedId, extent, position, videoAvailable, onSelect, onSeek }: {
  events: ReviewEvent[]; selectedId?: string; extent: number; position: number; videoAvailable: boolean;
  onSelect: (id: string) => void; onSeek: (seconds: number) => void;
}) {
  // Nearby marks occupy distinct lanes. Their hit targets remain separate even at the same time.
  const lanes: number[] = [];
  const marks = events.map((event) => {
    const pct = Math.min(98, Math.max(2, event.timecode_sec / extent * 100));
    let lane = lanes.findIndex((last) => pct - last >= 6);
    if (lane < 0) { lane = lanes.length; lanes.push(pct); } else lanes[lane] = pct;
    return { event, pct, lane };
  });
  return <section className="timeline" aria-label="Временная шкала событий">
    <div className="timeline-heading"><span>{videoAvailable ? "ВРЕМЕННАЯ ШКАЛА" : "ШКАЛА СОБЫТИЙ"}</span><span>{events.length} отметок <i /> {timecode(position, true)}</span></div>
    <div className="timeline-scroll"><div className="timeline-track" style={{ height: `${58 + Math.max(1, lanes.length) * 30}px` }}>
      <div className="timeline-ruler" aria-hidden="true">{Array.from({ length: 9 }, (_, i) => <span key={i}>{timecode(extent * i / 8)}</span>)}</div>
      <div className="timeline-position" style={{ left: `${Math.min(100, position / extent * 100)}%` }} aria-hidden="true" />
      {marks.map(({ event, pct, lane }, index) => <button key={event.id} type="button" className="timeline-marker" data-selected={selectedId === event.id} aria-pressed={selectedId === event.id} aria-label={`Отметка ${index + 1}: ${event.title}, ${timecode(event.timecode_sec, true)}`} style={{ left: `${pct}%`, top: `${36 + lane * 30}px` }} onClick={() => onSelect(event.id)}><span>{String(index + 1).padStart(2, "0")}</span><span className="marker-preview" role="tooltip" style={pct < 50 ? { left: -14, right: "auto" } : undefined}>{timecode(event.timecode_sec, true)} · {event.title}</span></button>)}
    </div></div>
    {videoAvailable && <input type="range" min={0} max={extent} step={0.1} value={Math.min(position, extent)} onChange={(e) => onSeek(Number(e.target.value))} aria-label="Позиция видео" aria-valuetext={timecode(position, true)} className="timeline-seek" />}
    {!videoAvailable && <p className="timeline-note">Масштаб по найденным отметкам. Длительность исходного видео неизвестна.</p>}
  </section>;
}
