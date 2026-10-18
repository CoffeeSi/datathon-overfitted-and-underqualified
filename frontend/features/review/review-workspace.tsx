"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "@/components/ui/icon";
import { Dialog } from "@/components/ui/dialog";
import { useAnalysisJob } from "@/features/analysis-job/use-analysis-job";
import { updateRecentStatus, useRecentJobs } from "@/features/recent-jobs/storage";
import { getVideo } from "@/lib/local-session";
import { timecode } from "@/lib/format";
import { validateVideoFile } from "@/features/upload/validation";
import { reviewEvents } from "./presentation";
import { Timeline } from "./timeline";
import { EvidenceGallery, EvidenceImage } from "./evidence";
import { ActionContext, UnresolvedActions } from "./action-context";

const verdictCopy = {
  violation: ["Требует внимания", "Сервис отметил возможные нарушения. Проверьте события в контексте записи."],
  no_violation: ["Нарушения не обнаружены", "Сервис не выделил событий для проверки. Это не гарантирует отсутствия нарушений за пределами возможностей анализатора."],
  insufficient_data: ["Недостаточно данных", "По этой записи сервис не смог сделать вывод. Проверьте качество и полноту видео перед новым анализом."],
};

export function ReviewWorkspace({ jobId }: { jobId: string }) {
  const { job, error, loading, paused, elapsedSec, retry } = useAnalysisJob(jobId);
  const recent = useRecentJobs().find((entry) => entry.jobId === jobId);
  const params = useSearchParams();
  const events = useMemo(() => reviewEvents(job?.status === "done" ? job.violations : []), [job]);
  const selected = events.find((entry) => entry.id === params.get("event")) ?? events[0];
  const selectedIndex = selected ? events.findIndex((entry) => entry.id === selected.id) : -1;
  const [search, setSearch] = useState("");
  const [type, setType] = useState("");
  const [source, setSource] = useState<{ file: File; url: string; verified: boolean } | null>(null);
  const [videoFailed, setVideoFailed] = useState(false);
  const [duration, setDuration] = useState(0);
  const [position, setPosition] = useState(0);
  const [localError, setLocalError] = useState("");
  const [notice, setNotice] = useState("");
  const [gallery, setGallery] = useState<number | null>(null);
  const [shortcuts, setShortcuts] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const pickFile = useRef<HTMLInputElement>(null);
  const sourceUrl = useRef<string | null>(null);
  const demo = job?.analysis_provider === "mock" || job?.analysis_provider === "synthetic-demo";

  useEffect(() => {
    if (job) updateRecentStatus(jobId, job.status);
  }, [jobId, job]);

  useEffect(() => {
    const file = getVideo(jobId);
    if (file) {
      const url = URL.createObjectURL(file);
      sourceUrl.current = url;
      queueMicrotask(() => setSource({ file, url, verified: true }));
    }
    return () => { if (sourceUrl.current) URL.revokeObjectURL(sourceUrl.current); sourceUrl.current = null; };
  }, [jobId]);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 3500);
    return () => clearTimeout(timer);
  }, [notice]);

  const seek = useCallback((seconds: number) => {
    setPosition(seconds);
    if (video.current && Number.isFinite(video.current.duration)) video.current.currentTime = Math.min(seconds, video.current.duration);
  }, []);

  const select = useCallback((id: string) => {
    const next = events.find((entry) => entry.id === id);
    if (!next) return;
    const url = new URL(window.location.href);
    url.searchParams.set("event", id);
    window.history.replaceState(null, "", `${url.pathname}${url.search}`);
    setGallery(null);
    seek(next.timecode_sec);
  }, [events, seek]);

  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.closest("input,textarea,select,button,a,video,audio,[contenteditable=true],dialog[open]") || document.querySelector("dialog[open]")) return;
      if (event.key === "?" || event.key === "F1") { event.preventDefault(); setShortcuts(true); }
      if ((event.key === "ArrowRight" || event.key === "j") && selectedIndex < events.length - 1) { event.preventDefault(); select(events[selectedIndex + 1].id); }
      if ((event.key === "ArrowLeft" || event.key === "k") && selectedIndex > 0) { event.preventDefault(); select(events[selectedIndex - 1].id); }
      if (event.code === "Space" && video.current && !videoFailed) {
        event.preventDefault();
        if (video.current.paused) void video.current.play().catch(() => setLocalError("Браузер не смог начать воспроизведение.")); else video.current.pause();
      }
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [events, selectedIndex, select, videoFailed]);

  async function copy(text: string, message: string) {
    try { await navigator.clipboard.writeText(text); setNotice(message); }
    catch { setNotice("Копирование недоступно. Выделите текст и скопируйте вручную."); }
  }

  function download() {
    if (!job) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(job, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `fokus-${jobId}.json`; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("JSON подготовлен для скачивания");
  }

  function attachVideo(file?: File) {
    if (!file) return;
    const problem = validateVideoFile(file, Infinity);
    if (problem) { setLocalError(problem); return; }
    if (sourceUrl.current) URL.revokeObjectURL(sourceUrl.current);
    const url = URL.createObjectURL(file);
    sourceUrl.current = url;
    // Reselection is playback-only: there is no server hash with which to verify identity.
    setSource({ file, url, verified: false });
    setVideoFailed(false); setDuration(0); setLocalError("");
  }

  const filtered = events.filter((entry) => (!type || entry.type === type) && `${entry.title} ${entry.description} ${entry.type}`.toLocaleLowerCase("ru").includes(search.toLocaleLowerCase("ru")));
  const types = [...new Map(events.map((entry) => [entry.type, entry.title])).entries()];
  const extent = duration || Math.max(30, ...events.map((entry) => entry.timecode_sec + 5));
  const playable = !!source && !videoFailed && duration > 0;

  return <main id="main-content" className="review-page">
    <p className="sr-only" role="status">{job?.status === "done" ? `Обработка завершена. ${verdictCopy[job.verdict][0]}. Событий: ${events.length}.` : job?.status === "failed" ? "Задание завершилось ошибкой." : job?.status === "processing" ? "Задание обрабатывается." : "Запрашиваем задание."}</p>
    <div className="review-topline"><Link href="/" className="text-button"><Icon name="arrow-left" /> Все задания</Link><span className="eyebrow">ПРОСМОТР / 02</span><button className="text-button" onClick={() => setShortcuts(true)}><Icon name="keyboard" /><span className="desktop-label">Горячие клавиши</span></button></div>
    <header className="review-heading"><div><p className="eyebrow">{demo ? "ДЕМО / СИНТЕТИЧЕСКИЕ ДАННЫЕ" : "АНАЛИЗ ВИДЕО"}</p><h1>{recent?.filename ?? source?.file.name ?? (demo ? "Моменты в фокусе." : "Просмотр записи")}</h1><button className="job-id text-button" onClick={() => void copy(jobId, "ID задания скопирован")} aria-label="Скопировать ID задания"><span className="mono">{jobId}</span><Icon name="copy" size={14} /></button></div>{job?.status === "done" && <button className="button button-secondary download-button" onClick={download}><Icon name="download" /> Скачать JSON</button>}</header>
    {demo && <div className="demo-banner"><span className="demo-badge">ДЕМО</span><p>{job.analysis_provider === "mock" ? "Подключён mock provider. Результат синтетический; содержимое загруженного видео не распознаётся." : "Безопасный пример интерфейса. События и иллюстрации созданы для демонстрации."}</p></div>}

    {error && <div className="request-error" role="alert"><Icon name="alert" /><div><strong>{error.status === 404 ? "Задание не найдено" : error.status === 401 || error.status === 403 ? "Нет доступа к сервису" : paused ? "Проверка приостановлена" : "Не удалось обновить состояние"}</strong><p>{error.message}</p>{error.status === 404 && <p>Ссылка может быть неверной или срок хранения результата истёк. Вернитесь к заданиям, чтобы открыть другой ID.</p>}</div><button className="button button-secondary" onClick={retry} disabled={loading}><Icon name="refresh" /> Проверить снова</button></div>}

    {!job && !error && <section className="waiting-state" aria-busy="true"><div className="activity-mark" aria-hidden="true"><span /><span /><span /></div><p className="eyebrow">ПОДКЛЮЧАЕМСЯ К ЗАДАНИЮ</p><h2>Собираем рабочее место.</h2><p>Проверяем состояние и доступность результата.</p></section>}
    {job?.status === "processing" && <section className="waiting-state"><div className="activity-mark" aria-hidden="true"><span /><span /><span /></div><p className="eyebrow">{paused ? "ПРОВЕРКА НА ПАУЗЕ" : "ЗАПИСЬ В ОБРАБОТКЕ"}</p><h2>Важные моменты<br /><span>скоро будут здесь.</span></h2><p role="status">{paused ? "Задание может продолжать работу на сервере." : "Сервис ещё готовит результат. Эту страницу можно закрыть и вернуться по ссылке."}</p><div className="waiting-clock"><Icon name="clock" /><span className="mono">{timecode(elapsedSec)}</span><span>наблюдаем на этой странице</span></div><p className="muted">Точный прогресс анализа сервис не сообщает.</p>{paused && !error && <button className="button button-secondary" onClick={retry}>Возобновить проверку</button>}</section>}
    {job?.status === "failed" && <section className="failed-state"><span className="state-symbol"><Icon name="alert" size={32} /></span><p className="eyebrow">ОБРАБОТКА НЕ ЗАВЕРШЕНА</p><h2>Не удалось разобрать запись.</h2><p>{job.error.message}</p><span className="error-code mono">Код: {job.error.code}</span><Link className="button button-primary" href="/">Выбрать видео заново <Icon name="arrow-right" /></Link><small>Повторная отправка создаст новое задание.</small></section>}

    {job?.status === "done" && <>
      <UnresolvedActions candidates={job.unresolved_candidates} />
      <div className="verdict-bar" data-verdict={job.verdict}><div className="verdict-icon"><Icon name={job.verdict === "violation" ? "eye" : job.verdict === "no_violation" ? "check" : "alert"} /></div><div><h2>{verdictCopy[job.verdict][0]}</h2><p>{verdictCopy[job.verdict][1]}</p></div><span className="event-total"><strong>{String(events.length).padStart(2, "0")}</strong><span>событий</span></span></div>
      <div className="review-grid">
        <div className="viewer-column">
          <section className="video-surface" aria-label="Видео и визуальный контекст">
            <div className="video-label"><span><span className="status-dot" />{source && !videoFailed ? "ЛОКАЛЬНОЕ ВИДЕО" : selected ? "КАДР СОБЫТИЯ" : "ИСХОДНАЯ ЗАПИСЬ"}</span><span className="mono">{timecode(source && !videoFailed ? position : selected?.evidence_frames[0]?.timestamp_sec ?? 0, true)}</span></div>
            <div className="video-stage">
              {source && !videoFailed ? <video key={source.url} ref={video} src={source.url} controls playsInline preload="metadata" aria-label="Исходное видео" onTimeUpdate={(e) => setPosition(e.currentTarget.currentTime)} onLoadedMetadata={(e) => { if (Number.isFinite(e.currentTarget.duration)) { setDuration(e.currentTarget.duration); e.currentTarget.currentTime = Math.min(selected?.timecode_sec ?? 0, e.currentTarget.duration); } }} onError={() => setVideoFailed(true)} /> : selected ? <button className="hero-evidence" onClick={() => setGallery(0)} aria-label="Увеличить кадр выбранного события"><EvidenceImage key={selected.id} frame={selected.evidence_frames[0]} alt={`${selected.title}, ${timecode(selected.evidence_frames[0].timestamp_sec, true)}`} /><span><Icon name="expand" /> Рассмотреть кадр</span></button> : <div className="video-placeholder"><Icon name="film" size={48} /><span>Здесь может быть ваша запись</span><p>Выберите исходный файл для локального просмотра.</p></div>}
            </div>
            <div className="video-caption"><span>{videoFailed ? "Кодек не поддерживается браузером. Кадры результата доступны ниже." : source ? source.verified ? "Исходный файл доступен в этой вкладке." : "Файл выбран локально. Соответствие заданию не проверено." : "Видео не хранится в браузере после перезагрузки. Результат остаётся доступным."}</span><button className="text-button" onClick={() => pickFile.current?.click()}><Icon name="folder" />{source ? "Заменить" : "Открыть .mov"}</button><input className="sr-only" ref={pickFile} type="file" accept=".mov,video/quicktime" tabIndex={-1} aria-label="Выбрать исходное видео для просмотра" onChange={(e) => { attachVideo(e.target.files?.[0]); e.target.value = ""; }} /></div>
          </section>
          {localError && <p className="form-error" role="alert">{localError}</p>}
          <Timeline events={events} selectedId={selected?.id} extent={extent} position={playable ? position : selected?.timecode_sec ?? 0} videoAvailable={playable} onSelect={select} onSeek={seek} />
          {selected && <ActionContext finding={selected} />}
          <section className="event-detail" aria-labelledby="detail-title">
            {selected ? <><div className="detail-heading"><div><p className="eyebrow">В КОНТЕКСТЕ / {String(selectedIndex + 1).padStart(2, "0")}</p><h2 id="detail-title">{selected.title}</h2></div><button className="time-copy" onClick={() => void copy(timecode(selected.timecode_sec, true), "Таймкод скопирован")} aria-label="Скопировать таймкод события"><span>{timecode(selected.timecode_sec, true)}</span><Icon name="copy" size={14} /></button></div><p className="event-description">{selected.description}</p><div className="evidence-heading"><span>КАДРЫ СОБЫТИЯ</span><span>{selected.evidence_frames.length} / JPEG</span></div><div className="evidence-strip">{selected.evidence_frames.map((frame, index) => <button key={`${selected.id}-${index}`} className="evidence-thumb" onClick={() => setGallery(index)} aria-label={`Открыть кадр ${index + 1}, ${timecode(frame.timestamp_sec, true)}`}><EvidenceImage frame={frame} alt={`Контекст ${selected.title}`} /><span className="mono">{timecode(frame.timestamp_sec, true)}<Icon name="expand" size={14} /></span></button>)}</div><details className="event-technical"><summary>Данные события</summary><dl><dt>Тип от сервиса</dt><dd>{selected.type}</dd><dt>Источник</dt><dd>{job.analysis_provider}</dd></dl></details></> : <div className="no-events-detail"><Icon name={job.verdict === "no_violation" ? "check" : "eye"} size={28} /><h2 id="detail-title">{job.verdict === "no_violation" ? "Список событий пуст." : "Нужен дополнительный контекст."}</h2><p>{job.verdict === "no_violation" ? "Вы можете просмотреть исходное видео самостоятельно или сохранить ответ сервиса в JSON." : "Попробуйте запись с лучшей видимостью кассовой зоны. Отсутствие событий в этом ответе не означает, что нарушений нет."}</p></div>}
          </section>
        </div>
        <aside className="events-panel" aria-labelledby="events-title"><div className="events-heading"><div><p className="eyebrow">НАВИГАЦИЯ ПО ЗАПИСИ</p><h2 id="events-title">События <span>{events.length}</span></h2></div><div className="event-nav"><button className="icon-button" aria-label="Предыдущее событие" disabled={selectedIndex <= 0} onClick={() => select(events[selectedIndex - 1].id)}><Icon name="chevron-left" /></button><button className="icon-button" aria-label="Следующее событие" disabled={selectedIndex < 0 || selectedIndex === events.length - 1} onClick={() => select(events[selectedIndex + 1].id)}><Icon name="chevron-right" /></button></div></div>
          {events.length >= 3 && <div className="event-filters"><label className="search-field"><Icon name="search" size={17} /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Найти в событиях" aria-label="Поиск событий" /></label><select value={type} onChange={(e) => setType(e.target.value)} aria-label="Тип события"><option value="">Все типы событий</option>{types.map(([value, title]) => <option value={value} key={value}>{title}</option>)}</select></div>}
          <ol className="events-list">{filtered.map((entry) => <li key={entry.id}><button className="event-row" data-selected={selected?.id === entry.id} aria-pressed={selected?.id === entry.id} onClick={() => select(entry.id)}><span className="event-index">{String(events.indexOf(entry) + 1).padStart(2, "0")}</span><span className="event-row-content"><span className="event-row-time mono">{timecode(entry.timecode_sec, true)}<Icon name="arrow-right" size={15} /></span><strong>{entry.title}</strong><span className="event-excerpt">{entry.description}</span></span></button></li>)}</ol>
          {filtered.length === 0 && <div className="events-empty"><Icon name="search" /><p>{events.length ? "По этому запросу ничего нет." : "Отметок в ответе нет."}</p>{events.length > 0 && <button className="text-button" onClick={() => { setSearch(""); setType(""); }}>Сбросить фильтры</button>}</div>}
          <div className="review-note"><Icon name="eye" /><p>Сначала — контекст.<br /><span>Результат сервиса помогает найти момент. Окончательный вывод требует проверки специалистом.</span></p></div>
        </aside>
      </div>
    </>}
    {selected && <EvidenceGallery frames={selected.evidence_frames} title={`Кадры · ${selected.title}`} index={gallery} onIndex={setGallery} onClose={() => setGallery(null)} />}
    <Dialog open={shortcuts} onClose={() => setShortcuts(false)} title="Всё под рукой"><div className="shortcuts"><p>Когда фокус вне полей и кнопок:</p><dl><dt><kbd>←</kbd> <kbd>→</kbd></dt><dd>Предыдущее / следующее событие</dd><dt><kbd>Пробел</kbd></dt><dd>Пауза / воспроизведение локального видео</dd><dt><kbd>?</kbd></dt><dd>Открыть эту справку</dd><dt><kbd>Tab</kbd></dt><dd>Переход между элементами</dd><dt><kbd>Esc</kbd></dt><dd>Закрыть просмотр кадра или справку</dd></dl><p className="muted">В просмотре кадра стрелки переключают изображения.</p></div></Dialog>
    <div className={`toast ${notice ? "visible" : ""}`} role="status">{notice && <><Icon name="check" size={16} />{notice}</>}</div>
  </main>;
}
