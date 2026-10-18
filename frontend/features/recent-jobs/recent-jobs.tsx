"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRef, useState, type FormEvent } from "react";
import { Icon } from "@/components/ui/icon";
import { forgetRecentJobs, useRecentJobs } from "./storage";

const labels = { processing: "В обработке", done: "Готово", failed: "Не завершено" };
const dateFormat = new Intl.DateTimeFormat("ru-RU", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });

export function RecentJobs() {
  const jobs = useRecentJobs();
  const router = useRouter();
  const [jobId, setJobId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);

  function openJob(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const id = jobId.trim();
    if (!/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(id)) {
      setError("Введите полный ID задания: пять групп символов через дефис.");
      return;
    }
    router.push(`/review/${encodeURIComponent(id.toLowerCase())}`);
  }

  return (
    <section className="recent-section" aria-labelledby="recent-title">
      <div className="section-heading">
        <div><p className="eyebrow">НА ЭТОМ УСТРОЙСТВЕ</p><h2 ref={titleRef} tabIndex={-1} id="recent-title">Недавние видео</h2></div>
        {jobs.length > 0 && <button className="text-button" type="button" onClick={() => { forgetRecentJobs(); titleRef.current?.focus(); }}>Очистить список <Icon name="close" /></button>}
      </div>
      {jobs.length > 0 && <p className="recent-hint">Загрузки сохраняются здесь автоматически. Нажмите на видео, чтобы открыть анализ.</p>}
      {jobs.length > 0 ? (
        <ul className="recent-list">
          {jobs.map((job) => <li key={job.jobId}>
            <Link href={`/review/${encodeURIComponent(job.jobId)}`} className="recent-row">
              <Icon name="film" />
              <span className="recent-file"><strong>{job.filename}</strong><span className="recent-meta">{dateFormat.format(new Date(job.createdAt))} · {job.jobId.slice(0, 8)}</span></span>
              <span className="recent-status" data-status={job.status}><span className="status-dot" />{labels[job.status]}</span>
              <Icon name="arrow-right" />
            </Link>
          </li>)}
        </ul>
      ) : (
        <div className="recent-empty"><Icon name="folder" /><div><p>Здесь начнётся ваша история просмотра.</p><span>После отправки видео ссылка на задание сохранится в этом браузере.</span></div></div>
      )}
      <form className="recent-open-form" onSubmit={openJob}>
        <label htmlFor="open-job">Уже есть ID задания?</label>
        <div><input id="open-job" name="jobId" value={jobId} onChange={(event) => { setJobId(event.target.value); setError(null); }} placeholder="Вставьте ID" autoComplete="off" autoCapitalize="none" spellCheck={false} maxLength={128} aria-invalid={!!error} aria-describedby={error ? "open-job-error" : undefined} /><button className="button button-secondary" type="submit" disabled={!jobId.trim()}>Открыть <Icon name="arrow-right" /></button></div>
        {error && <p id="open-job-error" className="form-error" role="alert">{error}</p>}
      </form>
    </section>
  );
}
