"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState, type DragEvent } from "react";
import { Icon } from "@/components/ui/icon";
import { fetchService, uploadVideo } from "@/lib/client";
import { ApiError, type ServiceStatus } from "@/lib/contracts";
import { setVideo } from "@/lib/local-session";
import { RecentJobs } from "@/features/recent-jobs/recent-jobs";
import { rememberJob } from "@/features/recent-jobs/storage";
import { formatFileSize, validateVideoFile } from "./validation";

type Selection = { file: File; url: string; duration: number | null; previewFailed: boolean };

function durationLabel(seconds: number): string {
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60).toString().padStart(2, "0")}:${(total % 60).toString().padStart(2, "0")}`;
}

export function UploadWorkspace() {
  const router = useRouter();
  const [service, setService] = useState<ServiceStatus | null>(null);
  const [checking, setChecking] = useState(true);
  const [serviceError, setServiceError] = useState(false);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const chooseButtonRef = useRef<HTMLButtonElement>(null);
  const previewUrl = useRef<string | null>(null);
  const dragDepth = useRef(0);
  const uploadController = useRef<AbortController | null>(null);
  const serviceController = useRef<AbortController | null>(null);
  const sending = useRef(false);
  const mounted = useRef(false);

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    serviceController.current = controller;
    fetchService(controller.signal)
      .then((status) => { if (!controller.signal.aborted) setService(status); })
      .catch(() => { if (!controller.signal.aborted) setServiceError(true); })
      .finally(() => { if (!controller.signal.aborted) setChecking(false); });
    return () => {
      mounted.current = false;
      serviceController.current?.abort();
      uploadController.current?.abort();
      if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    };
  }, []);

  async function checkService() {
    serviceController.current?.abort();
    const controller = new AbortController();
    serviceController.current = controller;
    setChecking(true);
    setServiceError(false);
    try {
      const status = await fetchService(controller.signal);
      if (!controller.signal.aborted) setService(status);
    } catch {
      if (!controller.signal.aborted) setServiceError(true);
    } finally {
      if (!controller.signal.aborted) setChecking(false);
    }
  }

  function clearSelection() {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    previewUrl.current = null;
    setSelection(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  function selectFiles(files: FileList | File[]) {
    if (sending.current) return;
    if (files.length !== 1) {
      setError("Выберите одно видео. Каждая запись обрабатывается отдельным заданием.");
      return;
    }
    const file = files[0];
    const validationError = validateVideoFile(file, service?.maxUploadBytes ?? Infinity);
    if (validationError) { setError(validationError); return; }
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    const url = URL.createObjectURL(file);
    previewUrl.current = url;
    setSelection({ file, url, duration: null, previewFailed: false });
    setProgress(null);
    setError(null);
  }

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    if (event.dataTransfer.files.length > 0) selectFiles(event.dataTransfer.files);
  }

  async function startUpload() {
    if (!selection || sending.current || !service?.ready || serviceError || checking) return;
    const validationError = validateVideoFile(selection.file, service.maxUploadBytes);
    if (validationError) { setError(validationError); return; }
    sending.current = true;
    setUploading(true);
    setError(null);
    const controller = new AbortController();
    uploadController.current = controller;
    try {
      const accepted = await uploadVideo(selection.file, setProgress, controller.signal);
      if (controller.signal.aborted) return;
      setVideo(accepted.job_id, selection.file);
      rememberJob({ jobId: accepted.job_id, filename: selection.file.name, createdAt: new Date().toISOString(), status: "processing" });
      clearSelection();
      setUploading(false);
      router.push(`/review/${encodeURIComponent(accepted.job_id)}`);
      sending.current = false;
    } catch (caught) {
      if (!mounted.current) return;
      if (controller.signal.aborted) {
        setError("Передача прервана. Если сервис уже принял файл, задание может продолжить обработку. Повторная отправка создаст новое задание.");
      } else {
        setError(caught instanceof ApiError ? caught.message : "Не удалось подтвердить приём файла. Автоматического повтора не будет; проверьте подключение перед новой отправкой.");
      }
      sending.current = false;
      setUploading(false);
    }
  }

  const offline = serviceError || service?.health === "offline";
  const ready = !checking && !offline && service?.ready === true;
  const mock = service?.provider === "mock";
  const serviceLabel = checking ? "Проверяем подключение" : offline ? "Нет связи с сервисом" : ready ? mock ? "Подключён демоанализатор" : "Сервис готов к анализу" : "Анализатор пока не подключён";

  return (
    <main id="main-content" className="home-page">
      <header className="home-heading">
        <div><p className="eyebrow">РАБОЧЕЕ МЕСТО / 01</p><h1 className="home-title">Посмотрите<br /><span>внимательнее.</span></h1></div>
        <p className="home-intro">От полной записи — к важным моментам.<br />Загрузите кассовое видео, изучите найденные события и проверьте детали.</p>
      </header>

      <div className="home-grid">
        <section className="upload-panel" aria-labelledby="upload-title">
          <div className="upload-panel-header"><div><span className="section-index">01</span><h2 id="upload-title">Новый анализ</h2></div><span className="upload-format">MOV / {service ? `до ${formatFileSize(service.maxUploadBytes)}` : "один файл"}</span></div>

          <div className="upload-dropzone" data-dragging={dragging} data-has-file={!!selection}
            onDragEnter={(event) => { event.preventDefault(); if (sending.current || !event.dataTransfer.types.includes("Files")) return; dragDepth.current += 1; setDragging(true); }}
            onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = sending.current ? "none" : "copy"; }}
            onDragLeave={(event) => { event.preventDefault(); dragDepth.current = Math.max(0, dragDepth.current - 1); if (dragDepth.current === 0) setDragging(false); }}
            onDrop={drop}>
            <input ref={inputRef} type="file" accept=".mov,video/quicktime" className="sr-only" id="video-upload" tabIndex={-1} aria-label="Выбрать видео MOV" disabled={uploading} onChange={(event) => { if (event.target.files?.length) selectFiles(event.target.files); event.target.value = ""; }} />
            {selection ? <>
              <div className="upload-file">
                <Icon name="film" /><div className="upload-file-info"><strong className="upload-file-name">{selection.file.name}</strong><span className="upload-file-meta">{formatFileSize(selection.file.size)}{selection.duration !== null ? ` / ${durationLabel(selection.duration)}` : ""} / исходный файл</span></div>
                <button type="button" className="icon-button" disabled={uploading} aria-label="Убрать выбранное видео" onClick={() => { clearSelection(); requestAnimationFrame(() => chooseButtonRef.current?.focus()); }}><Icon name="close" /></button>
              </div>
              {!selection.previewFailed ? <div className="upload-preview">
                <video key={selection.url} src={selection.url} controls playsInline preload="metadata" aria-label={`Предпросмотр ${selection.file.name}`}
                  onLoadedMetadata={(event) => { const duration = event.currentTarget.duration; if (Number.isFinite(duration)) setSelection((current) => current?.url === selection.url ? { ...current, duration } : current); }}
                  onError={() => setSelection((current) => current?.url === selection.url ? { ...current, previewFailed: true } : current)} />
                <span className="upload-preview-note">Локальный предпросмотр</span>
              </div> : <div className="upload-preview-fallback"><Icon name="film" /><p>Этот кодек не воспроизводится в браузере.</p><span>Видео можно отправить на анализ. Найденные кадры будут доступны в результате.</span></div>}
              <button type="button" className="text-button upload-replace" disabled={uploading} onClick={() => inputRef.current?.click()}><Icon name="refresh" /> Выбрать другой файл</button>
            </> : <>
              <div className="upload-symbol" aria-hidden="true"><span className="upload-symbol-corner" /><Icon name="upload" /><span className="upload-symbol-corner" /></div>
              <p className="upload-drop-title">{dragging ? "Отпустите видео здесь" : "Всё начинается с записи"}</p>
              <p id="upload-help" className="upload-drop-copy">Перетащите .mov сюда или выберите файл.<br />Важные моменты соберём на одной временной шкале.</p>
              <button ref={chooseButtonRef} type="button" className="button button-primary" onClick={() => inputRef.current?.click()} aria-describedby="upload-help"><Icon name="plus" /> Выбрать видео</button>
              <span className="upload-local-note">Отправка начнётся только после вашего подтверждения</span>
            </>}
          </div>

          {error && <div className="upload-error" role="alert"><Icon name="alert" /><p>{error}</p></div>}

          {selection && <div className="upload-actions">
            {uploading ? <div className="upload-progress">
              <div><span>{progress === 100 ? "Ожидаем подтверждение приёма" : "Передаём видео"}</span><strong aria-hidden="true">{progress === null ? "…" : `${progress}%`}</strong></div>
              <div className="upload-progress-track" role="progressbar" aria-label="Передача видео" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress ?? undefined} data-indeterminate={progress === null}><span className="upload-progress-fill" style={{ width: progress === null ? "35%" : `${progress}%` }} /></div>
              <p role="status">{progress === 100 ? "Файл передан в приложение. Анализ начнётся после подтверждения сервиса." : "Прогресс передачи файла, без учёта анализа."}</p>
              <button type="button" className="text-button" onClick={() => uploadController.current?.abort()}>Прервать передачу</button>
            </div> : <><p>{mock ? "Демоанализатор вернёт синтетический результат. Содержимое видео не распознаётся." : ready ? "Видео будет передано сервису для анализа." : "Предпросмотр доступен. Для отправки нужен подключённый анализатор."}</p><button type="button" className="button button-primary" disabled={!ready} onClick={startUpload}>{mock ? "Запустить демоанализ" : "Начать анализ"}<Icon name="arrow-right" /></button></>}
          </div>}

          <div className="service-state" data-state={checking ? "checking" : ready ? mock ? "demo" : "ready" : "unavailable"}>
            <span className="status-dot" /><span role="status">{serviceLabel}</span><button className="icon-button" type="button" onClick={checkService} disabled={checking || uploading} aria-label="Проверить подключение к сервису"><Icon name="refresh" /></button>
          </div>
        </section>

        <aside className="process-rail" aria-labelledby="process-title">
          <p className="eyebrow">ОТ ЗАПИСИ К РЕШЕНИЮ</p><h2 id="process-title" className="process-title">Каждый момент.<br />В своём контексте.</h2>
          <ol className="process-steps">
            <li className="process-step"><span className="process-step-number">01</span><div className="process-step-copy"><h3>Загрузите запись</h3><p>Один файл — одно задание. Выберите видео и запустите анализ.</p></div></li>
            <li className="process-step"><span className="process-step-number">02</span><div className="process-step-copy"><h3>Перейдите к моменту</h3><p>События, таймкоды и кадры собраны рядом с видеозаписью.</p></div></li>
            <li className="process-step"><span className="process-step-number">03</span><div className="process-step-copy"><h3>Проверьте детали</h3><p>Сопоставьте результат с контекстом. Окончательное решение — за специалистом.</p></div></li>
          </ol>
          <div className="rail-footer"><Icon name="eye" /><p>Внимание к деталям.<br /><strong>Решение — за вами.</strong></p></div>
          {service?.demoEnabled && <Link href="/review/demo-violation" className="button button-secondary demo-entry">Открыть демопример <Icon name="arrow-right" /><span>Синтетические события и кадры</span></Link>}
        </aside>
      </div>
      <RecentJobs />
    </main>
  );
}
