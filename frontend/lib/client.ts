import { ApiError, parseAccepted, parseJob, parseService, type Job, type ServiceStatus } from "./contracts";

export { ApiError } from "./contracts";

function errorFromPayload(status: number, value: unknown): ApiError {
  if (value && typeof value === "object" && "error" in value) {
    const error = value.error;
    if (error && typeof error === "object" && "message" in error && typeof error.message === "string") {
      const code = "code" in error && typeof error.code === "string" ? error.code : undefined;
      return new ApiError(error.message, status, code);
    }
  }
  return new ApiError("Не удалось получить ответ сервиса. Попробуйте ещё раз.", status, "request_failed");
}

async function requestJson(url: string, signal?: AbortSignal): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(url, { signal, cache: "no-store", credentials: "same-origin", headers: { Accept: "application/json" } });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new ApiError("Нет связи с сервисом. Проверьте подключение и возобновите проверку.", 0, "network_error");
  }
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) throw errorFromPayload(response.status, payload);
  return payload;
}

export async function fetchService(signal?: AbortSignal): Promise<ServiceStatus> {
  return parseService(await requestJson("/api/review/service", signal));
}

export async function fetchJob(jobId: string, signal?: AbortSignal, demoStartedAt?: number): Promise<Job> {
  const query = jobId.startsWith("demo-") && demoStartedAt !== undefined
    ? `?demoStartedAt=${encodeURIComponent(demoStartedAt)}` : "";
  const job = parseJob(await requestJson(`/api/review/jobs/${encodeURIComponent(jobId)}${query}`, signal));
  if (job.job_id !== jobId) throw new ApiError("Сервис вернул другое задание. Обновите проверку.", 502, "invalid_response");
  return job;
}

/** XHR exposes bytes sent to this BFF. It does not measure server analysis progress. */
export function uploadVideo(file: File, onProgress: (percent: number | null) => void, signal?: AbortSignal): Promise<{ job_id: string }> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new DOMException("Upload aborted", "AbortError")); return; }
    const xhr = new XMLHttpRequest();
    const abort = () => xhr.abort();
    const cleanup = () => signal?.removeEventListener("abort", abort);
    xhr.open("POST", "/api/review/analyze");
    xhr.timeout = 185_000;
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => onProgress(event.lengthComputable ? Math.round(event.loaded / event.total * 100) : null);
    xhr.onload = () => {
      cleanup();
      if (xhr.status !== 202) { reject(errorFromPayload(xhr.status, xhr.response)); return; }
      try { resolve(parseAccepted(xhr.response)); } catch (error) { reject(error); }
    };
    xhr.onerror = () => { cleanup(); reject(new ApiError("Связь прервалась. Приём файла не подтверждён; автоматического повтора не будет.", 0, "network_error")); };
    xhr.ontimeout = () => { cleanup(); reject(new ApiError("Ответ о приёме файла не получен вовремя. Сервис мог принять его; повторная отправка создаст новое задание.", 504, "upload_timeout")); };
    xhr.onabort = () => { cleanup(); reject(new DOMException("Upload aborted", "AbortError")); };
    signal?.addEventListener("abort", abort, { once: true });
    const form = new FormData();
    form.append("file", file);
    onProgress(0);
    xhr.send(form);
  });
}
