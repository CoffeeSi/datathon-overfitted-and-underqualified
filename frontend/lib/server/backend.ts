import "server-only";
import { ApiError } from "../contracts";

export function serverConfig() {
  const backend = new URL(process.env.BACKEND_URL || "http://127.0.0.1:8000");
  if (!["http:", "https:"].includes(backend.protocol) || backend.username || backend.password || backend.search || backend.hash || backend.pathname !== "/") {
    throw new ApiError("Проверьте настройки подключения приложения к сервису.", 503, "invalid_configuration");
  }
  const megabytes = Number(process.env.MAX_UPLOAD_MB || "100");
  if (!Number.isSafeInteger(megabytes) || megabytes < 1 || megabytes > 10_240) {
    throw new ApiError("Проверьте лимит загрузки в настройках приложения.", 503, "invalid_configuration");
  }
  return {
    backendOrigin: backend.origin,
    apiKey: process.env.FORTEBANK_API_KEY || null,
    maxUploadBytes: megabytes * 1024 * 1024,
    demoEnabled: process.env.DEMO_ENABLED === "true",
  };
}

export function privateJson(value: unknown, status = 200): Response {
  return Response.json(value, { status, headers: {
    "Cache-Control": "private, no-store, max-age=0",
    "X-Content-Type-Options": "nosniff",
  } });
}

/** Synthetic providers are permitted only in explicitly enabled demo sessions. */
export function assertAnalysisProvider(provider: unknown): void {
  if (typeof provider !== "string" || !provider.trim()) {
    throw new ApiError("Анализатор ещё не подключён.", 503, "provider_unavailable");
  }
  if (!serverConfig().demoEnabled && (provider === "mock" || provider === "synthetic-demo")) {
    throw new ApiError("Подключён тестовый анализатор. Запустите backend с настоящей ML-моделью.", 503, "synthetic_provider_disabled");
  }
}

export function errorResponse(error: unknown): Response {
  const known = error instanceof ApiError ? error : new ApiError("Сервис временно недоступен. Попробуйте ещё раз.", 503, "service_unavailable");
  return privateJson({ error: { code: known.code ?? "request_failed", message: known.message } }, known.status || 503);
}

function upstreamError(status: number, payload: unknown): ApiError {
  const detail = payload && typeof payload === "object" && "detail" in payload && typeof payload.detail === "string" ? payload.detail : "";
  const known: Record<string, [string, string]> = {
    "API key is not configured": ["key_not_configured", "Доступ к анализу ещё не настроен на сервере."],
    "Invalid API key": ["unauthorized", "Сервис отклонил ключ доступа. Проверьте серверные настройки."],
    "A local video analysis provider is not configured": ["provider_unavailable", "Анализатор ещё не подключён. Видео пока нельзя отправить на анализ."],
    "The analysis queue is full; retry shortly": ["queue_full", "Очередь анализа заполнена. Попробуйте отправить видео позже."],
    "Upload exceeds the configured size limit": ["upload_too_large", "Файл превышает допустимый размер сервиса."],
    "Only .mov uploads are supported": ["unsupported_format", "Сервис принимает видео в формате .mov."],
    "A video file is required in the 'file' field": ["missing_file", "Выберите один видеофайл для загрузки."],
    "The uploaded .mov file is unreadable": ["invalid_video", "Сервис не смог прочитать видео. Проверьте файл."],
    "Job not found": ["job_not_found", "Задание не найдено. Возможно, срок его хранения истёк."],
  };
  const fallback: Record<number, [string, string]> = {
    400: ["invalid_request", "Сервис не принял запрос. Проверьте выбранный файл."],
    401: ["unauthorized", "Сервис отклонил ключ доступа. Проверьте серверные настройки."],
    403: ["forbidden", "Сервис не разрешил этот запрос."],
    404: ["job_not_found", "Задание не найдено. Возможно, срок его хранения истёк."],
    413: ["upload_too_large", "Файл превышает допустимый размер сервиса."],
    415: ["unsupported_format", "Сервис принимает видео в формате .mov."],
    422: ["invalid_upload", "Сервис не смог принять файл. Проверьте формат и содержимое видео."],
    503: ["service_unavailable", "Анализ сейчас недоступен. Попробуйте позже."],
    504: ["upstream_timeout", "Сервис не ответил вовремя. Задание могло продолжить работу."],
  };
  const mapped = Object.hasOwn(known, detail) ? known[detail] : undefined;
  const [code, message] = mapped ?? fallback[status] ?? ["upstream_error", "Сервис не смог обработать запрос. Попробуйте ещё раз."];
  return new ApiError(message, status >= 400 && status <= 599 ? status : 502, code);
}

type BackendOptions = {
  signal?: AbortSignal;
  method?: "GET" | "POST";
  body?: ReadableStream<Uint8Array>;
  contentType?: string;
  timeoutMs?: number;
  authenticated?: boolean;
};

export async function backendJson(path: string, options: BackendOptions = {}): Promise<unknown> {
  if (!["/healthz", "/readyz", "/api/v1/analyze"].includes(path) && !/^\/api\/v1\/ui\/jobs\/[a-zA-Z0-9_-]{1,80}$/.test(path)) {
    throw new ApiError("Некорректный адрес запроса.", 400, "invalid_endpoint");
  }
  const config = serverConfig();
  if (options.authenticated && !config.apiKey) throw new ApiError("Ключ доступа к анализу ещё не настроен на сервере приложения.", 503, "key_not_configured");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? 12_000);
  const abort = () => controller.abort();
  options.signal?.addEventListener("abort", abort, { once: true });
  if (options.signal?.aborted) controller.abort();
  const headers = new Headers({ Accept: "application/json" });
  if (options.authenticated && config.apiKey) headers.set("X-API-Key", config.apiKey);
  if (options.contentType) headers.set("Content-Type", options.contentType);
  try {
    const init: RequestInit & { duplex?: "half" } = {
      method: options.method ?? "GET", headers, cache: "no-store", redirect: "manual", signal: controller.signal,
    };
    if (options.body) { init.body = options.body; init.duplex = "half"; }
    const response = await fetch(`${config.backendOrigin}${path}`, init);
    const payload: unknown = await response.json().catch(() => null);
    if (controller.signal.aborted) throw new DOMException("Upstream timed out", "AbortError");
    if (!response.ok) throw upstreamError(response.status, payload);
    if (payload === null) throw new ApiError("Сервис вернул некорректный ответ.", 502, "invalid_response");
    return payload;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (controller.signal.aborted) throw new ApiError("Сервис не ответил вовремя. Приём файла или выполнение задания могли продолжиться; автоматического повтора не будет.", 504, "upstream_timeout");
    throw new ApiError("Нет соединения с сервисом анализа. Проверьте его запуск.", 503, "backend_unavailable");
  } finally {
    clearTimeout(timeout);
    options.signal?.removeEventListener("abort", abort);
  }
}

export function safeJobError(code: string): string {
  const messages: Record<string, string> = {
    analyzer_not_configured: "Анализатор ещё не подключён.",
    analysis_timeout: "Анализ превысил допустимое время. Попробуйте более короткий ролик.",
    analysis_failed: "Сервис не смог проанализировать это видео.",
    job_deadline_exceeded: "Истекло время ожидания обработки задания.",
  };
  return Object.hasOwn(messages, code) ? messages[code] : "Обработка задания завершилась ошибкой. Можно отправить видео заново.";
}
