import { ApiError } from "../contracts";

export const MULTIPART_OVERHEAD_BYTES = 1024 * 1024;
export const JOB_ID_PATTERN = /^[a-zA-Z0-9_-]{1,80}$/;

export function assertJobId(jobId: string): void {
  if (!JOB_ID_PATTERN.test(jobId)) throw new ApiError("Некорректный идентификатор задания.", 400, "invalid_job_id");
}

export function validateUploadRequest(request: Request, maxUploadBytes: number): void {
  const origin = request.headers.get("origin");
  const requestUrl = new URL(request.url);
  // Next's Node adapter can normalize Request.url to localhost even when the browser
  // uses 127.0.0.1. Host is the actual destination and cannot be set by browser JS.
  // Do not trust arbitrary X-Forwarded-Host values supplied by callers.
  const destination = new URL(`${requestUrl.protocol}//${request.headers.get("host") || requestUrl.host}`);
  if (origin !== destination.origin || request.headers.get("sec-fetch-site") === "cross-site") {
    throw new ApiError("Загрузите видео со страницы приложения.", 403, "invalid_origin");
  }
  const type = request.headers.get("content-type") ?? "";
  if (!/^multipart\/form-data;\s*boundary=(?:"[^"\r\n]{1,70}"|[^;\s\r\n]{1,70})\s*$/i.test(type)) {
    throw new ApiError("Нужен видеофайл в поле file формы загрузки.", 422, "invalid_multipart");
  }
  const length = request.headers.get("content-length");
  if (length !== null && (!/^\d+$/.test(length) || !Number.isSafeInteger(Number(length)))) {
    throw new ApiError("Некорректный размер запроса.", 400, "invalid_content_length");
  }
  if (length !== null && Number(length) > maxUploadBytes + MULTIPART_OVERHEAD_BYTES) {
    throw new ApiError("Файл превышает допустимый размер.", 413, "upload_too_large");
  }
  if (!request.body) throw new ApiError("Выберите видеофайл.", 422, "missing_file");
}

/** Applies backpressure and checks streamed/chunked requests even without Content-Length. */
export function limitUploadStream(body: ReadableStream<Uint8Array>, limit: number, onExceeded: () => void): ReadableStream<Uint8Array> {
  let received = 0;
  return body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      received += chunk.byteLength;
      if (received > limit) {
        onExceeded();
        controller.error(new ApiError("Файл превышает допустимый размер.", 413, "upload_too_large"));
        return;
      }
      controller.enqueue(chunk);
    },
  }));
}
