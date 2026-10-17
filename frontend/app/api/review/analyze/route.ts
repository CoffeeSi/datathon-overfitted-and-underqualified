import { ApiError, parseAccepted } from "@/lib/contracts";
import { backendJson, errorResponse, privateJson, serverConfig } from "@/lib/server/backend";
import { limitUploadStream, MULTIPART_OVERHEAD_BYTES, validateUploadRequest } from "@/lib/server/policy";

export const runtime = "nodejs";

export async function POST(request: Request) {
  let limitExceeded = false;
  try {
    const config = serverConfig();
    validateUploadRequest(request, config.maxUploadBytes);
    // Stream multipart unchanged. FastAPI owns the single-file, filename, container and exact file-size checks.
    const body = limitUploadStream(request.body!, config.maxUploadBytes + MULTIPART_OVERHEAD_BYTES, () => { limitExceeded = true; });
    const accepted = parseAccepted(await backendJson("/api/v1/analyze", {
      method: "POST", authenticated: true, signal: request.signal,
      body, contentType: request.headers.get("content-type")!, timeoutMs: 180_000,
    }));
    return privateJson(accepted, 202);
  } catch (error) {
    return errorResponse(limitExceeded ? new ApiError("Файл превышает допустимый размер.", 413, "upload_too_large") : error);
  }
}
