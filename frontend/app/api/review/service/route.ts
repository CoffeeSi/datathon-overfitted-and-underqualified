import { backendJson, errorResponse, privateJson, serverConfig } from "@/lib/server/backend";
import type { ServiceStatus } from "@/lib/contracts";

export const runtime = "nodejs";

export async function GET(request: Request) {
  try {
    const config = serverConfig();
    const [health, readiness] = await Promise.allSettled([
      backendJson("/healthz", { signal: request.signal, timeoutMs: 4_000 }),
      backendJson("/readyz", { signal: request.signal, timeoutMs: 4_000 }),
    ]);
    const isOnline = health.status === "fulfilled" && !!health.value && typeof health.value === "object" && "status" in health.value && health.value.status === "ok";
    const readyData = readiness.status === "fulfilled" && readiness.value && typeof readiness.value === "object" ? readiness.value as Record<string, unknown> : null;
    const provider = readyData?.status === "ready" && typeof readyData.analysis_provider === "string" && readyData.analysis_provider.trim() ? readyData.analysis_provider : null;
    const result: ServiceStatus = {
      health: isOnline ? "online" : "offline",
      ready: isOnline && provider !== null && config.apiKey !== null,
      provider, maxUploadBytes: config.maxUploadBytes, demoEnabled: config.demoEnabled,
    };
    return privateJson(result);
  } catch (error) { return errorResponse(error); }
}
