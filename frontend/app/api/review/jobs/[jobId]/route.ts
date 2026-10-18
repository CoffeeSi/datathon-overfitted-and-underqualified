import { ApiError, parseJob } from "@/lib/contracts";
import { assertAnalysisProvider, backendJson, errorResponse, privateJson, safeJobError, serverConfig } from "@/lib/server/backend";
import { assertJobId } from "@/lib/server/policy";
import { demoJob } from "@/lib/server/demo";

export const runtime = "nodejs";

export async function GET(request: Request, context: { params: Promise<{ jobId: string }> }) {
  try {
    const { jobId } = await context.params;
    assertJobId(jobId);
    if (jobId.startsWith("demo-")) {
      if (!serverConfig().demoEnabled) throw new ApiError("Демонстрационный режим выключен.", 404, "job_not_found");
      return privateJson(await demoJob(jobId, new URL(request.url).searchParams.get("demoStartedAt")));
    }
    const job = parseJob(await backendJson(`/api/v1/ui/jobs/${jobId}`, { authenticated: true, signal: request.signal }));
    if (!serverConfig().demoEnabled && (job.analysis_provider === "mock" || job.analysis_provider === "synthetic-demo")) {
      throw new ApiError("Это задание создано тестовым анализатором. Загрузите видео заново с главной страницы для анализа настоящей моделью.", 410, "synthetic_result_disabled");
    }
    assertAnalysisProvider(job.analysis_provider);
    if (job.job_id !== jobId) throw new ApiError("Сервис вернул другое задание.", 502, "invalid_response");
    if (job.status === "failed") job.error.message = safeJobError(job.error.code);
    return privateJson(job);
  } catch (error) { return errorResponse(error); }
}
