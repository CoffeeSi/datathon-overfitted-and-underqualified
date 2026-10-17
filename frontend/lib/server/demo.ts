import "server-only";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { ApiError, type Job, type Finding } from "../contracts";

const ids = new Set(["demo-violation", "demo-no-violation", "demo-insufficient", "demo-failed", "demo-processing"]);

export async function demoJob(jobId: string, startedAt: string | null): Promise<Job> {
  if (!ids.has(jobId)) throw new ApiError("Демонстрационный сценарий не найден.", 404, "job_not_found");
  const base = { job_id: jobId, analysis_provider: "synthetic-demo" };
  if (jobId === "demo-failed") return { ...base, status: "failed", error: { code: "analysis_failed", message: "Демонстрация: обработка завершилась ошибкой. Реальное видео не отправлялось." } };
  if (jobId === "demo-no-violation") return { ...base, status: "done", verdict: "no_violation", violations: [] };
  if (jobId === "demo-insufficient") return { ...base, status: "done", verdict: "insufficient_data", violations: [] };
  const start = Number(startedAt);
  if (jobId === "demo-processing" && (!startedAt || !Number.isFinite(start) || Date.now() - start < 2_000)) {
    return { ...base, status: "processing" };
  }
  const images = await Promise.all([1, 2, 3].map(async (index) => (await readFile(path.join(process.cwd(), "public", "demo", `review-${index}.jpg`))).toString("base64")));
  const violations: Finding[] = [
    {
      type: "unsecured_accepted_cash", timecode_sec: 12,
      description: "Демонстрационный эпизод: на схеме видны купюры на открытой рабочей поверхности. Это синтетический пример интерфейса, а не результат анализа банковского видео.",
      evidence_frames: [{ timestamp_sec: 11.5, image_base64: images[0] }, { timestamp_sec: 12, image_base64: images[1] }, { timestamp_sec: 12.5, image_base64: images[2] }],
    },
    {
      type: "cashier_leaves_during_client_service", timecode_sec: 28,
      description: "Демонстрационный эпизод: сотрудник отходит от рабочего места. Соседние отметки показывают, как интерфейс помогает рассмотреть несколько событий в одном временном фрагменте. Для реального вывода необходимо установить, продолжалось ли обслуживание клиента, и проверить весь контекст записи. Иллюстрации созданы для демонстрации и не изображают реальный банк.",
      evidence_frames: [{ timestamp_sec: 28, image_base64: images[1] }, { timestamp_sec: 28.5, image_base64: images[2] }],
    },
    {
      type: "demo_context_review", timecode_sec: 28.4,
      description: "Синтетический пример нового типа события. Приложение сохраняет его описание и кадры, даже если для типа ещё нет русского названия.",
      evidence_frames: [{ timestamp_sec: 28.4, image_base64: images[2] }],
    },
  ];
  return { ...base, status: "done", verdict: "violation", violations };
}
