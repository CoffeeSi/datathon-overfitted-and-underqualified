/** Wire contracts. Keep backend field names here; presentation models live in features. */
export interface EvidenceFrame {
  timestamp_sec: number;
  image_base64: string;
}

export interface Finding {
  type: string;
  timecode_sec: number;
  description: string;
  evidence_frames: EvidenceFrame[];
  source_links?: RuleReference[];
  applicability?: string[];
  video_cannot_establish?: string[];
  binary_score?: number;
  type_confidence?: number;
}

export interface RuleReference {
  event_type: string;
  source: string;
  locator: string;
  url: string;
  title: string;
}
export interface UnresolvedCandidate {
  timestamp_sec: number;
  binary_score: number;
  reason: "background" | "unknown";
}

export type Verdict = "violation" | "no_violation" | "insufficient_data";
type JobBase = { job_id: string; analysis_provider: string };
export type Job =
  | (JobBase & { status: "processing" })
  | (JobBase & { status: "done"; verdict: Verdict; violations: Finding[]; unresolved_candidates?: UnresolvedCandidate[]; recognition_scope?: string[] })
  | (JobBase & { status: "failed"; error: { code: string; message: string } });

export interface ServiceStatus {
  health: "online" | "offline";
  ready: boolean;
  provider: string | null;
  maxUploadBytes: number;
  demoEnabled: boolean;
}

export class ApiError extends Error {
  constructor(message: string, public readonly status = 0, public readonly code?: string) {
    super(message);
    this.name = "ApiError";
  }
}

function invalid(): never {
  throw new ApiError("Сервис вернул неполный или некорректный ответ. Попробуйте проверить задание ещё раз.", 502, "invalid_response");
}

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return invalid();
  return value as Record<string, unknown>;
}

function text(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) return invalid();
  return value;
}

function seconds(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return invalid();
  return value;
}

function frame(value: unknown): EvidenceFrame {
  const data = object(value);
  const image = text(data.image_base64);
  // Validate the transport and JPEG signature; the <img> error state handles a damaged image.
  if (image.length % 4 !== 0 || !image.startsWith("/9j/") || !/^[A-Za-z0-9+/]+={0,2}$/.test(image)) return invalid();
  return { timestamp_sec: seconds(data.timestamp_sec), image_base64: image };
}

function finding(value: unknown): Finding {
  const data = object(value);
  if (!Array.isArray(data.evidence_frames) || data.evidence_frames.length < 1 || data.evidence_frames.length > 3) return invalid();
  return {
    type: text(data.type),
    timecode_sec: seconds(data.timecode_sec),
    description: text(data.description),
    evidence_frames: data.evidence_frames.map(frame),
    ...(data.source_links !== undefined ? { source_links: references(data.source_links, text(data.type)) } : {}),
    ...(data.applicability !== undefined ? { applicability: texts(data.applicability) } : {}),
    ...(data.video_cannot_establish !== undefined ? { video_cannot_establish: texts(data.video_cannot_establish) } : {}),
    ...(data.binary_score !== undefined ? { binary_score: probability(data.binary_score) } : {}),
    ...(data.type_confidence !== undefined ? { type_confidence: probability(data.type_confidence) } : {}),
  };
}

function probability(value: unknown): number {
  const result = seconds(value);
  return result <= 1 ? result : invalid();
}
function texts(value: unknown): string[] {
  return Array.isArray(value) ? value.map(text) : invalid();
}
function references(value: unknown, eventType: string): RuleReference[] {
  if (!Array.isArray(value)) return invalid();
  return value.map((entry) => {
    const data = object(entry);
    if (data.event_type !== eventType) return invalid();
    let url: URL;
    try { url = new URL(text(data.url)); } catch { return invalid(); }
    if (url.protocol !== "https:" || url.username || url.password) return invalid();
    return { event_type: eventType, url: url.href, source: text(data.source), locator: text(data.locator), title: text(data.title) };
  });
}
function unresolved(value: unknown): UnresolvedCandidate[] {
  if (!Array.isArray(value)) return invalid();
  return value.map((entry) => {
    const data = object(entry);
    if (data.reason !== "background" && data.reason !== "unknown") return invalid();
    return { timestamp_sec: seconds(data.timestamp_sec), binary_score: probability(data.binary_score), reason: data.reason };
  });
}

export function parseJob(value: unknown): Job {
  const data = object(value);
  const base = { job_id: text(data.job_id), analysis_provider: text(data.analysis_provider) };
  if (data.status === "processing") return { ...base, status: "processing" };
  if (data.status === "failed") {
    const error = object(data.error);
    return { ...base, status: "failed", error: { code: text(error.code), message: text(error.message) } };
  }
  if (data.status !== "done" || !Array.isArray(data.violations)) return invalid();
  const verdict = data.verdict;
  if (verdict !== "violation" && verdict !== "no_violation" && verdict !== "insufficient_data") return invalid();
  const violations = data.violations.map(finding);
  if ((verdict === "violation") !== (violations.length > 0)) return invalid();
  const candidates = data.unresolved_candidates === undefined ? undefined : unresolved(data.unresolved_candidates);
  if (verdict === "no_violation" && candidates?.length) return invalid();
  return { ...base, status: "done", verdict, violations,
    ...(candidates !== undefined ? { unresolved_candidates: candidates } : {}),
    ...(data.recognition_scope !== undefined ? { recognition_scope: texts(data.recognition_scope) } : {}),
  };
}

export function parseService(value: unknown): ServiceStatus {
  const data = object(value);
  if (data.health !== "online" && data.health !== "offline") return invalid();
  if (typeof data.ready !== "boolean" || typeof data.demoEnabled !== "boolean") return invalid();
  if (data.provider !== null && (typeof data.provider !== "string" || !data.provider.trim())) return invalid();
  if (typeof data.maxUploadBytes !== "number" || !Number.isSafeInteger(data.maxUploadBytes) || data.maxUploadBytes <= 0) return invalid();
  if (data.ready && (data.health !== "online" || data.provider === null)) return invalid();
  return {
    health: data.health, ready: data.ready, provider: data.provider,
    maxUploadBytes: data.maxUploadBytes, demoEnabled: data.demoEnabled,
  };
}

export function parseAccepted(value: unknown): { job_id: string } {
  return { job_id: text(object(value).job_id) };
}
