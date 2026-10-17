import { describe, expect, it } from "vitest";
import { ApiError, parseAccepted, parseJob, parseService } from "../lib/contracts";

const base = { job_id: "a-job", analysis_provider: "new-provider" };
const finding = {
  type: "a_new_type", timecode_sec: 28.4, description: "Новое наблюдение",
  evidence_frames: [{ timestamp_sec: 28.4, image_base64: "/9j/2Q==" }],
};

describe("backend wire contracts", () => {
  it("preserves unknown finding types and fractional timestamps, strips extra fields", () => {
    const job = parseJob({ ...base, status: "done", verdict: "violation", violations: [{ ...finding, confidence: 0.99 }], debug: "private" });
    expect(job).toEqual({ ...base, status: "done", verdict: "violation", violations: [finding] });
  });

  it("accepts processing and future error codes without assuming model progress", () => {
    expect(parseJob({ ...base, status: "processing", progress: 30 })).toEqual({ ...base, status: "processing" });
    expect(parseJob({ ...base, status: "failed", error: { code: "new_error", message: "Safe error" } })).toMatchObject({ status: "failed", error: { code: "new_error" } });
  });

  it.each(["no_violation", "insufficient_data"])("accepts empty %s results", (verdict) => {
    expect(parseJob({ ...base, status: "done", verdict, violations: [] })).toMatchObject({ verdict, violations: [] });
  });

  it.each([
    { ...base, status: "done", verdict: "violation", violations: [] },
    { ...base, status: "done", verdict: "no_violation", violations: [finding] },
    { ...base, status: "done", verdict: "insufficient_data", violations: [finding] },
    { ...base, status: "queued" },
    { ...base, status: "failed" },
    { job_id: "a-job", status: "processing" },
  ])("rejects contradictory or partial results", (payload) => {
    expect(() => parseJob(payload)).toThrow(ApiError);
  });

  it.each([NaN, Infinity, -1, "12"])("rejects invalid timestamp %s", (timecode_sec) => {
    expect(() => parseJob({ ...base, status: "done", verdict: "violation", violations: [{ ...finding, timecode_sec }] })).toThrow(ApiError);
  });

  it.each([[], [1, 2, 3, 4].map(() => finding.evidence_frames[0]), [{ timestamp_sec: 1, image_base64: "data:image/svg+xml,<svg>" }]])("rejects invalid evidence transport", (evidence_frames) => {
    expect(() => parseJob({ ...base, status: "done", verdict: "violation", violations: [{ ...finding, evidence_frames }] })).toThrow(ApiError);
  });

  it("requires a positive service limit and consistent readiness", () => {
    const service = { health: "online", ready: false, provider: null, maxUploadBytes: 1024, demoEnabled: false };
    expect(parseService(service)).toEqual(service);
    expect(() => parseService({ ...service, maxUploadBytes: 0 })).toThrow(ApiError);
    expect(() => parseService({ ...service, ready: true })).toThrow(ApiError);
    expect(() => parseService({ ...service, health: "offline", ready: true, provider: "mock" })).toThrow(ApiError);
  });

  it("does not invent processing status in the upload response", () => {
    expect(parseAccepted({ job_id: "one", secret: "discard" })).toEqual({ job_id: "one" });
    expect(() => parseAccepted({})).toThrow(ApiError);
  });
});
