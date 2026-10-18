import React from "react";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { ActionContext, UnresolvedActions } from "../features/review/action-context";
import { parseJob, type Finding } from "../lib/contracts";

const finding: Finding = {
  type: "safe_key_left_in_lock", timecode_sec: 12, description: "Действие для проверки",
  evidence_frames: [{ timestamp_sec: 12, image_base64: "/9j/2Q==" }],
  source_links: [{ event_type: "safe_key_left_in_lock", source: "№231", locator: "п. 80", title: "Правила",
    url: "https://adilet.zan.kz/rus/docs/V1900019680" }],
  applicability: ["Нужны внутренние документы"], video_cannot_establish: ["Доступ вне кадра неизвестен"],
  type_confidence: .8, binary_score: .9,
};
afterEach(cleanup);
describe("typed action UI", () => {
  it("preserves actual rule links and unresolved windows through the BFF parser", () => {
    const job = parseJob({ job_id: "test", analysis_provider: "local-r3d18-actions", status: "done", verdict: "violation",
      violations: [finding], unresolved_candidates: [{ timestamp_sec: 20, binary_score: .8, reason: "unknown" }] });
    expect(job).toMatchObject({ violations: [finding], unresolved_candidates: [{ reason: "unknown" }] });
  });
  it("renders server-provided references, applicability and limitations", () => {
    render(<ActionContext finding={finding} />);
    expect(screen.getByRole("link", { name: "№231 · п. 80" }).getAttribute("href")).toBe(finding.source_links![0].url);
    expect(screen.getByText("Доступ вне кадра неизвестен")).toBeTruthy();
    expect(screen.getByText(/не устанавливает юридическое/)).toBeTruthy();
  });
  it("does not invent rules and distinguishes unknown candidates from clean video", () => {
    render(<><ActionContext finding={{ ...finding, source_links: undefined }} /><UnresolvedActions candidates={[{ timestamp_sec: 20, binary_score: .9, reason: "unknown" }]} /></>);
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.getByText("Нужна ручная проверка")).toBeTruthy();
  });
  it.each(["javascript:alert(1)", "https://user:secret@example.com/"])("rejects unsafe link %s", (url) => {
    expect(() => parseJob({ job_id: "test", analysis_provider: "local", status: "done", verdict: "violation",
      violations: [{ ...finding, source_links: [{ ...finding.source_links![0], url }] }] })).toThrow();
  });
  it("rejects references for a different type and clean verdicts with unresolved candidates", () => {
    expect(() => parseJob({ job_id: "test", analysis_provider: "local", status: "done", verdict: "violation",
      violations: [{ ...finding, source_links: [{ ...finding.source_links![0], event_type: "different" }] }] })).toThrow();
    expect(() => parseJob({ job_id: "test", analysis_provider: "local", status: "done", verdict: "no_violation",
      violations: [], unresolved_candidates: [{ timestamp_sec: 20, binary_score: .8, reason: "unknown" }] })).toThrow();
  });
});
