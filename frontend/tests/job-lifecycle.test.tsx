// @vitest-environment jsdom

import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAnalysisJob } from "../features/analysis-job/use-analysis-job";
import { ApiError, type Job } from "../lib/contracts";
import { fetchJob } from "../lib/client";

vi.mock("../lib/client", () => ({ fetchJob: vi.fn() }));

const fetchMock = vi.mocked(fetchJob);
const processing = (id = "job-1"): Job => ({ job_id: id, analysis_provider: "test", status: "processing" });
const done = (id = "job-1"): Job => ({ job_id: id, analysis_provider: "test", status: "done", verdict: "no_violation", violations: [] });

async function tick(ms = 0) {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms); });
}

function setHidden(hidden: boolean) {
  Object.defineProperty(document, "hidden", { configurable: true, value: hidden });
  act(() => { document.dispatchEvent(new Event("visibilitychange")); });
}

describe("analysis job lifecycle", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-01T12:00:00Z"));
    fetchMock.mockReset();
    Object.defineProperty(document, "hidden", { configurable: true, value: false });
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("polls processing to done and stops after a terminal response", async () => {
    fetchMock.mockResolvedValueOnce(processing()).mockResolvedValue(done());
    const { result } = renderHook(() => useAnalysisJob("job-1"));
    await tick();
    expect(result.current.job?.status).toBe("processing");
    await tick(2_000);
    expect(result.current.job?.status).toBe("done");
    expect(result.current.elapsedSec).toBe(2);
    await tick(10 * 60_000);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.current.error).toBeNull();
    expect(result.current.paused).toBe(false);
  });

  it("stops on a failed job while preserving its backend error", async () => {
    fetchMock.mockResolvedValue({ job_id: "job-1", analysis_provider: "test", status: "failed", error: { code: "decode_failed", message: "Повреждено видео" } });
    const { result } = renderHook(() => useAnalysisJob("job-1"));
    await tick();
    await tick(10_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.current.job).toMatchObject({ status: "failed", error: { code: "decode_failed" } });
    expect(result.current.error).toBeNull();
  });

  it("retains processing during network errors and recovers with capped backoff", async () => {
    fetchMock.mockResolvedValueOnce(processing()).mockRejectedValue(new ApiError("Offline", 0, "network_error"));
    const { result } = renderHook(() => useAnalysisJob("job-1"));
    await tick();
    await tick(2_000);
    expect(result.current.job?.status).toBe("processing");
    expect(result.current.error?.code).toBe("network_error");
    expect(result.current.paused).toBe(false);
    // Subsequent retries wait 2, 4, 8, then at most 15 seconds.
    for (const delay of [2_000, 4_000, 8_000, 15_000]) {
      const before = fetchMock.mock.calls.length;
      await tick(delay - 1);
      expect(fetchMock).toHaveBeenCalledTimes(before);
      await tick(1);
      expect(fetchMock).toHaveBeenCalledTimes(before + 1);
    }
    fetchMock.mockResolvedValue(done());
    await tick(15_000);
    expect(result.current.job?.status).toBe("done");
    expect(result.current.error).toBeNull();
  });

  it.each([401, 403, 404, 410])("pauses after HTTP %i until an explicit retry", async (status) => {
    fetchMock.mockRejectedValueOnce(new ApiError("Unavailable", status)).mockResolvedValue(done());
    const { result } = renderHook(() => useAnalysisJob("job-1"));
    await tick();
    expect(result.current.paused).toBe(true);
    expect(result.current.job).toBeNull();
    await tick(20_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    act(() => result.current.retry());
    await tick();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.current.job?.status).toBe("done");
    expect(result.current.paused).toBe(false);
  });

  it("pauses hidden pages, aborts their request, and resumes when visible", async () => {
    let resolveOld!: (value: Job) => void;
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; })).mockResolvedValue(done());
    const { result } = renderHook(() => useAnalysisJob("job-1"));
    await tick();
    const oldSignal = fetchMock.mock.calls[0][1];
    setHidden(true);
    expect(oldSignal?.aborted).toBe(true);
    expect(result.current.paused).toBe(true);
    await tick(10_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await act(async () => resolveOld(processing()));
    expect(result.current.job).toBeNull();
    setHidden(false);
    await tick();
    expect(result.current.job?.status).toBe("done");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("never overlaps polling while a response is outstanding", async () => {
    let resolveRequest!: (value: Job) => void;
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => { resolveRequest = resolve; })).mockResolvedValue(done());
    const { result } = renderHook(() => useAnalysisJob("job-1"));
    await tick();
    await tick(30_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await act(async () => resolveRequest(processing()));
    await tick(1_999);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await tick(1);
    expect(result.current.job?.status).toBe("done");
  });

  it("discards old-job responses even when the transport ignores abort", async () => {
    let resolveOld!: (value: Job) => void;
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; })).mockResolvedValue(done("job-2"));
    const { result, rerender } = renderHook(({ id }) => useAnalysisJob(id), { initialProps: { id: "job-1" } });
    await tick();
    const oldSignal = fetchMock.mock.calls[0][1];
    rerender({ id: "job-2" });
    expect(result.current.job).toBeNull();
    expect(oldSignal?.aborted).toBe(true);
    await tick();
    expect(result.current.job?.job_id).toBe("job-2");
    await act(async () => resolveOld(done("job-1")));
    expect(result.current.job?.job_id).toBe("job-2");
    await tick(10_000);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("aborts outstanding requests and removes future work on unmount", async () => {
    let resolveRequest!: (value: Job) => void;
    fetchMock.mockImplementation(() => new Promise((resolve) => { resolveRequest = resolve; }));
    const { unmount } = renderHook(() => useAnalysisJob("job-1"));
    await tick();
    const signal = fetchMock.mock.calls[0][1];
    unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => resolveRequest(processing()));
    await tick(10_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("pauses the frontend budget without failing the job; retry renews budget and keeps demo time", async () => {
    fetchMock.mockResolvedValue(processing("demo-job"));
    const { result } = renderHook(() => useAnalysisJob("demo-job"));
    await tick();
    const demoStartedAt = fetchMock.mock.calls[0][2];
    await tick(5 * 60_000);
    expect(result.current.job?.status).toBe("processing");
    expect(result.current.error?.code).toBe("polling_timeout");
    expect(result.current.paused).toBe(true);
    const before = fetchMock.mock.calls.length;
    await tick(10_000);
    expect(fetchMock).toHaveBeenCalledTimes(before);
    act(() => result.current.retry());
    await tick();
    expect(fetchMock).toHaveBeenCalledTimes(before + 1);
    expect(result.current.paused).toBe(false);
    expect(result.current.error).toBeNull();
    expect(fetchMock.mock.calls.at(-1)?.[2]).toBe(demoStartedAt);
    expect(result.current.elapsedSec).toBeGreaterThanOrEqual(300);
    await tick(2_000);
    expect(fetchMock).toHaveBeenCalledTimes(before + 2);
  });
});
