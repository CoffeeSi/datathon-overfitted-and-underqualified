// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
vi.mock("server-only", () => ({}));
import { GET as service } from "../app/api/review/service/route";
import { POST as analyze } from "../app/api/review/analyze/route";
import { GET as job } from "../app/api/review/jobs/[jobId]/route";

const fetchMock = vi.fn();
const response = (body: unknown) => Response.json(body);

describe("real analysis mode", () => {
  beforeEach(() => {
    vi.stubEnv("DEMO_ENABLED", "false");
    vi.stubEnv("FORTEBANK_API_KEY", "test-private-key");
    vi.stubEnv("BACKEND_URL", "http://127.0.0.1:8001");
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockReset();
  });
  afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); });

  it("does not report a mock backend as ready for real uploads", async () => {
    fetchMock.mockResolvedValueOnce(response({ status: "ok" }))
      .mockResolvedValueOnce(response({ status: "ready", analysis_provider: "mock" }));
    const result = await service(new Request("http://localhost:3000/api/review/service"));
    expect(await result.json()).toMatchObject({ ready: false, provider: "mock", demoEnabled: false });
  });

  it("blocks a mock provider before forwarding the video", async () => {
    fetchMock.mockResolvedValueOnce(response({ status: "ready", analysis_provider: "mock" }));
    const result = await analyze(new Request("http://localhost:3000/api/review/analyze", {
      method: "POST", body: "not forwarded", headers: {
        origin: "http://localhost:3000", "content-type": "multipart/form-data; boundary=test",
      },
    }));
    expect(result.status).toBe(503);
    expect(await result.json()).toMatchObject({ error: { code: "synthetic_provider_disabled" } });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("forwards a real upload and returns the actual backend job ID", async () => {
    fetchMock.mockResolvedValueOnce(response({ status: "ready", analysis_provider: "local-r3d18-actions" }))
      .mockResolvedValueOnce(response({ job_id: "actual-backend-job" }));
    const form = new FormData();
    form.set("file", new File(["synthetic test bytes"], "test.mov"));
    const result = await analyze(new Request("http://localhost:3000/api/review/analyze", {
      method: "POST", body: form, headers: { origin: "http://localhost:3000" },
    }));
    expect(result.status).toBe(202);
    expect(await result.json()).toEqual({ job_id: "actual-backend-job" });
    expect(fetchMock.mock.calls[1][0]).toBe("http://127.0.0.1:8001/api/v1/analyze");
    expect(fetchMock.mock.calls[1][1].headers.get("X-API-Key")).toBe("test-private-key");
  });

  it("refuses historical mock results in real mode", async () => {
    fetchMock.mockResolvedValueOnce(response({ job_id: "old-job", analysis_provider: "mock",
      status: "done", verdict: "no_violation", violations: [] }));
    const result = await job(new Request("http://localhost:3000/api/review/jobs/old-job"),
      { params: Promise.resolve({ jobId: "old-job" }) });
    expect(result.status).toBe(410);
    expect(await result.json()).toMatchObject({ error: { code: "synthetic_result_disabled" } });
  });

  it("retains explicitly enabled synthetic sessions for development tests", async () => {
    vi.stubEnv("DEMO_ENABLED", "true");
    fetchMock.mockResolvedValueOnce(response({ status: "ok" }))
      .mockResolvedValueOnce(response({ status: "ready", analysis_provider: "mock" }));
    const result = await service(new Request("http://localhost:3000/api/review/service"));
    expect(await result.json()).toMatchObject({ ready: true, provider: "mock", demoEnabled: true });
  });
});
