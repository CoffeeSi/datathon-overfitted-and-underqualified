// @vitest-environment node
import { describe, expect, it, vi } from "vitest";
import { assertJobId, limitUploadStream, MULTIPART_OVERHEAD_BYTES, validateUploadRequest } from "../lib/server/policy";

function upload(headers: Record<string, string> = {}) {
  return new Request("http://localhost:3000/api/review/analyze", {
    method: "POST", body: "multipart bytes", headers: {
      origin: "http://localhost:3000", "content-type": "multipart/form-data; boundary=example", ...headers,
    },
  });
}

describe("BFF request boundary", () => {
  it("rejects cross-origin and missing-origin uploads before forwarding", () => {
    expect(() => validateUploadRequest(upload({ origin: "https://elsewhere.example" }), 100)).toThrow(/страницы приложения/);
    const request = upload();
    request.headers.delete("origin");
    expect(() => validateUploadRequest(request, 100)).toThrow();
    expect(() => validateUploadRequest(upload({ "sec-fetch-site": "cross-site" }), 100)).toThrow();
  });

  it("allows same-origin multipart, rejects bad envelopes and advertised oversize", () => {
    expect(() => validateUploadRequest(upload(), 100)).not.toThrow();
    expect(() => validateUploadRequest(upload({ "content-type": "application/json" }), 100)).toThrow();
    expect(() => validateUploadRequest(upload({ "content-length": String(MULTIPART_OVERHEAD_BYTES + 101) }), 100)).toThrow(/размер/);
    expect(() => validateUploadRequest(upload({ "content-length": "-1" }), 100)).toThrow();
  });

  it("accepts the real Host when Next normalizes Request.url, without trusting forwarded hosts", () => {
    expect(() => validateUploadRequest(upload({ host: "127.0.0.1:3000", origin: "http://127.0.0.1:3000" }), 100)).not.toThrow();
    expect(() => validateUploadRequest(upload({ host: "127.0.0.1:3000", origin: "http://elsewhere.example", "x-forwarded-host": "elsewhere.example" }), 100)).toThrow();
  });

  it.each(["../admin", "a/b", "?x=y", "", "a".repeat(81)])("blocks an invalid job path: %s", (id) => {
    expect(() => assertJobId(id)).toThrow();
  });

  it("accepts opaque identifiers without guessing job status", () => {
    expect(() => assertJobId("39c3c607-214b-42aa-a003-02ce382ef08f")).not.toThrow();
    expect(() => assertJobId("demo-no-violation")).not.toThrow();
  });

  it("bounds streamed uploads even when no Content-Length is present", async () => {
    const onExceeded = vi.fn();
    const body = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(new Uint8Array(3)); controller.enqueue(new Uint8Array(3)); controller.close(); } });
    await expect(new Response(limitUploadStream(body, 5, onExceeded)).arrayBuffer()).rejects.toMatchObject({ status: 413 });
    expect(onExceeded).toHaveBeenCalledOnce();
  });

  it("forwards each byte without parsing or copying the full video", async () => {
    const onExceeded = vi.fn();
    const body = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(Uint8Array.from([1, 2])); controller.enqueue(Uint8Array.from([3, 4])); controller.close(); } });
    const result = await new Response(limitUploadStream(body, 4, onExceeded)).arrayBuffer();
    expect([...new Uint8Array(result)]).toEqual([1, 2, 3, 4]);
    expect(onExceeded).not.toHaveBeenCalled();
  });
});
