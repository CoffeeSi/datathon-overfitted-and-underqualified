import React from "react";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UploadWorkspace } from "../features/upload/upload-workspace";
import { validateVideoFile } from "../features/upload/validation";
import { forgetRecentJobs, readRecentJobs } from "../features/recent-jobs/storage";
import { clearVideos, getVideo } from "../lib/local-session";
import { ApiError, type ServiceStatus } from "../lib/contracts";
import { fetchService, uploadVideo } from "../lib/client";

const { push } = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));
vi.mock("../lib/client", () => ({ fetchService: vi.fn(), uploadVideo: vi.fn() }));

const service: ServiceStatus = { health: "online", ready: true, provider: "test", maxUploadBytes: 1024, demoEnabled: false };
const fetchMock = vi.mocked(fetchService);
const uploadMock = vi.mocked(uploadVideo);
const createUrl = vi.fn();
const revokeUrl = vi.fn();

function choose(file: File) {
  fireEvent.change(screen.getByLabelText("Выбрать видео MOV"), { target: { files: [file] } });
}

describe("upload validation", () => {
  it("allows MOV with uncertain browser MIME and enforces inclusive byte limit", () => {
    expect(validateVideoFile({ name: "clip.MOV", size: 1024 }, 1024)).toBeNull();
    expect(validateVideoFile({ name: "clip.mov", size: 1025 }, 1024)).toContain("слишком большой");
    expect(validateVideoFile({ name: "clip.mp4", size: 1 }, 1024)).toContain(".mov");
    expect(validateVideoFile({ name: "clip.mov", size: 0 }, 1024)).toContain("пуст");
  });
});

describe("upload workflow", () => {
  beforeEach(() => {
    localStorage.clear();
    forgetRecentJobs();
    clearVideos();
    push.mockReset();
    fetchMock.mockReset().mockResolvedValue(service);
    uploadMock.mockReset();
    createUrl.mockReset().mockImplementation((file: File) => `blob:test-${file.name}`);
    revokeUrl.mockReset();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createUrl });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeUrl });
  });

  afterEach(cleanup);

  it("keeps browser-unsupported MOV eligible for analysis and revokes its preview", async () => {
    const view = render(<UploadWorkspace />);
    await screen.findByText("Сервис готов к анализу");
    choose(new File(["mov"], "clip.mov", { type: "video/quicktime" }));
    fireEvent.error(screen.getByLabelText("Предпросмотр clip.mov"));
    expect(screen.getByText("Этот кодек не воспроизводится в браузере.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Начать анализ" }) as HTMLButtonElement).disabled).toBe(false);
    view.unmount();
    expect(revokeUrl).toHaveBeenCalledWith("blob:test-clip.mov");
  });

  it("blocks unavailable provider without blocking local file selection", async () => {
    fetchMock.mockResolvedValue({ ...service, ready: false, provider: "unconfigured" });
    render(<UploadWorkspace />);
    await screen.findByText("Анализатор пока не подключён");
    choose(new File(["mov"], "clip.mov"));
    expect(screen.getByText("clip.mov")).toBeTruthy();
    const button = screen.getByRole("button", { name: "Начать анализ" }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    fireEvent.click(button);
    expect(uploadMock).not.toHaveBeenCalled();
  });

  it("submits once, stores only metadata and transfers the file in memory", async () => {
    let accept!: (value: { job_id: string }) => void;
    uploadMock.mockImplementation(() => new Promise((resolve) => { accept = resolve; }));
    const view = render(<UploadWorkspace />);
    await screen.findByText("Сервис готов к анализу");
    const file = new File(["private-content"], "clip.mov");
    choose(file);
    const start = screen.getByRole("button", { name: "Начать анализ" });
    act(() => { fireEvent.click(start); fireEvent.click(start); });
    expect(uploadMock).toHaveBeenCalledTimes(1);
    const id = "00000000-0000-0000-0000-000000000001";
    await act(async () => { accept({ job_id: id }); });
    expect(push).toHaveBeenCalledWith(`/review/${id}`);
    expect(screen.getByRole("link", { name: /clip.mov/ }).getAttribute("href")).toBe(`/review/${id}`);
    expect(getVideo(id)).toBe(file);
    expect(Object.keys(readRecentJobs()[0]).sort()).toEqual(["createdAt", "filename", "jobId", "status"]);
    expect(JSON.stringify(localStorage)).not.toContain("private-content");
    expect(revokeUrl).toHaveBeenCalledWith("blob:test-clip.mov");
    view.unmount();
    clearVideos();
    render(<UploadWorkspace />);
    expect(await screen.findByRole("link", { name: /clip.mov/ })).toBeTruthy();
    expect(screen.getByRole("link", { name: /clip.mov/ }).getAttribute("href")).toBe(`/review/${id}`);
  });

  it("does not retry an ambiguous upload failure", async () => {
    uploadMock.mockRejectedValue(new ApiError("Приём не подтверждён", 0, "network_error"));
    render(<UploadWorkspace />);
    await screen.findByText("Сервис готов к анализу");
    choose(new File(["mov"], "clip.mov"));
    fireEvent.click(screen.getByRole("button", { name: "Начать анализ" }));
    await screen.findByRole("alert");
    await waitFor(() => expect((screen.getByRole("button", { name: "Начать анализ" }) as HTMLButtonElement).disabled).toBe(false));
    expect(uploadMock).toHaveBeenCalledTimes(1);
    expect(push).not.toHaveBeenCalled();
    expect(readRecentJobs()).toEqual([]);
  });
});
