import React from "react";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ReviewWorkspace } from "../features/review/review-workspace";
import { useAnalysisJob } from "../features/analysis-job/use-analysis-job";
import { ApiError, type Finding, type Job } from "../lib/contracts";
import { clearVideos, setVideo } from "../lib/local-session";
import { forgetRecentJobs } from "../features/recent-jobs/storage";

vi.mock("../features/analysis-job/use-analysis-job", () => ({ useAnalysisJob: vi.fn() }));

// Next integrates its search-param subscribers with the native History API.
// This adapter reproduces that public behavior without mounting a Next server.
vi.mock("next/navigation", async () => {
  const { useSyncExternalStore } = await import("react");
  const subscribe = (notify: () => void) => {
    window.addEventListener("test:location", notify);
    return () => window.removeEventListener("test:location", notify);
  };
  return {
    useSearchParams: () => new URLSearchParams(useSyncExternalStore(subscribe, () => window.location.search, () => "")),
  };
});

const id = "00000000-0000-0000-0000-000000000001";
const findings: Finding[] = [
  { type: "safe_key_left_in_lock", timecode_sec: 12, description: "У двери виден ключ. Проверьте контекст.", evidence_frames: [{ timestamp_sec: 11.5, image_base64: "/9j/AAAA" }, { timestamp_sec: 12.5, image_base64: "/9j/AAAB" }] },
  { type: "new_action_from_provider", timecode_sec: 25, description: "Описание нового типа от сервиса.", evidence_frames: [{ timestamp_sec: 25, image_base64: "/9j/AAAC" }] },
  { type: "cash_counted_without_client_present", timecode_sec: 25, description: "Вторая отметка в ту же секунду.", evidence_frames: [{ timestamp_sec: 25, image_base64: "/9j/AAAD" }] },
];
const retry = vi.fn();
const jobMock = vi.mocked(useAnalysisJob);
const ready = (overrides: Partial<ReturnType<typeof useAnalysisJob>> = {}): ReturnType<typeof useAnalysisJob> => ({
  job: { job_id: id, analysis_provider: "test", status: "done", verdict: "violation", violations: findings },
  error: null, loading: false, paused: false, elapsedSec: 0, retry, ...overrides,
});

describe("review workspace", () => {
  beforeEach(() => {
    clearVideos();
    localStorage.clear();
    forgetRecentJobs();
    retry.mockReset();
    jobMock.mockReset().mockReturnValue(ready());
    window.history.replaceState(null, "", `/review/${id}`);
    const originalReplace = window.history.replaceState.bind(window.history);
    vi.spyOn(window.history, "replaceState").mockImplementation((...args) => {
      originalReplace(...args);
      window.dispatchEvent(new Event("test:location"));
    });
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:local-video") });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
    Object.defineProperty(HTMLDialogElement.prototype, "showModal", { configurable: true, value: function(this: HTMLDialogElement) { this.setAttribute("open", ""); } });
    Object.defineProperty(HTMLDialogElement.prototype, "close", { configurable: true, value: function(this: HTMLDialogElement) { this.removeAttribute("open"); } });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("links marker/list selection, details, URL and local video seeking, including equal timestamps", async () => {
    setVideo(id, new File(["local"], "clip.mov"));
    render(<ReviewWorkspace jobId={id} />);
    const video = await screen.findByLabelText("Исходное видео") as HTMLVideoElement;
    Object.defineProperty(video, "duration", { configurable: true, value: 60 });
    fireEvent.loadedMetadata(video);
    expect(video.currentTime).toBe(12);

    const unknownMarker = screen.getByRole("button", { name: /^Отметка 2:/ });
    const nextMarker = screen.getByRole("button", { name: /^Отметка 3:/ });
    expect(unknownMarker.style.top).not.toBe(nextMarker.style.top);
    fireEvent.click(unknownMarker);
    expect(video.currentTime).toBe(25);
    expect(unknownMarker.getAttribute("aria-pressed")).toBe("true");
    expect(window.location.search).toContain("event=");
    const unknownDetail = screen.getByRole("region", { name: "Событие для проверки" });
    expect(within(unknownDetail).getByText("Описание нового типа от сервиса.")).toBeTruthy();
    fireEvent.click(within(unknownDetail).getByText("Данные события"));
    expect(within(unknownDetail).getByText("new_action_from_provider")).toBeTruthy();

    const navigation = screen.getByRole("complementary", { name: /События/ });
    const sameTimeRow = within(navigation).getByRole("button", { name: /Вторая отметка в ту же секунду/ });
    fireEvent.click(sameTimeRow);
    expect(sameTimeRow.getAttribute("aria-pressed")).toBe("true");
    expect(nextMarker.getAttribute("aria-pressed")).toBe("true");
    expect(unknownMarker.getAttribute("aria-pressed")).toBe("false");
    expect(screen.getByRole("region", { name: "Пересчёт без клиента в кадре" })).toBeTruthy();
    expect(video.currentTime).toBe(25);

    fireEvent.change(screen.getByRole("slider", { name: "Позиция видео" }), { target: { value: "15.5" } });
    expect(video.currentTime).toBe(15.5);
    video.currentTime = 31;
    fireEvent.timeUpdate(video);
    expect((screen.getByRole("slider", { name: "Позиция видео" }) as HTMLInputElement).value).toBe("31");
  });

  it("filters navigation without deleting events from the timeline and can clear the filter", () => {
    render(<ReviewWorkspace jobId={id} />);
    fireEvent.change(screen.getByRole("textbox", { name: "Поиск событий" }), { target: { value: "ничего такого нет" } });
    expect(screen.getByText("По этому запросу ничего нет.")).toBeTruthy();
    expect(screen.getAllByRole("button", { name: /^Отметка \d:/ })).toHaveLength(3);
    fireEvent.click(screen.getByRole("button", { name: "Сбросить фильтры" }));
    const navigation = screen.getByRole("complementary", { name: /События/ });
    expect(within(navigation).getByRole("button", { name: /Описание нового типа/ })).toBeTruthy();
  });

  it.each([
    ["no_violation", "Нарушения не обнаружены", "Список событий пуст."],
    ["insufficient_data", "Недостаточно данных", "Нужен дополнительный контекст."],
  ] as const)("renders an honest empty %s result", (verdict, title, emptyTitle) => {
    jobMock.mockReturnValue(ready({ job: { job_id: id, analysis_provider: "test", status: "done", verdict, violations: [] } }));
    render(<ReviewWorkspace jobId={id} />);
    expect(screen.getByRole("heading", { name: title })).toBeTruthy();
    expect(screen.getByRole("heading", { name: emptyTitle })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /^Отметка/ })).toBeNull();
    expect(screen.getByRole("button", { name: "Скачать JSON" })).toBeTruthy();
    expect((screen.getByRole("button", { name: "Следующее событие" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("keeps backend failed distinct from a missing job and offers explicit recovery", () => {
    const failed: Job = { job_id: id, analysis_provider: "test", status: "failed", error: { code: "decode_failed", message: "Не удалось прочитать видео." } };
    jobMock.mockReturnValue(ready({ job: failed }));
    const view = render(<ReviewWorkspace jobId={id} />);
    expect(screen.getByRole("heading", { name: "Не удалось разобрать запись." })).toBeTruthy();
    expect(screen.getByText("Код: decode_failed")).toBeTruthy();
    expect(screen.getByRole("link", { name: /Выбрать видео заново/ }).getAttribute("href")).toBe("/");
    expect(screen.queryByRole("button", { name: "Скачать JSON" })).toBeNull();

    jobMock.mockReturnValue(ready({ job: null, error: new ApiError("Задание неизвестно", 404), paused: true }));
    view.rerender(<ReviewWorkspace jobId={id} />);
    expect(screen.getByRole("alert").textContent).toContain("Задание не найдено");
    expect(screen.queryByRole("heading", { name: "Не удалось разобрать запись." })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Проверить снова" }));
    expect(retry).toHaveBeenCalledTimes(1);
  });

  it("labels mock results and never shows an invented processing percentage", () => {
    jobMock.mockReturnValue(ready({ job: { job_id: id, analysis_provider: "mock", status: "processing" }, elapsedSec: 42 }));
    render(<ReviewWorkspace jobId={id} />);
    expect(screen.getByText(/Результат синтетический/)).toBeTruthy();
    expect(screen.getByText("00:42")).toBeTruthy();
    expect(screen.getByText("Точный прогресс анализа сервис не сообщает.")).toBeTruthy();
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("recovers from an evidence image error and supports gallery buttons, arrows and close", async () => {
    render(<ReviewWorkspace jobId={id} />);
    fireEvent.click(screen.getByRole("button", { name: "Увеличить кадр выбранного события" }));
    const gallery = screen.getByRole("dialog", { name: "Кадры · Ключ в замке" });
    const first = within(gallery).getByRole("img", { name: "Кадры · Ключ в замке, 00:11.5" });
    fireEvent.error(first);
    expect(within(gallery).getByRole("img", { name: /кадр недоступен/ })).toBeTruthy();
    fireEvent.click(within(gallery).getByRole("button", { name: "Следующий кадр" }));
    expect(within(gallery).getByRole("img", { name: "Кадры · Ключ в замке, 00:12.5" }).getAttribute("src")).toContain("/9j/AAAB");
    expect((within(gallery).getByRole("button", { name: "Следующий кадр" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.keyDown(within(gallery).getByLabelText("Просмотр кадров. Стрелки влево и вправо переключают кадры."), { key: "ArrowLeft" });
    expect(within(gallery).getByRole("img", { name: "Кадры · Ключ в замке, 00:11.5" })).toBeTruthy();
    fireEvent(gallery, new Event("cancel", { cancelable: true }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Кадры · Ключ в замке" })).toBeNull());
  });

  it("navigates gallery frames immediately when focus is on the dialog close button", () => {
    render(<ReviewWorkspace jobId={id} />);
    fireEvent.click(screen.getByRole("button", { name: "Увеличить кадр выбранного события" }));
    const gallery = screen.getByRole("dialog", { name: "Кадры · Ключ в замке" });
    const close = within(gallery).getByRole("button", { name: "Закрыть" });
    close.focus();
    expect(document.activeElement).toBe(close);
    fireEvent.keyDown(close, { key: "ArrowRight" });
    expect(within(gallery).getByRole("img", { name: "Кадры · Ключ в замке, 00:12.5" })).toBeTruthy();
    expect(screen.getByRole("region", { name: "Ключ в замке" })).toBeTruthy();
    fireEvent.keyDown(close, { key: "ArrowLeft" });
    expect(within(gallery).getByRole("img", { name: "Кадры · Ключ в замке, 00:11.5" })).toBeTruthy();
  });

  it("leaves native player keyboard controls alone", async () => {
    setVideo(id, new File(["local"], "clip.mov"));
    render(<ReviewWorkspace jobId={id} />);
    const video = await screen.findByLabelText("Исходное видео") as HTMLVideoElement;
    Object.defineProperty(video, "duration", { configurable: true, value: 60 });
    fireEvent.loadedMetadata(video);
    const play = vi.spyOn(video, "play").mockResolvedValue();
    const right = new KeyboardEvent("keydown", { key: "ArrowRight", code: "ArrowRight", bubbles: true, cancelable: true });
    const space = new KeyboardEvent("keydown", { key: " ", code: "Space", bubbles: true, cancelable: true });
    fireEvent(video, right);
    fireEvent(video, space);
    expect(right.defaultPrevented).toBe(false);
    expect(space.defaultPrevented).toBe(false);
    expect(play).not.toHaveBeenCalled();
    expect(screen.getByRole("region", { name: "Ключ в замке" })).toBeTruthy();
  });

  it("does not promote a user-selected playback file to a verified source after remount", async () => {
    const view = render(<ReviewWorkspace jobId={id} />);
    await act(async () => {
      fireEvent.change(screen.getByLabelText("Выбрать исходное видео для просмотра"), { target: { files: [new File(["different"], "another.mov")] } });
    });
    expect(screen.getByText("Файл выбран локально. Соответствие заданию не проверено.")).toBeTruthy();
    expect(screen.getByLabelText("Исходное видео")).toBeTruthy();
    view.unmount();
    await act(async () => { render(<ReviewWorkspace jobId={id} />); });
    expect(screen.queryByText("Исходный файл доступен в этой вкладке.")).toBeNull();
    expect(screen.queryByLabelText("Исходное видео")).toBeNull();
  });
});
