"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchJob } from "@/lib/client";
import { ApiError, type Job } from "@/lib/contracts";

const POLL_INTERVAL_MS = 2_000;
const MAX_BACKOFF_MS = 15_000;
const POLL_BUDGET_MS = 5 * 60_000;

interface JobState {
  jobId: string;
  job: Job | null;
  error: ApiError | null;
  loading: boolean;
  paused: boolean;
  elapsedSec: number;
}

function initialState(jobId: string): JobState {
  return { jobId, job: null, error: null, loading: true, paused: false, elapsedSec: 0 };
}

/** Elapsed time measures this page's observation, never the server's processing time. */
export function useAnalysisJob(jobId: string) {
  const [state, setState] = useState(() => initialState(jobId));
  const [attempt, setAttempt] = useState(0);
  const lifecycle = useRef<{ jobId: string; startedAt: number } | null>(null);
  const retry = useCallback(() => setAttempt((value) => value + 1), []);

  useEffect(() => {
    if (lifecycle.current?.jobId !== jobId) {
      lifecycle.current = { jobId, startedAt: Date.now() };
    }
    const startedAt = lifecycle.current.startedAt;
    const budgetDeadline = Date.now() + POLL_BUDGET_MS;
    let active = true;
    let halt: "terminal" | "manual" | "budget" | null = null;
    let failures = 0;
    let inFlight = false;
    let requestVersion = 0;
    let controller: AbortController | null = null;
    let nextPoll: ReturnType<typeof setTimeout> | undefined;
    let budgetTimer: ReturnType<typeof setTimeout> | undefined;
    let clockTimer: ReturnType<typeof setInterval> | undefined;

    const publish = (patch: Partial<JobState>) => {
      if (!active) return;
      setState((previous) => ({
        ...(previous.jobId === jobId ? previous : initialState(jobId)),
        ...patch,
      }));
    };

    const cancelRequest = () => {
      // A custom transport may settle after abort: the version guard still rejects it.
      requestVersion += 1;
      controller?.abort();
      controller = null;
      inFlight = false;
      clearTimeout(nextPoll);
    };

    const pauseForBudget = () => {
      if (!active || halt === "terminal" || halt === "manual") return;
      halt = "budget";
      cancelRequest();
      publish({
        loading: false,
        paused: true,
        error: new ApiError(
          "Автоматическая проверка приостановлена через 5 минут. Задание может продолжать работу на сервере. Возобновите проверку, чтобы получить результат.",
          0,
          "polling_timeout",
        ),
      });
    };

    const poll = async () => {
      if (!active || halt || document.hidden || inFlight) return;
      if (Date.now() >= budgetDeadline) { pauseForBudget(); return; }
      inFlight = true;
      const version = ++requestVersion;
      const request = new AbortController();
      controller = request;
      publish({ loading: true, paused: false });

      try {
        const job = await fetchJob(jobId, request.signal, startedAt);
        if (!active || request.signal.aborted || version !== requestVersion) return;
        failures = 0;
        publish({ job, error: null, loading: false });
        if (job.status === "done" || job.status === "failed") {
          halt = "terminal";
          clearTimeout(budgetTimer);
          clearInterval(clockTimer);
          publish({ elapsedSec: Math.floor((Date.now() - startedAt) / 1_000) });
        } else {
          nextPoll = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (cause) {
        if (!active || request.signal.aborted || version !== requestVersion) return;
        const error = cause instanceof ApiError
          ? cause
          : new ApiError("Не удалось проверить задание. Повторим запрос после восстановления связи.", 0, "network_error");
        publish({ error, loading: false });
        if (error.status === 401 || error.status === 403 || error.status === 404) {
          halt = "manual";
          clearTimeout(budgetTimer);
          publish({ paused: true });
        } else {
          failures += 1;
          const delay = Math.min(POLL_INTERVAL_MS * 2 ** Math.min(failures - 1, 4), MAX_BACKOFF_MS);
          nextPoll = setTimeout(poll, delay);
        }
      } finally {
        if (version === requestVersion) {
          inFlight = false;
          controller = null;
        }
      }
    };

    const onVisibilityChange = () => {
      if (halt) return;
      if (document.hidden) {
        cancelRequest();
        publish({ paused: true, loading: false });
      } else {
        publish({ paused: false });
        void poll();
      }
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    // Initialize asynchronously so a superseded effect cannot publish an old job.
    queueMicrotask(() => {
      if (!active) return;
      publish({ error: null, loading: !document.hidden, paused: document.hidden });
      clockTimer = setInterval(() => {
        publish({ elapsedSec: Math.floor((Date.now() - startedAt) / 1_000) });
      }, 1_000);
      budgetTimer = setTimeout(pauseForBudget, Math.max(0, budgetDeadline - Date.now()));
      void poll();
    });

    return () => {
      active = false;
      cancelRequest();
      clearTimeout(budgetTimer);
      clearInterval(clockTimer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [jobId, attempt]);

  // Hide the previous job immediately during navigation, before the effect runs.
  const current = state.jobId === jobId ? state : initialState(jobId);
  return {
    job: current.job,
    error: current.error,
    loading: current.loading,
    paused: current.paused,
    elapsedSec: current.elapsedSec,
    retry,
  };
}
