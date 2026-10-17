"use client";

import { useSyncExternalStore } from "react";
import type { Job } from "@/lib/contracts";

export interface RecentJob {
  jobId: string;
  filename: string;
  createdAt: string;
  status: Job["status"];
}

const STORAGE_KEY = "fokus.recent-jobs.v1";
const CHANGE_EVENT = "fokus:recent-jobs";
const EMPTY_JOBS: RecentJob[] = [];
let cachedRaw: string | null | undefined;
let cachedJobs: RecentJob[] = EMPTY_JOBS;
let memoryOnly = false;

function isRecentJob(value: unknown): value is RecentJob {
  if (!value || typeof value !== "object") return false;
  const entry = value as Record<string, unknown>;
  return typeof entry.jobId === "string" && /^[a-zA-Z0-9_-]{1,128}$/.test(entry.jobId)
    && typeof entry.filename === "string" && entry.filename.length <= 512
    && typeof entry.createdAt === "string" && Number.isFinite(Date.parse(entry.createdAt))
    && (entry.status === "processing" || entry.status === "done" || entry.status === "failed");
}

/** The cached snapshot is stable between changes, as useSyncExternalStore requires. */
export function readRecentJobs(): RecentJob[] {
  if (typeof window === "undefined") return EMPTY_JOBS;
  if (memoryOnly) return cachedJobs;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw === cachedRaw) return cachedJobs;
    cachedRaw = raw;
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    cachedJobs = Array.isArray(parsed) ? parsed.filter(isRecentJob).slice(0, 20).map(({ jobId, filename, createdAt, status }) => ({ jobId, filename, createdAt, status })) : [];
  } catch {
    cachedJobs = EMPTY_JOBS;
  }
  return cachedJobs;
}

function writeRecentJobs(jobs: RecentJob[]): void {
  cachedJobs = jobs;
  cachedRaw = JSON.stringify(jobs);
  try { window.localStorage.setItem(STORAGE_KEY, cachedRaw); } catch { memoryOnly = true; }
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

export function rememberJob(job: RecentJob): void {
  if (!isRecentJob(job)) return;
  const cleanJob = { jobId: job.jobId, filename: job.filename, createdAt: job.createdAt, status: job.status };
  writeRecentJobs([cleanJob, ...readRecentJobs().filter((item) => item.jobId !== job.jobId)].slice(0, 20));
}

export function updateRecentStatus(jobId: string, status: Job["status"]): void {
  const jobs = readRecentJobs();
  if (!jobs.some((job) => job.jobId === jobId && job.status !== status)) return;
  writeRecentJobs(jobs.map((job) => job.jobId === jobId ? { ...job, status } : job));
}

export function forgetRecentJobs(): void {
  writeRecentJobs([]);
}

function subscribe(onChange: () => void): () => void {
  const onStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY || event.key === null) onChange();
  };
  window.addEventListener(CHANGE_EVENT, onChange);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(CHANGE_EVENT, onChange);
    window.removeEventListener("storage", onStorage);
  };
}

export function useRecentJobs(): RecentJob[] {
  return useSyncExternalStore(subscribe, readRecentJobs, () => EMPTY_JOBS);
}
