/** Files live only in this tab's memory. No binary data enters persistent storage. */
const videos = new Map<string, File>();
const MAX_OPEN_FILES = 3;

export function setVideo(jobId: string, file: File): void {
  videos.delete(jobId);
  videos.set(jobId, file);
  while (videos.size > MAX_OPEN_FILES) {
    const oldest = videos.keys().next().value;
    if (oldest !== undefined) videos.delete(oldest);
  }
}

export function getVideo(jobId: string): File | undefined {
  return videos.get(jobId);
}

export function forgetVideo(jobId: string): void {
  videos.delete(jobId);
}

export function clearVideos(): void {
  videos.clear();
}
