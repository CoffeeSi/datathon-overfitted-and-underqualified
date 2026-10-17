export function timecode(value: number, precise = false) {
  if (!Number.isFinite(value) || value < 0) return "—:—";
  const total = Math.floor(value);
  const seconds = String(total % 60).padStart(2, "0");
  const minutes = String(Math.floor(total / 60) % 60).padStart(2, "0");
  const hours = Math.floor(total / 3600);
  const tenth = Math.floor(value * 10 + 1e-8) % 10;
  return `${hours ? `${String(hours).padStart(2, "0")}:` : ""}${minutes}:${seconds}${precise ? `.${tenth}` : ""}`;
}
