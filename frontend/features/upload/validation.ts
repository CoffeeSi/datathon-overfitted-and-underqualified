export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} КБ`;
  return `${(bytes / (1024 * 1024)).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} МБ`;
}

/** Browser MIME varies for MOV; the server verifies the container after upload. */
export function validateVideoFile(file: Pick<File, "name" | "size">, maxBytes: number): string | null {
  if (!/\.mov$/i.test(file.name)) return "Выберите видео в формате .mov. Переименование другого формата не меняет его содержимое.";
  if (file.size === 0) return "Этот файл пуст. Выберите исходное видео.";
  if (!Number.isFinite(file.size) || file.size < 0) return "Не удалось определить размер файла. Выберите его ещё раз.";
  if (file.size > maxBytes) return `Файл слишком большой. Допустимый размер — до ${formatFileSize(maxBytes)}.`;
  return null;
}
