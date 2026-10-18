import type { Finding, UnresolvedCandidate } from "@/lib/contracts";
import { timecode } from "@/lib/format";

export function ActionContext({ finding }: { finding: Finding }) {
  if (!finding.source_links?.length) return null;
  return <section aria-label="Связанные правила" className="event-technical">
    <h3>Основания для проверки</h3>
    <p>Ссылки связаны с распознанным действием. Применимость правила проверяет специалист; модель не устанавливает юридическое нарушение.</p>
    <ul>{finding.source_links.map((reference, index) => <li key={`${reference.url}-${index}`}>
      <a href={reference.url} target="_blank" rel="noopener noreferrer">{reference.source} · {reference.locator}</a>
    </li>)}</ul>
    {!!finding.applicability?.length && <><h4>Условия применимости</h4><ul>{finding.applicability.map((item) => <li key={item}>{item}</li>)}</ul></>}
    {!!finding.video_cannot_establish?.length && <><h4>По видео нельзя установить</h4><ul>{finding.video_cannot_establish.map((item) => <li key={item}>{item}</li>)}</ul></>}
    {finding.type_confidence !== undefined && <p>Оценка типа моделью: {(finding.type_confidence * 100).toFixed(0)}%. Это некалиброванный score, а не вероятность нарушения закона.</p>}
  </section>;
}

export function UnresolvedActions({ candidates }: { candidates?: UnresolvedCandidate[] }) {
  if (!candidates?.length) return null;
  return <section className="event-technical" aria-label="Неопределённые кандидаты">
    <h2>Нужна ручная проверка</h2>
    <p>Бинарный анализ выделил {candidates.length} окон, но распознаватель не подтвердил поддерживаемый тип. Это не означает отсутствие события.</p>
    <details><summary>Таймкоды неопределённых окон</summary>
      <ul>{candidates.map((candidate, index) => <li key={`${candidate.timestamp_sec}-${index}`}>
        {timecode(candidate.timestamp_sec, true)} · {candidate.reason === "background" ? "Низкие оценки известных действий" : "Тип не определён"}
      </li>)}</ul>
    </details>
  </section>;
}
