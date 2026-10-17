import Link from "next/link";
export default function NotFound() {
  return <main id="main-content" className="page-container empty-page"><span className="eyebrow">404 / ВНЕ КАДРА</span><h1>Такой страницы нет.</h1><p>Начните новый анализ или откройте задание по его номеру.</p><Link className="button button-primary" href="/">На рабочее место</Link></main>;
}
