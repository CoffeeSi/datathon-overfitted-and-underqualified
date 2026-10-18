"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Dialog } from "./ui/dialog";
import { Icon } from "./ui/icon";
export function AppHeader() {
  const pathname = usePathname();
  const [help, setHelp] = useState(false);
  return <>
    <header className="site-header"><div className="header-inner">
      <Link href="/" className="brand" aria-label="Overfitted and Underqualified — на главную"><span className="brand-symbol" aria-hidden="true"><i/><i/><i/><i/><b/></span><span className="brand-name">Overfitted and Underqualified</span><span className="brand-caption">VIDEO<br/>REVIEW</span></Link>
      <nav aria-label="Основная навигация"><Link className={pathname === "/" ? "nav-link active" : "nav-link"} href="/">Рабочее место</Link><button className="nav-link" onClick={() => setHelp(true)}>Как это работает <span className="tiny-arrow">↗</span></button></nav>
      <span className="header-edition">DATATHON <span>’26</span></span>
    </div></header>
    <Dialog open={help} onClose={() => setHelp(false)} title="Внимание к каждому моменту">
      <div className="help-content"><p>Overfitted and Underqualified помогает перейти от видео к событиям, которые стоит рассмотреть внимательнее.</p>
        <ol className="help-steps"><li><b>Загрузите запись</b><span>Выберите файл .mov. Видео передаётся настроенному серверу анализа.</span></li><li><b>Дождитесь результата</b><span>Можно сохранить номер задания и вернуться к нему позже.</span></li><li><b>Проверьте контекст</b><span>Перемещайтесь по таймкодам, рассматривайте кадры и читайте описание события.</span></li></ol>
        <p className="muted">Результат требует проверки человеком. Приложение не устанавливает личность сотрудника и не делает юридических выводов.</p>
        <div className="help-signature"><Icon name="eye"/> Overfitted and Underqualified / ForteBank Datathon</div>
      </div>
    </Dialog>
  </>;
}
