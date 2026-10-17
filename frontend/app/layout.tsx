import type { Metadata } from "next";
import "@fontsource-variable/manrope";
import "./globals.css";
import { AppHeader } from "@/components/app-header";

export const metadata: Metadata = {
  title: { default: "Фокус — видеопросмотр", template: "%s · Фокус" },
  description: "Рабочее место для внимательного просмотра видео и событий. ForteBank Datathon — Overfitted & Underqualified.",
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="ru" data-scroll-behavior="smooth"><body>
    <a className="skip-link" href="#main-content">Перейти к содержимому</a>
    <AppHeader />{children}
    <footer className="site-footer"><span>фокус<span className="accent">.</span> / VIDEO REVIEW</span><span>Overfitted & Underqualified <span className="footer-separator">/</span> Datathon ’26</span></footer>
  </body></html>;
}
