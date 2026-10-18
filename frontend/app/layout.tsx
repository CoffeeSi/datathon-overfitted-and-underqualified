import type { Metadata } from "next";
import "@fontsource-variable/manrope";
import "./globals.css";
import { AppHeader } from "@/components/app-header";

export const metadata: Metadata = {
  title: { default: "Overfitted and Underqualified", template: "%s · Overfitted and Underqualified" },
  description: "Рабочее место для внимательного просмотра видео и событий. ForteBank Datathon — Overfitted and Underqualified.",
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="ru" data-scroll-behavior="smooth"><body>
    <a className="skip-link" href="#main-content">Перейти к содержимому</a>
    <AppHeader />{children}
    <footer className="site-footer"><span>Overfitted and Underqualified / VIDEO REVIEW</span><span>ForteBank <span className="footer-separator">/</span> Datathon ’26</span></footer>
  </body></html>;
}
