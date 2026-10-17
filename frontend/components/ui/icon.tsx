import type { SVGProps } from "react";
const paths = {
  upload: <path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" />,
  "arrow-right": <path d="M4 12h16m-6-6 6 6-6 6" />,
  "arrow-left": <path d="M20 12H4m6-6-6 6 6 6" />,
  play: <path d="m9 5 11 7-11 7Z" />,
  pause: <path d="M8 5v14M16 5v14" />,
  check: <path d="m5 12 4 4L19 6" />,
  close: <path d="m6 6 12 12M6 18 18 6" />,
  "chevron-left": <path d="m15 5-7 7 7 7" />,
  "chevron-right": <path d="m9 5 7 7-7 7" />,
  copy: <><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M15 8V3H3v13h5"/></>,
  download: <path d="M12 3v13m-5-5 5 5 5-5M4 17v4h16v-4" />,
  search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/></>,
  plus: <path d="M12 5v14M5 12h14" />,
  film: <><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 8h4M3 16h4M17 8h4M17 16h4"/></>,
  clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
  refresh: <><path d="M20 7V3m0 4h-4M4 17v4m0-4h4"/><path d="M20 7a9 9 0 0 0-16 2m0 8a9 9 0 0 0 16-2"/></>,
  expand: <path d="M9 3H3v6m12-6h6v6M3 15v6h6m6 0h6v-6" />,
  keyboard: <><rect x="2" y="5" width="20" height="14" rx="2"/><path d="M6 9h1m4 0h1m4 0h1M6 12h1m4 0h1m4 0h1M7 16h10"/></>,
  alert: <><path d="m12 3 10 18H2Z"/><path d="M12 9v4m0 3h.01"/></>,
  eye: <><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></>,
  file: <><path d="M14 2H5v20h14V7Z"/><path d="M14 2v6h5M8 13h8m-8 4h5"/></>,
  folder: <path d="M3 6V4h6l2 3h10v13H3Z" />,
  more: <><circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/></>,
  external: <path d="M14 3h7v7m0-7L10 14M10 3H3v18h18v-7" />,
} as const;
export type IconName = keyof typeof paths;
export function Icon({ name, size = 20, ...props }: SVGProps<SVGSVGElement> & { name: IconName; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>;
}
