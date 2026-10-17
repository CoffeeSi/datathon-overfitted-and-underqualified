"use client";
import { useEffect, useRef, type ReactNode, type KeyboardEventHandler } from "react";
import { Icon } from "./icon";
export function Dialog({ open, onClose, title, children, className = "", onKeyDown }: { open: boolean; onClose: () => void; title: string; children: ReactNode; className?: string; onKeyDown?: KeyboardEventHandler<HTMLDialogElement> }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog || !open) return;
    dialog.showModal();
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { dialog.close(); document.body.style.overflow = previous; };
  }, [open]);
  return <dialog ref={ref} className={`dialog ${className}`} aria-label={title} onCancel={onClose} onKeyDown={onKeyDown}
    onClick={(event) => { if (event.target === ref.current) { const r = ref.current.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) onClose(); } }}>
    <div className="dialog-heading"><h2>{title}</h2><button className="icon-button" aria-label="Закрыть" onClick={onClose}><Icon name="close" /></button></div>
    {open && children}
  </dialog>;
}
