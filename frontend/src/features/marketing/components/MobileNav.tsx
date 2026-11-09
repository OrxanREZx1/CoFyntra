'use client';

import Link from 'next/link';
import { useEffect, useId, useRef, useState } from 'react';
import { Menu, X } from 'lucide-react';
import { buttonPrimary, buttonSecondary } from '../styles';

const LINKS = [
  { href: '#how-it-works', label: 'How It Works' },
  { href: '#features', label: 'Features' },
];

/** Compact, accessible menu for small screens. */
export function MobileNav() {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
        buttonRef.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  return (
    <div className="md:hidden">
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls={panelId}
        aria-label={open ? 'Close menu' : 'Open menu'}
        className="w-9 h-9 inline-flex items-center justify-center rounded-lg text-[var(--color-text-primary)] hover:bg-[var(--color-bg-hover)]"
      >
        {open ? <X className="w-5 h-5" aria-hidden="true" /> : <Menu className="w-5 h-5" aria-hidden="true" />}
      </button>
      {open && (
        <div
          id={panelId}
          className="absolute inset-x-0 top-full border-b border-[var(--color-border)] bg-[var(--color-bg-base)] shadow-[var(--shadow-card)]"
        >
          <nav aria-label="Mobile" className="flex flex-col gap-1 px-4 py-4">
            {LINKS.map((l) => (
              <a
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                className="px-2 py-2.5 rounded-lg text-[15px] text-[var(--color-text-primary)] hover:bg-[var(--color-bg-hover)]"
              >
                {l.label}
              </a>
            ))}
            <div className="grid grid-cols-2 gap-2 pt-3">
              <Link href="/login" className={buttonSecondary} onClick={() => setOpen(false)}>
                Log In
              </Link>
              <Link href="/signup" className={buttonPrimary} onClick={() => setOpen(false)}>
                Get Started
              </Link>
            </div>
          </nav>
        </div>
      )}
    </div>
  );
}
