'use client';

import { Moon, Sun } from 'lucide-react';
import { cn } from '@/lib/utils';
import { setTheme, useTheme } from './theme';

/** Accessible light/dark switch. Renders a neutral placeholder until the theme is known. */
export function ThemeToggle({ className }: { className?: string }) {
  const theme = useTheme();
  const isDark = theme === 'dark';

  return (
    <button
      type="button"
      role="switch"
      aria-checked={theme === undefined ? undefined : isDark}
      aria-label="Dark mode"
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      className={cn(
        'inline-flex items-center justify-center w-8 h-8 rounded-lg',
        'text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-hover)]',
        className,
      )}
    >
      {theme === undefined ? (
        <span className="w-4 h-4" aria-hidden="true" />
      ) : isDark ? (
        <Sun className="w-4 h-4" aria-hidden="true" />
      ) : (
        <Moon className="w-4 h-4" aria-hidden="true" />
      )}
    </button>
  );
}
