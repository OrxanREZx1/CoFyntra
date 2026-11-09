import { cn } from '@/lib/utils';

export const container = 'mx-auto w-full max-w-6xl px-4 sm:px-6 lg:px-8';

export const buttonPrimary = cn(
  'inline-flex items-center justify-center gap-2 h-11 px-5 rounded-xl text-sm font-semibold',
  'bg-[var(--color-primary)] text-[var(--color-on-primary)] hover:bg-[var(--color-primary-hover)] transition-colors',
);

export const buttonSecondary = cn(
  'inline-flex items-center justify-center gap-2 h-11 px-5 rounded-xl text-sm font-semibold',
  'border border-[var(--color-border)] text-[var(--color-text-primary)] bg-[var(--color-bg-elevated)]/60',
  'hover:border-[var(--color-border-active)] hover:bg-[var(--color-bg-hover)] transition-colors',
);

export const navLink =
  'text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] rounded-md transition-colors';

/** Small UI-mockup surface used in illustrations. */
export const mockCard =
  'rounded-2xl border border-[var(--color-border)] bg-[var(--color-bg-elevated)] shadow-[var(--shadow-card)]';
