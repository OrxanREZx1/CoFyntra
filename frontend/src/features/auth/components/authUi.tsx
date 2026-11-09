'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useTransition } from 'react';
import { ArrowRight, Info } from 'lucide-react';
import { cn } from '@/lib/utils';
import { BACKEND_MODE } from '@/lib/config';
import { readDemoProfile } from '@/features/onboarding/profileStore';
import { startDemoSession, useDemoSession } from '../demoSession';
import { nextDemoRoute } from '../useDemoAccess';

export const primaryButton = cn(
  'w-full h-11 inline-flex items-center justify-center gap-2 rounded-xl text-sm font-semibold',
  'bg-[var(--color-primary)] text-[var(--color-on-primary)] hover:bg-[var(--color-primary-hover)]',
  'disabled:opacity-60 disabled:cursor-not-allowed transition-colors',
);

export function Spinner() {
  return <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" aria-hidden="true" />;
}

/** Honest explanation of what the demo auth does and doesn't do. */
export function DemoAuthNotice() {
  if (BACKEND_MODE === 'api') {
    return (
      <div className="flex gap-2.5 p-3 rounded-xl bg-[var(--color-warning-muted)] border border-[var(--color-warning)]/25" role="status">
        <Info className="w-4 h-4 mt-0.5 flex-shrink-0 text-[var(--color-warning)]" aria-hidden="true" />
        <p className="text-xs leading-relaxed text-[var(--color-text-secondary)]">
          <span className="font-semibold text-[var(--color-text-primary)]">Live authentication is not integrated.</span>{' '}
          No authentication API contract is available, so credentials will not be accepted or sent.
        </p>
      </div>
    );
  }

  return (
    <div className="flex gap-2.5 p-3 rounded-xl bg-[var(--color-surface-secondary)]/60 border border-[var(--color-border)]">
      <Info className="w-4 h-4 mt-0.5 flex-shrink-0 text-[var(--color-secondary)]" aria-hidden="true" />
      <p className="text-xs leading-relaxed text-[var(--color-text-secondary)]">
        <span className="font-semibold text-[var(--color-text-primary)]">Demo mode.</span> Accounts aren&apos;t available yet,
        so nothing here is sent to a server and your password is never stored. Only your name is kept in this browser to
        personalize the demo.
      </p>
    </div>
  );
}

/** Enter the demo without filling in any form. */
export function ExploreDemoButton() {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  if (BACKEND_MODE !== 'demo') return null;

  return (
    <button
      type="button"
      disabled={pending}
      onClick={() =>
        startTransition(() => {
          startDemoSession(null);
          router.push(nextDemoRoute(readDemoProfile() !== null));
        })
      }
      className="w-full h-11 inline-flex items-center justify-center gap-2 rounded-xl text-sm font-medium border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-border-active)] hover:bg-[var(--color-bg-hover)] disabled:opacity-60"
    >
      {pending && <Spinner />} Explore the demo without an account
    </button>
  );
}

/** Shown on auth pages when a demo session already exists in this browser. */
export function ExistingSessionBanner() {
  const session = useDemoSession();
  if (BACKEND_MODE !== 'demo') return null;
  if (!session) return null;
  const href = nextDemoRoute(readDemoProfile() !== null);
  return (
    <Link
      href={href}
      className="flex items-center justify-between gap-3 p-3 rounded-xl border border-[var(--color-secondary)]/30 bg-[var(--color-secondary-muted)] text-sm text-[var(--color-text-primary)] hover:border-[var(--color-secondary)]"
    >
      <span>
        You&apos;re already in the demo{session.displayName ? ` as ${session.displayName}` : ''}.
      </span>
      <span className="flex items-center gap-1 font-medium text-[var(--color-secondary)] whitespace-nowrap">
        Continue <ArrowRight className="w-4 h-4" aria-hidden="true" />
      </span>
    </Link>
  );
}

export function Divider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 text-xs text-[var(--color-text-muted)]" role="separator">
      <span className="h-px flex-1 bg-[var(--color-border)]" />
      {label}
      <span className="h-px flex-1 bg-[var(--color-border)]" />
    </div>
  );
}
