import Link from 'next/link';
import { ArrowLeft, GitMerge, Lock, Users } from 'lucide-react';
import { Logo } from '@/components/ui/Logo';
import { ThemeToggle } from '@/features/theme/ThemeToggle';

const POINTS = [
  { icon: Lock, title: 'Private by default', body: 'Each founder thinks out loud with their own AI.' },
  { icon: Users, title: 'Connected knowledge', body: 'Important findings are shared — with attribution.' },
  { icon: GitMerge, title: 'Decide together', body: 'Directions become official only with both approvals.' },
];

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-dvh grid lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      {/* Brand panel (desktop) */}
      <aside className="hidden lg:flex flex-col justify-between p-10 bg-[var(--color-surface-secondary)] border-r border-[var(--color-border)]">
        <Logo />
        <div className="max-w-sm">
          <p className="font-display text-3xl leading-tight text-[var(--color-text-primary)]">
            Think independently.
            <br />
            Build in sync.
          </p>
          <ul className="mt-8 space-y-5">
            {POINTS.map(({ icon: Icon, title, body }) => (
              <li key={title} className="flex gap-3">
                <span className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 bg-[var(--color-bg-base)] border border-[var(--color-border)]">
                  <Icon className="w-4 h-4 text-[var(--color-secondary)]" aria-hidden="true" />
                </span>
                <span>
                  <span className="block text-sm font-semibold text-[var(--color-text-primary)]">{title}</span>
                  <span className="block text-sm text-[var(--color-text-secondary)]">{body}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
        <p className="text-xs text-[var(--color-text-muted)]">Your conversations stay private. Your startup knowledge stays connected.</p>
      </aside>

      {/* Form column */}
      <main className="flex flex-col min-w-0">
        <div className="flex items-center justify-between gap-3 px-5 sm:px-8 h-16">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] rounded-lg"
          >
            <ArrowLeft className="w-4 h-4" aria-hidden="true" /> Back to home
          </Link>
          <div className="flex items-center gap-2">
            <Logo size="sm" className="lg:hidden" />
            <ThemeToggle />
          </div>
        </div>
        <div className="flex-1 flex items-start sm:items-center justify-center px-5 sm:px-8 pb-12 pt-4">
          <div className="w-full max-w-md">{children}</div>
        </div>
      </main>
    </div>
  );
}
